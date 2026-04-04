"""
Generic agent factory for dualmirakl.

Builds agent instances from ScenarioConfig. Replaces the hardcoded
agent_rolesv3.py persona definitions with config-driven agent creation.

Phase 2 of the general-purpose refactoring.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import numpy as np

from simulation.config.scenario import ScenarioConfig, RoleConfig, ArchetypeProfile

logger = logging.getLogger(__name__)


class AgentSpec:
    """
    A fully resolved agent specification ready for use by the simulation.

    Created by AgentFactory from ScenarioConfig. Holds the role config,
    assigned archetype profile, and provides prompt rendering.
    """

    def __init__(
        self,
        agent_id: str,
        role: RoleConfig,
        profile: Optional[ArchetypeProfile] = None,
        domain_context: str = "",
    ):
        self.agent_id = agent_id
        self.role = role
        self.slot = role.slot
        self.type = role.type
        self.max_tokens = role.max_tokens
        self.system_prompt_template = role.system_prompt
        self.profile = profile
        self.domain_context = domain_context

    def render_prompt(self, tick_state: Optional[dict] = None) -> str:
        """
        Render the system prompt template with available variables.

        Variables are filled from (in priority order):
        1. tick_state dict (runtime values)
        2. archetype profile properties
        3. archetype metadata (label, description)
        4. built-in values (agent_name, domain_context)

        Unknown variables are left as-is (no KeyError).
        """
        variables: dict[str, Any] = {}

        # Built-ins
        variables["agent_name"] = self.agent_id
        variables["domain_context"] = self.domain_context

        # Archetype profile
        if self.profile:
            variables["archetype_label"] = self.profile.label
            variables["archetype_profile"] = self.profile.description
            variables.update(self.profile.properties)

        # Runtime tick state (highest priority — can override profile values)
        if tick_state:
            variables.update(tick_state)

        # Safe format: leave unknown {variables} untouched
        return _safe_format(self.system_prompt_template, variables)

    @property
    def backend(self) -> str:
        """Backend name derived from slot (for orchestrator routing)."""
        return self.slot

    def __repr__(self) -> str:
        profile_str = f" profile={self.profile.id}" if self.profile else ""
        return f"AgentSpec({self.agent_id}, {self.type}, {self.slot}{profile_str})"


# ── Archetype-driven parameter sampling ──────────────────────────────────────
ARCHETYPE_BETA_PARAMS: dict[str, tuple[float, float]] = {
    "high":   (5.0, 2.0),   # mode=0.80
    "medium": (3.0, 3.0),   # mode=0.50
    "low":    (2.0, 5.0),   # mode=0.20
}
DEFAULT_SUSCEPTIBILITY_BETA = (2.0, 3.0)
DEFAULT_RESILIENCE_BETA = (2.0, 5.0)


def sample_agent_params(
    profile: Optional[ArchetypeProfile],
    rng: np.random.RandomState,
) -> dict[str, float]:
    """Sample susceptibility and resilience from Beta distributions
    parameterized by the archetype profile's property levels."""
    susc_level = None
    if profile and hasattr(profile, "properties") and "susceptibility" in profile.properties:
        susc_level = str(profile.properties["susceptibility"]).lower().strip()
    susc_a, susc_b = ARCHETYPE_BETA_PARAMS.get(susc_level, DEFAULT_SUSCEPTIBILITY_BETA)

    res_level = None
    if profile and hasattr(profile, "properties") and "resilience" in profile.properties:
        res_level = str(profile.properties["resilience"]).lower().strip()
    res_a, res_b = ARCHETYPE_BETA_PARAMS.get(res_level, DEFAULT_RESILIENCE_BETA)

    return {
        "susceptibility": float(rng.beta(susc_a, susc_b)),
        "resilience": float(rng.beta(res_a, res_b)),
    }


def _safe_format(template: str, variables: dict) -> str:
    """Format a template string, leaving unknown {variables} as-is."""
    def replacer(match):
        key = match.group(1)
        if key in variables:
            return str(variables[key])
        return match.group(0)  # leave as {key}
    return re.sub(r'\{(\w+)\}', replacer, template)


class AgentSet:
    """
    Lightweight container for AgentSpec instances (Mesa-inspired).

    Supports filtering by slot/type, aggregation, and iteration.
    Returned by AgentFactory instead of plain dict-of-lists.

    Usage:
        agents = AgentFactory.from_config(config)
        participants = agents.by_type("participant")
        authority_agents = agents.by_slot("authority")
        all_ids = agents.agg("agent_id")
    """

    def __init__(self, specs: Optional[list[AgentSpec]] = None):
        self._specs: list[AgentSpec] = list(specs or [])

    def by_slot(self, slot: str) -> AgentSet:
        """Filter to agents on a specific GPU slot."""
        return AgentSet([s for s in self._specs if s.slot == slot])

    def by_type(self, agent_type: str) -> AgentSet:
        """Filter to agents of a specific type (observer, participant, environment)."""
        return AgentSet([s for s in self._specs if s.type == agent_type])

    def by_id(self, agent_id: str) -> Optional[AgentSpec]:
        """Get a single agent by ID, or None."""
        for s in self._specs:
            if s.agent_id == agent_id:
                return s
        return None

    def select(self, filter_func) -> AgentSet:
        """Filter with a custom predicate function."""
        return AgentSet([s for s in self._specs if filter_func(s)])

    def agg(self, attr: str) -> list:
        """Collect an attribute from all agents into a list."""
        return [getattr(s, attr) for s in self._specs]

    def ids(self) -> list[str]:
        """Shorthand for agg('agent_id')."""
        return self.agg("agent_id")

    def to_list(self) -> list[AgentSpec]:
        """Return the underlying list."""
        return list(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs)

    def __getitem__(self, idx):
        return self._specs[idx]

    def __repr__(self) -> str:
        types = {}
        for s in self._specs:
            types[s.type] = types.get(s.type, 0) + 1
        parts = [f"{t}={n}" for t, n in sorted(types.items())]
        return f"AgentSet({', '.join(parts)}, total={len(self)})"


class AgentFactory:
    """
    Creates AgentSpec instances from a ScenarioConfig.

    Handles:
    - Single-instance roles (observer, environment)
    - Multi-instance participant roles (count > 1)
    - Archetype profile assignment based on distribution
    """

    @staticmethod
    def from_config(
        config: ScenarioConfig,
        rng: Optional[np.random.RandomState] = None,
    ) -> AgentSet:
        """
        Build all agent specs from config. Returns an AgentSet.

        Legacy dict access:
          agents.by_type("observer")   replaces  result["observers"]
          agents.by_type("participant") replaces  result["participants"]
          agents.by_type("environment") replaces  result["environment"]
        """
        if rng is None:
            rng = np.random.RandomState(42)

        specs: list[AgentSpec] = []
        domain_context = config.domain_context

        for role in config.agents.roles:
            if role.type == "observer":
                specs.append(AgentSpec(role.id, role, domain_context=domain_context))

            elif role.type == "environment":
                specs.append(AgentSpec(role.id, role, domain_context=domain_context))

            elif role.type == "participant":
                count = role.count or 1
                profiles = _assign_profiles(
                    count, config.archetypes.profiles,
                    config.archetypes.distribution, rng,
                )
                for i in range(count):
                    agent_id = f"participant_{i}"
                    profile = profiles[i] if i < len(profiles) else None
                    specs.append(AgentSpec(
                        agent_id, role, profile=profile,
                        domain_context=domain_context,
                    ))

        return AgentSet(specs)


def _assign_profiles(
    count: int,
    profiles: list[ArchetypeProfile],
    distribution: dict[str, float],
    rng: np.random.RandomState,
) -> list[Optional[ArchetypeProfile]]:
    """
    Assign archetype profiles to N participants based on distribution fractions.

    Uses deterministic allocation: round(fraction * count) per profile,
    with random assignment of remainders.
    """
    if not profiles or not distribution:
        return [None] * count

    # Build profile lookup
    profile_map = {p.id: p for p in profiles}

    # Deterministic allocation
    allocated: list[ArchetypeProfile] = []
    remainder_pool: list[ArchetypeProfile] = []

    for profile_id, fraction in distribution.items():
        profile = profile_map.get(profile_id)
        if profile is None:
            logger.warning(f"Profile '{profile_id}' in distribution not found, skipping")
            continue
        n = int(round(fraction * count))
        allocated.extend([profile] * n)
        remainder_pool.append(profile)

    # Handle over/under allocation due to rounding
    while len(allocated) < count and remainder_pool:
        allocated.append(rng.choice(remainder_pool))
    allocated = allocated[:count]

    # Shuffle so profiles aren't grouped by ID
    rng.shuffle(allocated)

    return allocated
