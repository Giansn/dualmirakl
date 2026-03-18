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

from simulation.scenario import ScenarioConfig, RoleConfig, ArchetypeProfile

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


def _safe_format(template: str, variables: dict) -> str:
    """Format a template string, leaving unknown {variables} as-is."""
    def replacer(match):
        key = match.group(1)
        if key in variables:
            return str(variables[key])
        return match.group(0)  # leave as {key}
    return re.sub(r'\{(\w+)\}', replacer, template)


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
    ) -> dict[str, list[AgentSpec]]:
        """
        Build all agent specs from config.

        Returns a dict:
          {
            "observers": [AgentSpec, ...],
            "environment": [AgentSpec],
            "participants": [AgentSpec, ...],
          }
        """
        if rng is None:
            rng = np.random.RandomState(42)

        result: dict[str, list[AgentSpec]] = {
            "observers": [],
            "environment": [],
            "participants": [],
        }

        domain_context = config.domain_context

        for role in config.agents.roles:
            if role.type == "observer":
                spec = AgentSpec(role.id, role, domain_context=domain_context)
                result["observers"].append(spec)

            elif role.type == "environment":
                spec = AgentSpec(role.id, role, domain_context=domain_context)
                result["environment"].append(spec)

            elif role.type == "participant":
                count = role.count or 1
                profiles = _assign_profiles(
                    count, config.archetypes.profiles,
                    config.archetypes.distribution, rng,
                )
                for i in range(count):
                    agent_id = f"participant_{i}"
                    profile = profiles[i] if i < len(profiles) else None
                    spec = AgentSpec(
                        agent_id, role, profile=profile,
                        domain_context=domain_context,
                    )
                    result["participants"].append(spec)

        return result


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
