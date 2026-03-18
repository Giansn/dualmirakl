"""
ScenarioConfig — single source of truth for domain-specific simulation configuration.

Loads, validates, and provides typed access to scenario.yaml files.
Part of the dualmirakl general-purpose refactoring (Phase 1).

Usage:
    # Load and validate
    config = ScenarioConfig.load("scenarios/social_dynamics.yaml")

    # Dry-run validation (zero GPU)
    python -m simulation.scenario validate scenarios/foo.yaml

    # Programmatic
    config = ScenarioConfig.from_dict({...})
    report = config.validate()
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO = "scenarios/social_dynamics.yaml"

# Built-in prompt variables that are always available (filled by the engine)
BUILTIN_VARIABLES = frozenset({
    "domain_context",
    "environment_state",
    "tick_state",
    "agent_name",
    "archetype_label",
    "archetype_profile",
    "available_actions",
    "current_stimulus",
    "tick",
    "nudge",
    "memory_context",
})


# ── Pydantic models for typed YAML sections ──────────────────────────────────

class MetaConfig(BaseModel):
    name: str
    version: str = "1.0"
    description: str = ""


class RoleConfig(BaseModel):
    id: str
    slot: str  # "authority" | "swarm"
    type: str  # "observer" | "participant" | "environment"
    system_prompt: str
    max_tokens: int = 256
    count: Optional[int] = None  # only for participant templates

    @field_validator("slot")
    @classmethod
    def validate_slot(cls, v):
        if v not in ("authority", "swarm"):
            raise ValueError(f"slot must be 'authority' or 'swarm', got '{v}'")
        return v

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v not in ("observer", "participant", "environment"):
            raise ValueError(f"type must be 'observer', 'participant', or 'environment', got '{v}'")
        return v


class AgentsConfig(BaseModel):
    roles: list[RoleConfig]


class ArchetypeProfile(BaseModel):
    id: str
    label: str
    description: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)


class ArchetypesConfig(BaseModel):
    profiles: list[ArchetypeProfile] = Field(default_factory=list)
    distribution: dict[str, float] = Field(default_factory=dict)

    @field_validator("distribution")
    @classmethod
    def validate_distribution_values(cls, v):
        if v:
            for key, val in v.items():
                if not 0.0 <= val <= 1.0:
                    raise ValueError(f"Distribution value for '{key}' must be 0.0-1.0, got {val}")
        return v


class ActionInstance(BaseModel):
    id: str
    schema_name: str = Field(alias="schema", default="binary_decision")
    prompt_fragment: str = ""


class ActionsConfig(BaseModel):
    schemas: list[str] = Field(default_factory=list)
    instances: list[ActionInstance] = Field(default_factory=list)


class DistributionConfig(BaseModel):
    type: str = "beta"
    params: list[float] = Field(default_factory=lambda: [2.0, 5.0])


class ScoringConfig(BaseModel):
    mode: str = "ema"
    distributions: dict[str, DistributionConfig] = Field(default_factory=dict)
    parameters: dict[str, float] = Field(default_factory=dict)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in ("ema", "logistic", "custom"):
            raise ValueError(f"scoring mode must be 'ema', 'logistic', or 'custom', got '{v}'")
        return v


class TransitionRule(BaseModel):
    from_profile: str = Field(alias="from")
    to_profile: str = Field(alias="to")
    function: str
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class MemoryConfig(BaseModel):
    enabled: bool = True
    max_entries_per_agent: int = 20
    dedup_threshold: float = 0.85
    summary_interval: int = 10


class SafetyConfig(BaseModel):
    enabled: bool = True
    action_allowlist: list[str] = Field(default_factory=list)
    max_retries_on_invalid: int = 2
    fallback_action: str = ""


class ContextCategory(BaseModel):
    id: str
    description: str
    required: bool = True


class FlameConfig(BaseModel):
    enabled: bool = False
    population_size: int = 10000
    kappa: float = 0.05
    influencer_weight: float = 0.8
    sub_steps: int = 4


class EnvironmentConfig(BaseModel):
    tick_count: int = 100
    tick_unit: str = "step"
    initial_state: dict[str, Any] = Field(default_factory=dict)


# ── Main ScenarioConfig ──────────────────────────────────────────────────────

class ScenarioConfig(BaseModel):
    """
    Complete scenario configuration loaded from YAML.
    Single source of truth for all domain-specific simulation parameters.

    Frozen after creation: config cannot be mutated during a simulation run.
    Use replicate(seed) for parameter sweeps with different seeds.
    """
    model_config = {"frozen": True}

    meta: MetaConfig
    agents: AgentsConfig
    archetypes: ArchetypesConfig = Field(default_factory=lambda: ArchetypesConfig(profiles=[], distribution={}))
    actions: ActionsConfig = Field(default_factory=ActionsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    transitions: list[TransitionRule] = Field(default_factory=list)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    context_categories: list[ContextCategory] = Field(default_factory=list)
    flame: FlameConfig = Field(default_factory=FlameConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    # ── Loaders ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str) -> ScenarioConfig:
        """Load from YAML file. Raises FileNotFoundError or ValidationError."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")
        with open(p) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Scenario file must be a YAML mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> ScenarioConfig:
        """Create from a dictionary (for API usage)."""
        return cls.model_validate(data)

    def replicate(self, seed: Optional[int] = None, **overrides) -> ScenarioConfig:
        """
        Create a copy with a different seed or parameter overrides.
        Used for SA sweeps and multi-seed runs. Since ScenarioConfig is frozen,
        this is the only way to create a variant.

        Args:
            seed: New random seed (stored in environment.initial_state["seed"])
            **overrides: Nested overrides like scoring_alpha=0.2
        """
        data = self.model_dump(by_alias=True)
        if seed is not None:
            data.setdefault("environment", {}).setdefault("initial_state", {})["seed"] = seed
        for key, value in overrides.items():
            if key.startswith("scoring_"):
                param = key[len("scoring_"):]
                data.setdefault("scoring", {}).setdefault("parameters", {})[param] = value
            elif key.startswith("flame_"):
                param = key[len("flame_"):]
                data.setdefault("flame", {})[param] = value
            elif key.startswith("memory_"):
                param = key[len("memory_"):]
                data.setdefault("memory", {})[param] = value
        return ScenarioConfig.from_dict(data)

    # ── Validation ────────────────────────────────────────────────────────

    def validate_scenario(self, strict: bool = True) -> dict:
        """
        Deep validation beyond Pydantic type checking.
        Returns a report dict with 'errors' and 'warnings' lists.
        If strict=True, raises ValueError on first error.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Distribution profile IDs must match defined profiles
        profile_ids = {p.id for p in self.archetypes.profiles}
        for dist_id in self.archetypes.distribution:
            if dist_id not in profile_ids:
                errors.append(
                    f"Distribution references undefined profile '{dist_id}'. "
                    f"Defined profiles: {sorted(profile_ids)}"
                )

        # 2. Distribution fractions must sum to ~1.0
        if self.archetypes.distribution:
            total = sum(self.archetypes.distribution.values())
            if abs(total - 1.0) > 0.01:
                errors.append(
                    f"Archetype distribution fractions sum to {total:.3f}, expected 1.0"
                )

        # 3. Transition rules reference valid profiles
        for rule in self.transitions:
            if rule.from_profile not in profile_ids and profile_ids:
                errors.append(
                    f"Transition from '{rule.from_profile}' references undefined profile"
                )
            if rule.to_profile not in profile_ids and profile_ids:
                errors.append(
                    f"Transition to '{rule.to_profile}' references undefined profile"
                )

        # 4. Transition function names are resolvable
        for rule in self.transitions:
            try:
                from simulation.transitions import get_transition
                get_transition(rule.function)
            except (ImportError, KeyError) as e:
                errors.append(
                    f"Transition function '{rule.function}' not found: {e}"
                )

        # 5. Prompt template variables have sources
        known_vars = set(BUILTIN_VARIABLES)
        # Add archetype property keys
        for profile in self.archetypes.profiles:
            known_vars.update(profile.properties.keys())
        # Add environment initial_state keys
        known_vars.update(self.environment.initial_state.keys())

        for role in self.agents.roles:
            template_vars = set(re.findall(r'\{(\w+)\}', role.system_prompt))
            unknown = template_vars - known_vars
            if unknown:
                warnings.append(
                    f"Role '{role.id}' prompt references unknown variables: {sorted(unknown)}. "
                    f"These must be provided at runtime."
                )

        # 6. At least one participant role defined
        participant_roles = [r for r in self.agents.roles if r.type == "participant"]
        if not participant_roles:
            errors.append("No participant role defined in agents.roles")

        # 7. Safety allowlist references valid action instance IDs
        action_ids = {a.id for a in self.actions.instances}
        for allowed in self.safety.action_allowlist:
            if allowed not in action_ids and action_ids:
                warnings.append(
                    f"Safety allowlist entry '{allowed}' doesn't match any action instance ID"
                )

        # 8. Participant count defined
        for role in participant_roles:
            if role.count is None or role.count < 1:
                warnings.append(
                    f"Participant role '{role.id}' has no count specified, defaulting to 1"
                )

        report = {"errors": errors, "warnings": warnings, "valid": len(errors) == 0}

        if strict and errors:
            raise ValueError(
                f"Scenario validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        return report

    # ── Convenience accessors ─────────────────────────────────────────────

    def get_role(self, role_id: str) -> Optional[RoleConfig]:
        """Get a role config by ID."""
        for role in self.agents.roles:
            if role.id == role_id:
                return role
        return None

    def get_profile(self, profile_id: str) -> Optional[ArchetypeProfile]:
        """Get an archetype profile by ID."""
        for profile in self.archetypes.profiles:
            if profile.id == profile_id:
                return profile
        return None

    def participant_count(self) -> int:
        """Total number of participant instances to spawn."""
        return sum(
            r.count or 1
            for r in self.agents.roles
            if r.type == "participant"
        )

    def scoring_param(self, name: str, default: float = 0.0) -> float:
        """Get a scoring parameter by name with default."""
        return self.scoring.parameters.get(name, default)

    @property
    def domain_context(self) -> str:
        """One-line domain context for prompt injection."""
        return self.meta.description or self.meta.name


# ── CLI: dry-run validator ────────────────────────────────────────────────────

def _cli_validate(path: str) -> int:
    """Validate a scenario file and print a report. Returns exit code."""
    print(f"\n-- validating: {path} --\n")

    try:
        config = ScenarioConfig.load(path)
    except FileNotFoundError as e:
        print(f"  FAIL: {e}")
        return 1
    except yaml.YAMLError as e:
        print(f"  FAIL: YAML parse error:\n  {e}")
        return 1
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    print(f"  Scenario: {config.meta.name} v{config.meta.version}")
    print(f"  Description: {config.meta.description}")
    print(f"  Roles: {len(config.agents.roles)}")
    print(f"  Profiles: {len(config.archetypes.profiles)}")
    print(f"  Actions: {len(config.actions.instances)}")
    print(f"  Transitions: {len(config.transitions)}")
    print(f"  Participants: {config.participant_count()}")
    print(f"  Scoring: {config.scoring.mode}")
    print(f"  Memory: {'enabled' if config.memory.enabled else 'disabled'}")
    print(f"  Safety: {'enabled' if config.safety.enabled else 'disabled'}")
    print(f"  FLAME: {'enabled' if config.flame.enabled else 'disabled'}")
    print(f"  Ticks: {config.environment.tick_count}")
    print()

    # Run deep validation (non-strict to collect all issues)
    report = config.validate_scenario(strict=False)

    if report["warnings"]:
        print(f"  WARNINGS ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"    [!] {w}")
        print()

    if report["errors"]:
        print(f"  ERRORS ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"    [X] {e}")
        print(f"\n-- FAIL --")
        return 1

    print(f"-- PASS --")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "validate":
        print("Usage: python -m simulation.scenario validate <path/to/scenario.yaml>")
        sys.exit(1)
    sys.exit(_cli_validate(sys.argv[2]))
