"""
Tests for ScenarioConfig loader, validator, and transitions registry.

Run: python -m pytest tests/test_scenario.py -v
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.scenario import (
    ScenarioConfig, MetaConfig, RoleConfig, AgentsConfig,
    ArchetypeProfile, ArchetypesConfig, ScoringConfig,
    TransitionRule, MemoryConfig, SafetyConfig, FlameConfig,
    EnvironmentConfig, ActionsConfig, ActionInstance, ContextCategory,
    DEFAULT_SCENARIO, BUILTIN_VARIABLES,
)
from simulation.transitions import (
    get_transition, list_transitions, register_transition,
    escalation_sustained, recovery_sustained,
    threshold_cross, oscillation_detect,
)


# ── Minimal valid config for testing ──────────────────────────────────────────

def _minimal_config() -> dict:
    return {
        "meta": {"name": "test", "description": "test scenario"},
        "agents": {
            "roles": [
                {
                    "id": "participant_template",
                    "slot": "swarm",
                    "type": "participant",
                    "system_prompt": "You are a test agent.",
                    "count": 2,
                },
            ],
        },
    }


def _full_config() -> dict:
    return {
        "meta": {"name": "full-test", "version": "1.0", "description": "Full test"},
        "agents": {
            "roles": [
                {"id": "observer_a", "slot": "authority", "type": "observer",
                 "system_prompt": "Analyse {domain_context}.", "max_tokens": 512},
                {"id": "observer_b", "slot": "authority", "type": "observer",
                 "system_prompt": "Intervene if needed.", "max_tokens": 256},
                {"id": "environment", "slot": "authority", "type": "environment",
                 "system_prompt": "Generate stimuli for {domain_context}.", "max_tokens": 256},
                {"id": "participant_template", "slot": "swarm", "type": "participant",
                 "system_prompt": "You are {agent_name}, a {archetype_label}.",
                 "max_tokens": 128, "count": 4},
            ],
        },
        "archetypes": {
            "profiles": [
                {"id": "K1", "label": "High", "description": "High risk",
                 "properties": {"trait": "high"}},
                {"id": "K2", "label": "Low", "description": "Low risk",
                 "properties": {"trait": "low"}},
            ],
            "distribution": {"K1": 0.4, "K2": 0.6},
        },
        "scoring": {
            "mode": "ema",
            "distributions": {
                "susceptibility": {"type": "beta", "params": [2, 3]},
            },
            "parameters": {"alpha": 0.15, "K": 4, "threshold": 0.7},
        },
        "transitions": [
            {"from": "K2", "to": "K1", "function": "escalation_sustained",
             "params": {"threshold": 0.8, "consecutive_ticks": 3}},
        ],
        "memory": {"enabled": True, "max_entries_per_agent": 10},
        "safety": {"enabled": True},
        "environment": {"tick_count": 12, "tick_unit": "step"},
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIOCONFIG TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestFrozenConfig:
    def test_config_is_immutable(self):
        config = ScenarioConfig.from_dict(_minimal_config())
        with pytest.raises(Exception):
            config.meta = MetaConfig(name="changed")

    def test_replicate_with_seed(self):
        config = ScenarioConfig.from_dict(_full_config())
        copy = config.replicate(seed=123)
        assert copy.environment.initial_state.get("seed") == 123
        # Original unchanged
        assert config.environment.initial_state.get("seed") is None

    def test_replicate_with_scoring_override(self):
        config = ScenarioConfig.from_dict(_full_config())
        copy = config.replicate(scoring_alpha=0.3)
        assert copy.scoring.parameters["alpha"] == 0.3
        # Original unchanged
        assert config.scoring.parameters["alpha"] == 0.15

    def test_replicate_returns_new_instance(self):
        config = ScenarioConfig.from_dict(_full_config())
        copy = config.replicate(seed=42)
        assert config is not copy

    def test_replicate_preserves_other_fields(self):
        config = ScenarioConfig.from_dict(_full_config())
        copy = config.replicate(seed=99)
        assert copy.meta.name == config.meta.name
        assert len(copy.archetypes.profiles) == len(config.archetypes.profiles)
        assert copy.scoring.mode == config.scoring.mode


class TestFromDict:
    def test_minimal_config(self):
        config = ScenarioConfig.from_dict(_minimal_config())
        assert config.meta.name == "test"
        assert len(config.agents.roles) == 1
        assert config.participant_count() == 2

    def test_full_config(self):
        config = ScenarioConfig.from_dict(_full_config())
        assert config.meta.name == "full-test"
        assert len(config.agents.roles) == 4
        assert len(config.archetypes.profiles) == 2
        assert config.participant_count() == 4

    def test_defaults(self):
        config = ScenarioConfig.from_dict(_minimal_config())
        assert config.scoring.mode == "ema"
        assert config.memory.enabled is True
        assert config.flame.enabled is False
        assert config.environment.tick_count == 100

    def test_scoring_param(self):
        config = ScenarioConfig.from_dict(_full_config())
        assert config.scoring_param("alpha") == 0.15
        assert config.scoring_param("missing", 99.0) == 99.0

    def test_domain_context(self):
        config = ScenarioConfig.from_dict(_full_config())
        assert config.domain_context == "Full test"

    def test_get_role(self):
        config = ScenarioConfig.from_dict(_full_config())
        role = config.get_role("observer_a")
        assert role is not None
        assert role.slot == "authority"
        assert config.get_role("nonexistent") is None

    def test_get_profile(self):
        config = ScenarioConfig.from_dict(_full_config())
        profile = config.get_profile("K1")
        assert profile is not None
        assert profile.label == "High"
        assert config.get_profile("nonexistent") is None


class TestLoadFile:
    def test_load_social_dynamics(self):
        config = ScenarioConfig.load("scenarios/social_dynamics.yaml")
        assert config.meta.name == "social-dynamics"
        assert config.participant_count() == 4
        assert len(config.archetypes.profiles) == 3

    def test_load_template(self):
        config = ScenarioConfig.load("scenarios/_template.yaml")
        assert config.meta.name == "my-scenario"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            ScenarioConfig.load("scenarios/nonexistent.yaml")


class TestValidation:
    def test_valid_config_passes(self):
        config = ScenarioConfig.from_dict(_full_config())
        report = config.validate_scenario(strict=False)
        assert report["valid"] is True
        assert len(report["errors"]) == 0

    def test_distribution_sum_error(self):
        data = _full_config()
        data["archetypes"]["distribution"] = {"K1": 0.3, "K2": 0.3}  # sums to 0.6
        config = ScenarioConfig.from_dict(data)
        report = config.validate_scenario(strict=False)
        assert not report["valid"]
        assert any("sum" in e for e in report["errors"])

    def test_distribution_undefined_profile(self):
        data = _full_config()
        data["archetypes"]["distribution"]["K99"] = 0.1
        config = ScenarioConfig.from_dict(data)
        report = config.validate_scenario(strict=False)
        assert any("K99" in e for e in report["errors"])

    def test_transition_invalid_profile(self):
        data = _full_config()
        data["transitions"] = [
            {"from": "NONEXISTENT", "to": "K1", "function": "escalation_sustained"},
        ]
        config = ScenarioConfig.from_dict(data)
        report = config.validate_scenario(strict=False)
        assert any("NONEXISTENT" in e for e in report["errors"])

    def test_transition_invalid_function(self):
        data = _full_config()
        data["transitions"] = [
            {"from": "K1", "to": "K2", "function": "nonexistent_function"},
        ]
        config = ScenarioConfig.from_dict(data)
        report = config.validate_scenario(strict=False)
        assert any("nonexistent_function" in e for e in report["errors"])

    def test_no_participant_role_error(self):
        data = _minimal_config()
        data["agents"]["roles"][0]["type"] = "observer"
        config = ScenarioConfig.from_dict(data)
        report = config.validate_scenario(strict=False)
        assert any("participant" in e.lower() for e in report["errors"])

    def test_unknown_prompt_variables_warning(self):
        data = _minimal_config()
        data["agents"]["roles"][0]["system_prompt"] = "Hello {unknown_var}!"
        config = ScenarioConfig.from_dict(data)
        report = config.validate_scenario(strict=False)
        assert any("unknown_var" in w for w in report["warnings"])

    def test_strict_raises_on_error(self):
        data = _full_config()
        data["archetypes"]["distribution"] = {"K1": 0.3, "K2": 0.3}
        config = ScenarioConfig.from_dict(data)
        with pytest.raises(ValueError, match="validation failed"):
            config.validate_scenario(strict=True)

    def test_social_dynamics_validates(self):
        config = ScenarioConfig.load("scenarios/social_dynamics.yaml")
        report = config.validate_scenario(strict=False)
        assert report["valid"] is True, f"Errors: {report['errors']}"


class TestRoleValidation:
    def test_invalid_slot_raises(self):
        data = _minimal_config()
        data["agents"]["roles"][0]["slot"] = "invalid"
        with pytest.raises(Exception):
            ScenarioConfig.from_dict(data)

    def test_invalid_type_raises(self):
        data = _minimal_config()
        data["agents"]["roles"][0]["type"] = "invalid"
        with pytest.raises(Exception):
            ScenarioConfig.from_dict(data)

    def test_invalid_scoring_mode_raises(self):
        data = _minimal_config()
        data["scoring"] = {"mode": "invalid"}
        with pytest.raises(Exception):
            ScenarioConfig.from_dict(data)


# ══════════════════════════════════════════════════════════════════════════════
# TRANSITIONS REGISTRY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestTransitionRegistry:
    def test_builtin_transitions_registered(self):
        names = list_transitions()
        assert "escalation_sustained" in names
        assert "recovery_sustained" in names
        assert "threshold_cross" in names
        assert "oscillation_detect" in names

    def test_get_existing(self):
        fn = get_transition("escalation_sustained")
        assert callable(fn)

    def test_get_nonexistent_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            get_transition("nonexistent_function")

    def test_custom_registration(self):
        @register_transition("test_custom")
        def custom(agent_state, tick_history, **params):
            return True
        assert get_transition("test_custom") is custom


class TestEscalationSustained:
    def test_below_threshold(self):
        history = [{"score": 0.5}, {"score": 0.6}, {"score": 0.7}]
        assert not escalation_sustained({}, history, threshold=0.8, consecutive_ticks=3)

    def test_above_threshold(self):
        history = [{"score": 0.9}, {"score": 0.85}, {"score": 0.95}]
        assert escalation_sustained({}, history, threshold=0.8, consecutive_ticks=3)

    def test_too_short_history(self):
        history = [{"score": 0.9}]
        assert not escalation_sustained({}, history, threshold=0.8, consecutive_ticks=3)

    def test_mixed_above_below(self):
        history = [{"score": 0.9}, {"score": 0.5}, {"score": 0.9}]
        assert not escalation_sustained({}, history, threshold=0.8, consecutive_ticks=3)


class TestRecoverySustained:
    def test_above_threshold(self):
        history = [{"score": 0.5}] * 5
        assert not recovery_sustained({}, history, threshold=0.3, consecutive_ticks=5)

    def test_below_threshold(self):
        history = [{"score": 0.2}] * 5
        assert recovery_sustained({}, history, threshold=0.3, consecutive_ticks=5)

    def test_too_short(self):
        history = [{"score": 0.1}] * 3
        assert not recovery_sustained({}, history, threshold=0.3, consecutive_ticks=5)


class TestThresholdCross:
    def test_cross_up(self):
        history = [{"score": 0.4}, {"score": 0.6}]
        assert threshold_cross({}, history, threshold=0.5, direction="up")

    def test_no_cross_up(self):
        history = [{"score": 0.6}, {"score": 0.7}]
        assert not threshold_cross({}, history, threshold=0.5, direction="up")

    def test_cross_down(self):
        history = [{"score": 0.6}, {"score": 0.4}]
        assert threshold_cross({}, history, threshold=0.5, direction="down")

    def test_too_short(self):
        history = [{"score": 0.6}]
        assert not threshold_cross({}, history, threshold=0.5, direction="up")


class TestOscillationDetect:
    def test_high_oscillation(self):
        history = [{"score": 0.2}, {"score": 0.8}, {"score": 0.3},
                   {"score": 0.7}, {"score": 0.2}]
        assert oscillation_detect({}, history, window=5, amplitude=0.5)

    def test_low_oscillation(self):
        history = [{"score": 0.5}, {"score": 0.51}, {"score": 0.49},
                   {"score": 0.5}, {"score": 0.52}]
        assert not oscillation_detect({}, history, window=5, amplitude=0.5)

    def test_too_short(self):
        history = [{"score": 0.2}, {"score": 0.8}]
        assert not oscillation_detect({}, history, window=5, amplitude=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — SIM_LOOP INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestSimLoopScenarioIntegration:
    def test_run_simulation_accepts_scenario_config(self):
        """run_simulation() signature includes scenario_config param."""
        import inspect
        from simulation.sim_loop import run_simulation
        sig = inspect.signature(run_simulation)
        assert "scenario_config" in sig.parameters

    def test_scenario_config_extracts_params(self):
        """ScenarioConfig params override defaults when provided."""
        config = ScenarioConfig.load("scenarios/social_dynamics.yaml")
        assert config.environment.tick_count == 12
        assert config.participant_count() == 4
        assert config.scoring_param("alpha") == 0.15
        assert config.scoring_param("K") == 4
        assert config.scoring.mode == "ema"

    def test_network_scenario_loads_and_validates(self):
        config = ScenarioConfig.load("scenarios/network_resilience.yaml")
        report = config.validate_scenario(strict=False)
        assert report["valid"], f"Errors: {report['errors']}"
        assert config.participant_count() == 6

    def test_market_scenario_loads_and_validates(self):
        config = ScenarioConfig.load("scenarios/market_ecosystem.yaml")
        report = config.validate_scenario(strict=False)
        assert report["valid"], f"Errors: {report['errors']}"
        assert config.participant_count() == 6

    def test_detect_missing_context_with_scenario(self):
        from simulation.sim_loop import detect_missing_context
        config = ScenarioConfig.load("scenarios/network_resilience.yaml")
        result = detect_missing_context(scenario_config=config)
        # Should use network-specific categories
        missing_cats = [m["category"] for m in result["missing"]]
        assert "network_topology" in missing_cats or any(
            "network" in m["category"] for m in result.get("present", [])
        )

    def test_detect_missing_context_without_scenario(self):
        from simulation.sim_loop import detect_missing_context
        result = detect_missing_context()
        # Should use default categories
        missing_cats = [m["category"] for m in result["missing"]]
        # Default categories include scenario_description
        assert any("scenario" in c for c in missing_cats)
