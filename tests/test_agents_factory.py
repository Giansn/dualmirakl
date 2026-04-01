"""
Tests for the generic agent factory and pluggable scoring engines.

Run: python -m pytest tests/test_agents_factory.py -v
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.scenario import ScenarioConfig
from simulation.agents import AgentFactory, AgentSpec, AgentSet, _safe_format, _assign_profiles
from simulation.scoring import ScoreEngine, EMAScoreEngine, LogisticScoreEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _test_config() -> ScenarioConfig:
    return ScenarioConfig.from_dict({
        "meta": {"name": "test", "description": "Test scenario"},
        "agents": {
            "roles": [
                {"id": "observer_a", "slot": "authority", "type": "observer",
                 "system_prompt": "Analyse {domain_context}.", "max_tokens": 512},
                {"id": "observer_b", "slot": "authority", "type": "observer",
                 "system_prompt": "Intervene.", "max_tokens": 256},
                {"id": "environment", "slot": "authority", "type": "environment",
                 "system_prompt": "Generate stimuli.", "max_tokens": 256},
                {"id": "participant_template", "slot": "swarm", "type": "participant",
                 "system_prompt": "You are {agent_name}, a {archetype_label}. {archetype_profile}",
                 "max_tokens": 128, "count": 4},
            ],
        },
        "archetypes": {
            "profiles": [
                {"id": "K1", "label": "Bold", "description": "Risk-taker",
                 "properties": {"style": "aggressive"}},
                {"id": "K2", "label": "Cautious", "description": "Risk-averse",
                 "properties": {"style": "conservative"}},
            ],
            "distribution": {"K1": 0.5, "K2": 0.5},
        },
        "scoring": {
            "mode": "ema",
            "parameters": {"alpha": 0.2, "threshold": 0.7},
        },
    })


# ══════════════════════════════════════════════════════════════════════════════
# AGENT FACTORY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentFactory:
    def test_returns_agentset(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        assert isinstance(agents, AgentSet)

    def test_creates_all_roles(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        assert len(agents.by_type("observer")) == 2
        assert len(agents.by_type("environment")) == 1
        assert len(agents.by_type("participant")) == 4

    def test_participant_ids(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        ids = agents.by_type("participant").ids()
        assert ids == ["participant_0", "participant_1", "participant_2", "participant_3"]

    def test_observer_slot(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        for obs in agents.by_type("observer"):
            assert obs.slot == "authority"
            assert obs.backend == "authority"

    def test_participant_slot(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        for p in agents.by_type("participant"):
            assert p.slot == "swarm"

    def test_profiles_assigned(self):
        config = _test_config()
        agents = AgentFactory.from_config(config, rng=np.random.RandomState(42))
        profiles = [a.profile for a in agents.by_type("participant")]
        assert all(p is not None for p in profiles)
        labels = {p.label for p in profiles}
        assert labels == {"Bold", "Cautious"}

    def test_profile_distribution(self):
        """With 50/50 distribution and 4 agents, should get 2 of each."""
        config = _test_config()
        agents = AgentFactory.from_config(config, rng=np.random.RandomState(42))
        labels = [a.profile.label for a in agents.by_type("participant")]
        assert labels.count("Bold") == 2
        assert labels.count("Cautious") == 2

    def test_deterministic_with_seed(self):
        config = _test_config()
        a1 = AgentFactory.from_config(config, rng=np.random.RandomState(42))
        a2 = AgentFactory.from_config(config, rng=np.random.RandomState(42))
        labels1 = [a.profile.id for a in a1.by_type("participant")]
        labels2 = [a.profile.id for a in a2.by_type("participant")]
        assert labels1 == labels2

    def test_domain_context_propagated(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        assert agents.by_type("observer")[0].domain_context == "Test scenario"
        assert agents.by_type("participant")[0].domain_context == "Test scenario"

    def test_no_profiles_gives_none(self):
        config = ScenarioConfig.from_dict({
            "meta": {"name": "test", "description": "No profiles"},
            "agents": {"roles": [
                {"id": "p", "slot": "swarm", "type": "participant",
                 "system_prompt": "Hello", "count": 2},
            ]},
        })
        agents = AgentFactory.from_config(config)
        assert all(a.profile is None for a in agents.by_type("participant"))


class TestAgentSpec:
    def test_render_prompt_builtins(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        obs = agents.by_type("observer")[0]
        rendered = obs.render_prompt()
        assert "Test scenario" in rendered  # {domain_context}

    def test_render_prompt_profile(self):
        config = _test_config()
        agents = AgentFactory.from_config(config, rng=np.random.RandomState(42))
        p = agents.by_type("participant")[0]
        rendered = p.render_prompt()
        assert p.agent_id in rendered  # {agent_name}
        assert p.profile.label in rendered  # {archetype_label}

    def test_render_prompt_tick_state(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        obs = agents.by_type("observer")[0]
        rendered = obs.render_prompt({"domain_context": "OVERRIDE"})
        assert "OVERRIDE" in rendered

    def test_render_prompt_unknown_vars_preserved(self):
        spec = AgentSpec("test", _test_config().agents.roles[0])
        spec.system_prompt_template = "Hello {known} and {unknown}!"
        rendered = spec.render_prompt({"known": "world"})
        assert "world" in rendered
        assert "{unknown}" in rendered

    def test_repr(self):
        config = _test_config()
        agents = AgentFactory.from_config(config, rng=np.random.RandomState(42))
        r = repr(agents.by_type("participant")[0])
        assert "participant_0" in r
        assert "participant" in r
        assert "swarm" in r


class TestAgentSet:
    def test_by_slot(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        authority = agents.by_slot("authority")
        assert len(authority) == 3  # 2 observers + 1 environment
        swarm = agents.by_slot("swarm")
        assert len(swarm) == 4  # 4 participants

    def test_by_type(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        assert len(agents.by_type("observer")) == 2
        assert len(agents.by_type("participant")) == 4
        assert len(agents.by_type("environment")) == 1

    def test_by_id(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        obs = agents.by_id("observer_a")
        assert obs is not None
        assert obs.agent_id == "observer_a"
        assert agents.by_id("nonexistent") is None

    def test_select(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        big = agents.select(lambda a: a.max_tokens >= 256)
        assert len(big) == 3  # 2 observers (512, 256) + 1 env (256)

    def test_agg(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        types = agents.agg("type")
        assert "observer" in types
        assert "participant" in types

    def test_ids(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        ids = agents.by_type("participant").ids()
        assert len(ids) == 4
        assert "participant_0" in ids

    def test_repr(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        r = repr(agents)
        assert "AgentSet" in r
        assert "total=7" in r

    def test_iteration(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        count = sum(1 for _ in agents)
        assert count == 7

    def test_indexing(self):
        config = _test_config()
        agents = AgentFactory.from_config(config)
        first = agents[0]
        assert isinstance(first, AgentSpec)


class TestSafeFormat:
    def test_basic(self):
        assert _safe_format("Hello {name}!", {"name": "world"}) == "Hello world!"

    def test_unknown_preserved(self):
        assert _safe_format("{a} and {b}", {"a": "X"}) == "X and {b}"

    def test_empty_dict(self):
        assert _safe_format("{x}", {}) == "{x}"

    def test_no_vars(self):
        assert _safe_format("plain text", {"x": "y"}) == "plain text"


class TestAssignProfiles:
    def test_even_distribution(self):
        from simulation.scenario import ArchetypeProfile
        profiles = [
            ArchetypeProfile(id="A", label="A", properties={}),
            ArchetypeProfile(id="B", label="B", properties={}),
        ]
        result = _assign_profiles(4, profiles, {"A": 0.5, "B": 0.5}, np.random.RandomState(42))
        ids = [p.id for p in result]
        assert ids.count("A") == 2
        assert ids.count("B") == 2

    def test_uneven_distribution(self):
        from simulation.scenario import ArchetypeProfile
        profiles = [
            ArchetypeProfile(id="A", label="A", properties={}),
            ArchetypeProfile(id="B", label="B", properties={}),
        ]
        result = _assign_profiles(10, profiles, {"A": 0.7, "B": 0.3}, np.random.RandomState(42))
        ids = [p.id for p in result]
        assert ids.count("A") == 7
        assert ids.count("B") == 3

    def test_empty_profiles(self):
        result = _assign_profiles(3, [], {}, np.random.RandomState(42))
        assert all(p is None for p in result)


# ══════════════════════════════════════════════════════════════════════════════
# SCORE ENGINE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestEMAScoreEngine:
    def test_basic_update(self):
        engine = EMAScoreEngine(alpha=0.2)
        result = engine.update(0.3, 0.8)
        expected = 0.3 + 0.2 * (0.8 - 0.3)  # 0.4
        assert abs(result - expected) < 1e-6

    def test_dampening(self):
        engine = EMAScoreEngine(alpha=0.2)
        undamped = engine.update(0.3, 0.8, dampening=1.0)
        damped = engine.update(0.3, 0.8, dampening=0.5)
        assert damped < undamped

    def test_susceptibility(self):
        engine = EMAScoreEngine(alpha=0.2)
        low_sus = engine.update(0.3, 0.8, susceptibility=0.2)
        high_sus = engine.update(0.3, 0.8, susceptibility=0.9)
        assert high_sus > low_sus

    def test_resilience(self):
        engine = EMAScoreEngine(alpha=0.2)
        no_res = engine.update(0.3, 0.8, resilience=0.0)
        high_res = engine.update(0.3, 0.8, resilience=0.8)
        assert high_res < no_res

    def test_clamps_to_01(self):
        engine = EMAScoreEngine(alpha=1.0)
        assert engine.update(0.99, 1.5) <= 1.0
        assert engine.update(0.01, -0.5) >= 0.0

    def test_matches_sim_loop_update_score(self):
        """EMAScoreEngine must produce identical results to sim_loop.update_score."""
        from simulation.signal_computation import update_score
        engine = EMAScoreEngine(alpha=0.15)
        for current, signal, damp, sus, res in [
            (0.3, 0.8, 1.0, 1.0, 0.0),
            (0.5, 0.2, 0.6, 0.5, 0.3),
            (0.9, 0.1, 1.0, 0.8, 0.1),
        ]:
            expected = update_score(current, signal, damp, 0.15, "ema", 6.0, sus, res)
            got = engine.update(current, signal, damp, sus, res)
            assert abs(got - expected) < 1e-10, f"Mismatch: {got} != {expected}"


class TestLogisticScoreEngine:
    def test_saturation(self):
        engine = LogisticScoreEngine(alpha=0.2, logistic_k=6.0)
        # At high current, delta should be smaller than at low current
        delta_low = engine.update(0.3, 0.8) - 0.3
        delta_high = engine.update(0.85, 0.95) - 0.85
        assert delta_low > delta_high  # sigmoid saturates at extremes

    def test_matches_sim_loop_update_score(self):
        from simulation.signal_computation import update_score
        engine = LogisticScoreEngine(alpha=0.15, logistic_k=6.0)
        for current, signal, damp, sus, res in [
            (0.3, 0.8, 1.0, 1.0, 0.0),
            (0.5, 0.2, 0.6, 0.5, 0.3),
            (0.9, 0.1, 1.0, 0.8, 0.1),
        ]:
            expected = update_score(current, signal, damp, 0.15, "logistic", 6.0, sus, res)
            got = engine.update(current, signal, damp, sus, res)
            assert abs(got - expected) < 1e-10, f"Mismatch: {got} != {expected}"


class TestScoreEngineFactory:
    def test_from_config_ema(self):
        engine = ScoreEngine.from_config({"mode": "ema", "parameters": {"alpha": 0.1}})
        assert isinstance(engine, EMAScoreEngine)
        assert engine.alpha == 0.1

    def test_from_config_logistic(self):
        engine = ScoreEngine.from_config({
            "mode": "logistic",
            "parameters": {"alpha": 0.2, "logistic_k": 8.0},
        })
        assert isinstance(engine, LogisticScoreEngine)
        assert engine.logistic_k == 8.0

    def test_from_config_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scoring mode"):
            ScoreEngine.from_config({"mode": "quantum"})

    def test_from_scenario_config(self):
        config = _test_config()
        engine = ScoreEngine.from_config(config.scoring)
        assert isinstance(engine, EMAScoreEngine)
        assert engine.alpha == 0.2

    def test_from_social_dynamics_yaml(self):
        config = ScenarioConfig.load("scenarios/social_dynamics.yaml")
        engine = ScoreEngine.from_config(config.scoring)
        assert isinstance(engine, EMAScoreEngine)
        assert engine.alpha == 0.15
