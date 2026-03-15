"""Tests for causal_model.py and history_matching.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


class TestCausalDAG:
    def test_dag_structure(self):
        from simulation.causal_model import DAG_EDGES, DAG_NODES
        assert len(DAG_EDGES) == 12
        assert len(DAG_NODES) == 10
        # All edge endpoints must be valid nodes
        for parent, child in DAG_EDGES:
            assert parent in DAG_NODES, f"{parent} not in DAG_NODES"
            assert child in DAG_NODES, f"{child} not in DAG_NODES"

    def test_griffiths_cascade_present(self):
        from simulation.causal_model import DAG_EDGES
        cascade = [
            ("usage_frequency", "salience"),
            ("salience", "conflict"),
            ("conflict", "mood_modification"),
            ("mood_modification", "tolerance"),
            ("tolerance", "withdrawal"),
            ("withdrawal", "relapse"),
        ]
        for edge in cascade:
            assert edge in DAG_EDGES, f"Missing Griffiths cascade edge: {edge}"

    def test_feedback_loop(self):
        from simulation.causal_model import DAG_EDGES
        assert ("relapse", "usage_frequency") in DAG_EDGES

    def test_priors_valid(self):
        from simulation.causal_model import PRIORS, DAG_NODES
        for node in DAG_NODES:
            assert node in PRIORS, f"No prior for {node}"
            a, b = PRIORS[node]
            assert a > 0 and b > 0, f"Invalid Beta params for {node}: ({a}, {b})"


class TestAgentSampling:
    def test_sample_agent_params_count(self):
        from simulation.causal_model import sample_agent_params
        agents = sample_agent_params(n_agents=6, seed=42)
        assert len(agents) == 6

    def test_sample_agent_params_keys(self):
        from simulation.causal_model import sample_agent_params, DAG_NODES
        agents = sample_agent_params(n_agents=1, seed=42)
        agent = agents[0]
        for node in DAG_NODES:
            assert node in agent, f"Missing node {node}"
        assert "susceptibility" in agent
        assert "resilience" in agent

    def test_sample_agent_params_range(self):
        from simulation.causal_model import sample_agent_params
        agents = sample_agent_params(n_agents=20, seed=42)
        for ag in agents:
            for key, val in ag.items():
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_sample_agent_params_heterogeneous(self):
        from simulation.causal_model import sample_agent_params
        agents = sample_agent_params(n_agents=20, seed=42)
        sus = [a["susceptibility"] for a in agents]
        assert max(sus) - min(sus) > 0.1, "Susceptibility not heterogeneous enough"

    def test_sample_reproducible(self):
        from simulation.causal_model import sample_agent_params
        a1 = sample_agent_params(n_agents=5, seed=42)
        a2 = sample_agent_params(n_agents=5, seed=42)
        for x, y in zip(a1, a2):
            assert x["susceptibility"] == y["susceptibility"]

    def test_node_to_param_mapping(self):
        from simulation.causal_model import sample_agent_params
        agents = sample_agent_params(n_agents=1, seed=42)
        ag = agents[0]
        assert ag["susceptibility"] == ag["usage_frequency"]
        assert ag["resilience"] == ag["self_regulation"]


class TestHistoryMatching:
    def test_pattern_target(self):
        from simulation.history_matching import PatternTarget
        t = PatternTarget("test", 0.1, 0.5, extractor=lambda x: 0.3)
        assert t.name == "test"
        assert t.low == 0.1
        assert t.high == 0.5

    def test_default_targets(self):
        from simulation.history_matching import default_targets
        targets = default_targets()
        assert len(targets) == 4
        names = {t.name for t in targets}
        assert "prevalence" in names
        assert "mean_score" in names

    def test_extractors(self):
        from simulation.history_matching import (
            fraction_above_threshold, mean_final_score,
            score_std, trajectory_monotonicity,
        )
        scores = [0.3, 0.5, 0.7, 0.9]
        assert fraction_above_threshold(scores, 0.7) == 0.5
        assert abs(mean_final_score(scores) - 0.6) < 1e-6
        assert score_std(scores) > 0
        logs = [[0.1, 0.2, 0.3], [0.5, 0.4, 0.3]]
        mono = trajectory_monotonicity(logs)
        assert 0.0 <= mono <= 1.0

    def test_history_matcher_runs(self):
        from simulation.history_matching import (
            HistoryMatcher, PatternTarget, mean_final_score,
        )
        targets = [
            PatternTarget("mean", 0.3, 0.6, tolerance=0.1,
                          extractor=lambda s: mean_final_score(s)),
        ]
        bounds = [(0.1, 0.4), (0.2, 0.8)]

        def sim_func(x):
            return {"mean": float(x[0] + x[1]) / 2}

        hm = HistoryMatcher(targets, bounds, param_names=["a", "b"],
                            n_waves=2, samples_per_wave=16, seed=42)
        nroy = hm.run(sim_func, verbose=False)
        assert len(nroy) > 0
        assert hm.summary()["n_waves"] == 2

    def test_run_history_matching_integration(self):
        from simulation.history_matching import run_history_matching
        result = run_history_matching(
            n_waves=2, samples_per_wave=16, n_ticks=6, verbose=False,
        )
        assert "final_nroy_size" in result
        assert result["n_waves"] == 2
        assert len(result["nroy_points"]) >= 0
