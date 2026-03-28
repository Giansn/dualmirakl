"""
Tests for Phase C: scenario trees, GP emulators, ABC-SMC calibration.

Run: python -m pytest tests/test_phase_c.py -v
"""

import sys
import os
import json
import math

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Scenario tree tests ──────────────────────────────────────────────────────

class TestScenarioTree:

    @staticmethod
    def _make_logs(n_runs=10, n_agents=4, n_ticks=12, seed=42):
        """Generate synthetic score logs for testing."""
        rng = np.random.RandomState(seed)
        logs = []
        for r in range(n_runs):
            base = 0.3 + 0.4 * (r / n_runs)  # runs span 0.3-0.7
            agents = []
            for a in range(n_agents):
                scores = [base + rng.normal(0, 0.05) for _ in range(n_ticks)]
                agents.append(scores)
            logs.append(agents)
        return logs

    def test_build_trivial_tree(self):
        from simulation.scenario_tree import build_scenario_tree
        logs = self._make_logs(n_runs=1)
        tree = build_scenario_tree(logs)
        assert tree.id == "root"
        assert tree.probability == 1.0
        assert tree.n_supporting_runs == 1

    def test_build_two_cluster_tree(self):
        """Two distinct run groups should produce 2+ branches."""
        from simulation.scenario_tree import build_scenario_tree
        rng = np.random.RandomState(42)
        logs = []
        # Group 1: low scores
        for _ in range(5):
            logs.append([[0.2 + rng.normal(0, 0.02) for _ in range(12)] for _ in range(4)])
        # Group 2: high scores
        for _ in range(5):
            logs.append([[0.8 + rng.normal(0, 0.02) for _ in range(12)] for _ in range(4)])
        tree = build_scenario_tree(logs, max_depth=2, max_branches=3)
        assert tree.n_supporting_runs == 10
        assert len(tree.children) >= 2

    def test_branch_probabilities_sum(self):
        from simulation.scenario_tree import build_scenario_tree
        logs = self._make_logs(n_runs=20)
        tree = build_scenario_tree(logs, max_depth=2, max_branches=4)
        if tree.children:
            prob_sum = sum(c.probability for c in tree.children)
            assert abs(prob_sum - 1.0) < 0.01

    def test_max_depth_respected(self):
        from simulation.scenario_tree import build_scenario_tree
        logs = self._make_logs(n_runs=30)
        tree = build_scenario_tree(logs, max_depth=2)
        assert tree.depth() <= 2

    def test_min_branch_prob_pruning(self):
        from simulation.scenario_tree import build_scenario_tree
        logs = self._make_logs(n_runs=20)
        tree = build_scenario_tree(logs, max_depth=1, max_branches=4, min_branch_prob=0.2)
        if tree.children:
            for child in tree.children:
                assert child.probability >= 0.15  # some tolerance

    def test_tree_serialization(self):
        from simulation.scenario_tree import build_scenario_tree, tree_to_dict
        logs = self._make_logs(n_runs=10)
        tree = build_scenario_tree(logs, max_depth=2)
        d = tree_to_dict(tree)
        assert "id" in d
        assert "probability" in d
        assert "metrics" in d
        json.dumps(d)  # must be JSON-serializable

    def test_tree_to_flat_scenarios(self):
        from simulation.scenario_tree import build_scenario_tree, tree_to_flat_scenarios
        logs = self._make_logs(n_runs=20)
        tree = build_scenario_tree(logs, max_depth=2, max_branches=3)
        scenarios = tree_to_flat_scenarios(tree)
        assert len(scenarios) >= 1
        for s in scenarios:
            assert "leaf_id" in s
            assert "probability" in s
            assert "path" in s

    def test_empty_logs(self):
        from simulation.scenario_tree import build_scenario_tree
        tree = build_scenario_tree([])
        assert tree.id == "root"
        assert tree.probability == 1.0


class TestDupacovaReduction:

    def test_reduce_to_target(self):
        from simulation.scenario_tree import build_scenario_tree, reduce_tree
        rng = np.random.RandomState(42)
        logs = []
        for i in range(20):
            base = 0.1 + 0.8 * (i / 20)
            logs.append([[base + rng.normal(0, 0.03) for _ in range(12)] for _ in range(4)])
        tree = build_scenario_tree(logs, max_depth=1, max_branches=6, min_branch_prob=0.01)
        n_before = tree.n_leaves()
        if n_before > 3:
            reduced = reduce_tree(tree, target_scenarios=3)
            assert reduced.n_leaves() <= 3

    def test_probability_conservation(self):
        from simulation.scenario_tree import build_scenario_tree, reduce_tree
        rng = np.random.RandomState(42)
        logs = []
        for i in range(15):
            base = 0.2 + 0.6 * (i / 15)
            logs.append([[base + rng.normal(0, 0.02) for _ in range(8)] for _ in range(3)])
        tree = build_scenario_tree(logs, max_depth=1, max_branches=5, min_branch_prob=0.01)
        if tree.n_leaves() > 2:
            reduced = reduce_tree(tree, target_scenarios=2)
            if reduced.children:
                total_prob = sum(c.probability for c in reduced.children)
                assert abs(total_prob - 1.0) < 0.05


# ── GP emulator tests ────────────────────────────────────────────────────────

class TestGPEmulator:

    def test_fit_predict_1d(self):
        from simulation.gp_emulator import GPEmulator
        gp = GPEmulator()
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = np.sin(2 * np.pi * X).ravel()
        gp.fit(X, y)
        X_new = np.array([[0.25], [0.75]])
        mean, std = gp.predict(X_new)
        assert abs(mean[0] - np.sin(2 * np.pi * 0.25)) < 0.3
        assert abs(mean[1] - np.sin(2 * np.pi * 0.75)) < 0.3

    def test_fit_predict_2d(self):
        from simulation.gp_emulator import GPEmulator
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (30, 2))
        y = X[:, 0] + 0.5 * X[:, 1]  # simple linear
        gp = GPEmulator()
        gp.fit(X, y)
        X_new = np.array([[0.5, 0.5]])
        mean, _ = gp.predict(X_new)
        assert abs(mean[0] - 0.75) < 0.2

    def test_uncertainty_grows_far(self):
        from simulation.gp_emulator import GPEmulator, _HAS_SKLEARN
        if not _HAS_SKLEARN:
            pytest.skip("sklearn not available")
        gp = GPEmulator()
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = np.sin(X).ravel()
        gp.fit(X, y)
        _, std_near = gp.predict(np.array([[0.5]]))
        _, std_far = gp.predict(np.array([[5.0]]))
        assert std_far[0] > std_near[0]

    def test_validate_r2(self):
        from simulation.gp_emulator import GPEmulator
        gp = GPEmulator()
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (30, 2))
        y = X[:, 0] ** 2 + X[:, 1]  # smooth function
        gp.fit(X, y)
        cv = gp.validate(n_folds=5)
        assert "r2" in cv
        assert cv["r2"] > 0.5  # should be well above for a smooth function

    def test_suggest_uncertainty(self):
        from simulation.gp_emulator import GPEmulator
        gp = GPEmulator()
        X = np.linspace(0, 0.5, 10).reshape(-1, 1)  # only trained in [0, 0.5]
        y = X.ravel() ** 2
        gp.fit(X, y)
        suggestions = gp.suggest_next([(0, 1)], n_suggestions=3, strategy="uncertainty")
        assert suggestions.shape == (3, 1)
        # Suggestions should be in [0, 1]
        assert np.all(suggestions >= 0) and np.all(suggestions <= 1)

    def test_suggest_expected_improvement(self):
        from simulation.gp_emulator import GPEmulator, _HAS_SKLEARN
        if not _HAS_SKLEARN:
            pytest.skip("sklearn not available")
        gp = GPEmulator()
        X = np.linspace(0, 1, 15).reshape(-1, 1)
        y = (X.ravel() - 0.3) ** 2  # minimum at 0.3
        gp.fit(X, y)
        suggestions = gp.suggest_next([(0, 1)], n_suggestions=3, strategy="expected_improvement")
        assert suggestions.shape == (3, 1)

    def test_emulator_result_fields(self):
        from simulation.gp_emulator import GPEmulator
        gp = GPEmulator()
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = X.ravel()
        result = gp.fit(X, y)
        assert result.n_train == 10
        assert result.r2_score > 0.9
        assert result.backend in ("sklearn_gp", "scipy_rbf")


# ── History matcher + emulator tests ─────────────────────────────────────────

class TestHistoryMatcherEmulator:

    @staticmethod
    def _simple_sim(x):
        return {"mean_score": float(0.3 * x[0] + 0.7 * x[1])}

    def test_run_with_emulator(self):
        from simulation.history_matching import HistoryMatcher, PatternTarget
        targets = [PatternTarget(
            name="mean_score", low=0.4, high=0.6,
            extractor=lambda o: o.get("mean_score", 0),
        )]
        hm = HistoryMatcher(
            targets=targets,
            bounds=[(0, 1), (0, 1)],
            param_names=["a", "b"],
            n_waves=3,
            samples_per_wave=32,
            seed=42,
        )
        nroy = hm.run_with_emulator(self._simple_sim, emulator_after_wave=1, verbose=False)
        assert len(nroy) > 0

    def test_emulator_reduces_calls(self):
        """Emulator mode should make fewer direct sim calls than full direct."""
        from simulation.history_matching import HistoryMatcher, PatternTarget

        call_count = {"direct": 0, "emulator": 0}

        def counting_sim(x):
            call_count["current"] += 1
            return {"mean_score": float(0.5 * x[0] + 0.5 * x[1])}

        targets = [PatternTarget(
            name="mean_score", low=0.3, high=0.7,
            extractor=lambda o: o.get("mean_score", 0),
        )]

        # Direct: all waves use sim
        call_count["current"] = 0
        hm_d = HistoryMatcher(targets=targets, bounds=[(0, 1), (0, 1)],
                              n_waves=3, samples_per_wave=32, seed=42)
        hm_d.run(counting_sim, verbose=False)
        call_count["direct"] = call_count["current"]

        # Emulator: wave 1 direct, waves 2-3 emulator + 20% verification
        call_count["current"] = 0
        hm_e = HistoryMatcher(targets=targets, bounds=[(0, 1), (0, 1)],
                              n_waves=3, samples_per_wave=32, seed=42)
        hm_e.run_with_emulator(counting_sim, emulator_after_wave=1, verbose=False)
        call_count["emulator"] = call_count["current"]

        assert call_count["emulator"] < call_count["direct"]


# ── ABC-SMC tests ────────────────────────────────────────────────────────────

class TestABCSMC:

    @staticmethod
    def _simple_sim(theta):
        """Ground truth: output = 2*theta[0] + theta[1]."""
        return {"metric": float(2 * theta[0] + theta[1])}

    def test_abc_recovers_parameter(self):
        from simulation.abc_calibration import abc_smc, ABCPrior
        # Ground truth: theta=[0.3, 0.4] → metric = 2*0.3 + 0.4 = 1.0
        priors = [
            ABCPrior("a", "uniform", low=0, high=1),
            ABCPrior("b", "uniform", low=0, high=1),
        ]
        observed = {"metric": 1.0}
        result = abc_smc(
            self._simple_sim, priors, observed,
            n_particles=50, n_populations=3,
            initial_epsilon=1.0, seed=42, verbose=False,
        )
        assert len(result.posterior_samples) > 0
        # Posterior mean of 2*a + b should be near 1.0
        post_mean = result.posterior_mean()
        predicted = 2 * post_mean[0] + post_mean[1]
        assert abs(predicted - 1.0) < 0.3

    def test_abc_decreasing_epsilon(self):
        from simulation.abc_calibration import abc_smc, ABCPrior
        priors = [ABCPrior("x", "uniform", low=0, high=1)]
        observed = {"val": 0.5}

        def sim(theta):
            return {"val": float(theta[0])}

        result = abc_smc(sim, priors, observed, n_particles=30, n_populations=4,
                        initial_epsilon=1.0, seed=42, verbose=False)
        if len(result.populations) >= 2:
            eps = [p.epsilon for p in result.populations]
            # Each epsilon should be <= previous
            for i in range(1, len(eps)):
                assert eps[i] <= eps[i - 1] + 0.01  # small tolerance

    def test_abc_weights_normalized(self):
        from simulation.abc_calibration import abc_smc, ABCPrior
        priors = [ABCPrior("x", "uniform", low=0, high=1)]
        observed = {"val": 0.5}

        def sim(theta):
            return {"val": float(theta[0])}

        result = abc_smc(sim, priors, observed, n_particles=30, n_populations=2,
                        initial_epsilon=1.0, seed=42, verbose=False)
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_abc_prior_types(self):
        from simulation.abc_calibration import ABCPrior
        rng = np.random.default_rng(42)
        # Uniform
        p = ABCPrior("u", "uniform", low=0, high=1)
        samples = p.sample(rng, 100)
        assert np.all(samples >= 0) and np.all(samples <= 1)
        # Normal
        p = ABCPrior("n", "normal", mean=5, std=1)
        samples = p.sample(rng, 100)
        assert abs(samples.mean() - 5) < 0.5
        # Beta
        p = ABCPrior("b", "beta", alpha=2, beta=5)
        samples = p.sample(rng, 100)
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_abc_result_to_dict(self):
        from simulation.abc_calibration import abc_smc, ABCPrior
        priors = [ABCPrior("x", "uniform", low=0, high=1)]
        result = abc_smc(
            lambda t: {"v": float(t[0])},
            priors, {"v": 0.5},
            n_particles=20, n_populations=2,
            initial_epsilon=1.0, seed=42, verbose=False,
        )
        d = result.to_dict()
        assert "posterior_mean" in d
        assert "n_populations" in d
        json.dumps(d)

    def test_make_abc_sim_func(self):
        from simulation.abc_calibration import make_abc_sim_func
        sim = make_abc_sim_func(
            param_names=["alpha", "susceptibility"],
            bounds=[(0.05, 0.4), (0.1, 0.9)],
        )
        result = sim(np.array([0.15, 0.5]))
        assert "mean_score" in result
        assert "score_std" in result
        assert 0 <= result["mean_score"] <= 1

    def test_default_distance(self):
        from simulation.abc_calibration import default_distance
        d = default_distance({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0})
        assert d == 0.0
        d = default_distance({"a": 0.0}, {"a": 1.0})
        assert d == 1.0


class TestCalibrationPipeline:

    def test_hm_then_abc(self):
        from simulation.abc_calibration import calibration_pipeline, ABCPrior

        def sim(theta):
            return {"metric": float(0.5 * theta[0] + 0.5 * theta[1])}

        priors = [
            ABCPrior("a", "uniform", low=0, high=1),
            ABCPrior("b", "uniform", low=0, high=1),
        ]
        result = calibration_pipeline(
            sim_func=sim,
            priors=priors,
            observed={"metric": 0.5},
            param_names=["a", "b"],
            bounds=[(0, 1), (0, 1)],
            hm_waves=2, hm_samples=32,
            abc_particles=30, abc_populations=2,
            seed=42,
        )
        assert "nroy_fraction" in result
        assert "abc_result" in result
        assert "posterior_mean" in result


# ── Ensemble all_score_logs field tests ──────────────────────────────────────

class TestEnsembleScoreLogs:

    def test_ensemble_result_has_field(self):
        from simulation.ensemble import EnsembleResult
        r = EnsembleResult(experiment_id="test")
        assert hasattr(r, "all_score_logs")
        assert r.all_score_logs == []

    def test_nested_result_has_field(self):
        from simulation.ensemble import NestedEnsembleResult
        r = NestedEnsembleResult(experiment_id="test")
        assert hasattr(r, "all_score_logs")
        assert r.all_score_logs == []
