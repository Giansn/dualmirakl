"""
Tests for Optuna optimization module.
Tests parameter space definitions and surrogate objective without optuna.
"""

import pytest
import numpy as np

from simulation.optimize import (
    DUALMIRAKL_PARAMS,
    FLAME_PARAMS,
)


class TestParamSpace:
    def test_dualmirakl_params_complete(self):
        expected = {"alpha", "K", "threshold", "dampening",
                    "susceptibility", "resilience", "logistic_k"}
        assert set(DUALMIRAKL_PARAMS.keys()) == expected

    def test_flame_params_complete(self):
        expected = {"flame_kappa", "flame_influencer_weight",
                    "flame_drift_sigma", "flame_interaction_radius",
                    "flame_sub_steps"}
        assert set(FLAME_PARAMS.keys()) == expected

    def test_all_params_have_bounds(self):
        for name, spec in {**DUALMIRAKL_PARAMS, **FLAME_PARAMS}.items():
            assert "low" in spec, f"{name} missing 'low'"
            assert "high" in spec, f"{name} missing 'high'"
            assert "type" in spec, f"{name} missing 'type'"
            assert spec["low"] < spec["high"], f"{name}: low >= high"

    def test_bounds_match_dualmirakl_ranges(self):
        # alpha range should cover the default 0.15
        assert DUALMIRAKL_PARAMS["alpha"]["low"] <= 0.15
        assert DUALMIRAKL_PARAMS["alpha"]["high"] >= 0.15
        # K should cover default 4
        assert DUALMIRAKL_PARAMS["K"]["low"] <= 4
        assert DUALMIRAKL_PARAMS["K"]["high"] >= 4

    def test_flame_kappa_allows_negative(self):
        # kappa < 0 = polarization, must be in search space
        assert FLAME_PARAMS["flame_kappa"]["low"] < 0


class TestSurrogateObjective:
    """Test surrogate objective directly using a mock trial."""

    def test_runs_without_optuna(self):
        """Surrogate objective uses update_score, not optuna internals."""
        from simulation.sim_loop import update_score
        # Just verify update_score works with the param ranges
        score = update_score(0.3, 0.6, 1.0, 0.15,
                             mode="logistic", logistic_k=6.0,
                             susceptibility=0.5, resilience=0.2)
        assert 0.0 <= score <= 1.0

    def test_param_types(self):
        for name, spec in DUALMIRAKL_PARAMS.items():
            assert spec["type"] in ("float", "int"), f"{name}: bad type {spec['type']}"
        for name, spec in FLAME_PARAMS.items():
            assert spec["type"] in ("float", "int"), f"{name}: bad type {spec['type']}"


optuna = pytest.importorskip("optuna", reason="optuna not installed")


class TestOptunaIntegration:
    """Test actual Optuna integration (skipped if optuna not installed)."""

    def test_fast_mode(self):
        from simulation.optimize import run_optimization

        # Run just 3 trials to verify it works
        study = run_optimization(
            mode="fast", n_trials=3,
            n_ticks=6, include_flame=False,
        )
        assert study.best_value is not None
        assert len(study.trials) == 3
        assert "alpha" in study.best_params

    def test_fast_mode_with_flame(self):
        from simulation.optimize import run_optimization

        study = run_optimization(
            mode="fast", n_trials=3,
            n_ticks=6, include_flame=True,
        )
        assert "flame_kappa" in study.best_params
        assert "alpha" in study.best_params
        assert study.best_trial.user_attrs.get("flame_pop_mean") is not None
