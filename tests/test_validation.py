"""Tests for stats/validation.py — convergence_check."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from stats.validation import convergence_check


class TestConvergenceCheck:
    def test_converged_series(self):
        """A series that starts noisy then settles should report converged=True."""
        rng = np.random.RandomState(42)
        noisy_start = 5.0 + rng.normal(0, 1.0, size=60)
        settled_tail = 5.0 + rng.normal(0, 0.001, size=40)
        series = np.concatenate([noisy_start, settled_tail])
        result = convergence_check(series)
        assert result["converged"] == True
        assert result["variance_ratio"] < 0.01

    def test_non_converged_drifting(self):
        """A series with linear drift should report converged=False."""
        series = np.linspace(0.0, 10.0, 100)
        result = convergence_check(series)
        assert result["converged"] == False

    def test_too_few_points(self):
        """Series shorter than 2*window should return early with reason."""
        series = np.array([1.0, 2.0, 3.0])
        result = convergence_check(series, window=20)
        assert result["converged"] == False
        assert result["reason"] == "Too few data points"

    def test_constant_series(self):
        """All-same values: variance_ratio=0 but t-test yields NaN, so converged=False."""
        series = np.full(100, 3.14)
        result = convergence_check(series)
        # ttest_ind on identical arrays produces NaN p_value,
        # so the p_value > 0.05 check fails => converged is False
        assert result["converged"] == False
        assert result["variance_ratio"] == 0.0

    def test_late_drift(self):
        """Series that is stationary then suddenly jumps should report converged=False."""
        rng = np.random.RandomState(99)
        stable = 5.0 + rng.normal(0, 0.01, size=80)
        jumped = 50.0 + rng.normal(0, 0.01, size=20)
        series = np.concatenate([stable, jumped])
        result = convergence_check(series)
        assert result["converged"] == False
