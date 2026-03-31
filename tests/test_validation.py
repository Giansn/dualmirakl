"""Tests for stats/validation.py — bootstrap_ci standalone tests."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from stats.validation import bootstrap_ci


class TestBootstrapCI:
    def test_determinism(self):
        """Same input + same seed -> identical CI."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result1 = bootstrap_ci(values)
        result2 = bootstrap_ci(values)
        assert result1["ci_lower"] == result2["ci_lower"]
        assert result1["ci_upper"] == result2["ci_upper"]
        assert result1["point_estimate"] == result2["point_estimate"]
        assert result1["ci_width"] == result2["ci_width"]

    def test_narrow_ci_for_tight_data(self):
        """Values with low variance -> narrow CI width."""
        tight = np.array([5.0, 5.01, 4.99, 5.0, 5.02, 4.98])
        result = bootstrap_ci(tight)
        assert result["ci_width"] < 0.1

    def test_wide_ci_for_spread_data(self):
        """Values with high variance -> wider CI width."""
        tight = np.array([5.0, 5.01, 4.99, 5.0, 5.02, 4.98])
        spread = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        tight_result = bootstrap_ci(tight)
        spread_result = bootstrap_ci(spread)
        assert spread_result["ci_width"] > tight_result["ci_width"]

    def test_single_value(self):
        """Single value should still work (CI width ~0)."""
        values = np.array([42.0])
        result = bootstrap_ci(values)
        assert result["point_estimate"] == 42.0
        assert result["ci_width"] == pytest.approx(0.0, abs=1e-10)

    def test_custom_statistic(self):
        """Pass statistic=np.median, verify it works."""
        values = np.array([1.0, 2.0, 3.0, 100.0])
        result = bootstrap_ci(values, statistic=np.median)
        assert result["point_estimate"] == np.median(values)
        assert result["ci_lower"] <= result["ci_upper"]

    def test_ci_contains_point_estimate(self):
        """CI should contain the point estimate."""
        rng = np.random.RandomState(123)
        values = rng.normal(loc=5.0, scale=2.0, size=50)
        result = bootstrap_ci(values)
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
