"""
Tests for Phase D: scoring metrics and HTML report generation.

Run: python -m pytest tests/test_phase_d.py -v
"""

import sys
import os

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Scoring metrics tests ────────────────────────────────────────────────────

class TestCRPS:

    def test_perfect_forecast(self):
        from stats.scoring import crps
        # All members = observed → CRPS = 0
        score = crps(np.array([0.5, 0.5, 0.5]), 0.5)
        assert abs(score) < 1e-6

    def test_worse_than_perfect(self):
        from stats.scoring import crps
        # Spread-out forecast → positive CRPS
        score = crps(np.array([0.1, 0.5, 0.9]), 0.5)
        assert score > 0

    def test_biased_forecast(self):
        from stats.scoring import crps
        # All members far from observed → large CRPS
        score = crps(np.array([0.9, 0.95, 1.0]), 0.1)
        assert score > 0.5

    def test_single_member(self):
        from stats.scoring import crps
        score = crps(np.array([0.3]), 0.5)
        assert abs(score - 0.2) < 1e-6

    def test_empty(self):
        from stats.scoring import crps
        assert crps(np.array([]), 0.5) == float("inf")

    def test_fair_crps_leq_biased(self):
        from stats.scoring import crps
        forecasts = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        observed = 0.5
        biased = crps(forecasts, observed, fair=False)
        fair = crps(forecasts, observed, fair=True)
        assert fair < biased

    def test_fair_single_member_equals_biased(self):
        from stats.scoring import crps
        # n=1 → spread term is 0 regardless of denominator
        forecasts = np.array([0.3])
        observed = 0.5
        biased = crps(forecasts, observed, fair=False)
        fair = crps(forecasts, observed, fair=True)
        assert abs(fair - biased) < 1e-12

    def test_fair_perfect_still_zero(self):
        from stats.scoring import crps
        forecasts = np.array([0.5, 0.5, 0.5])
        assert abs(crps(forecasts, 0.5, fair=True)) < 1e-12


class TestBrierScore:

    def test_perfect(self):
        from stats.scoring import brier_score
        score = brier_score(np.array([1.0, 0.0, 1.0]), np.array([1, 0, 1]))
        assert abs(score) < 1e-6

    def test_worst(self):
        from stats.scoring import brier_score
        score = brier_score(np.array([0.0, 1.0]), np.array([1, 0]))
        assert abs(score - 1.0) < 1e-6

    def test_calibrated(self):
        from stats.scoring import brier_score
        score = brier_score(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 0, 1, 0]))
        assert abs(score - 0.25) < 1e-6


class TestWasserstein:

    def test_identical(self):
        from stats.scoring import wasserstein_dist
        a = np.array([1.0, 2.0, 3.0])
        assert wasserstein_dist(a, a) < 1e-6

    def test_shifted(self):
        from stats.scoring import wasserstein_dist
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        assert abs(wasserstein_dist(a, b) - 1.0) < 1e-6


class TestScoreEnsemble:

    def test_basic(self):
        from stats.scoring import score_ensemble
        forecasts = {"mean_score": np.array([0.4, 0.5, 0.6])}
        observed = {"mean_score": 0.5}
        result = score_ensemble(forecasts, observed)
        assert "mean_score" in result
        assert "crps" in result["mean_score"]
        assert result["mean_score"]["crps"] >= 0

    def test_no_overlap(self):
        from stats.scoring import score_ensemble
        result = score_ensemble({"a": np.array([1.0])}, {"b": 2.0})
        assert result == {}


# ── HTML report tests ────────────────────────────────────────────────────────

class TestReportGeneration:

    @staticmethod
    def _mock_ensemble():
        return {
            "experiment_id": "test_001",
            "n_runs_completed": 5,
            "n_runs_failed": 0,
            "convergence": {"achieved": True, "final_cv": 0.03, "threshold": 0.05},
            "ensemble_summary": {
                "percentile_bands": {
                    str(i): {
                        "mean": 0.4 + i * 0.02,
                        "std": 0.05,
                        "p5": 0.3 + i * 0.01,
                        "p25": 0.35 + i * 0.015,
                        "median": 0.4 + i * 0.02,
                        "p75": 0.45 + i * 0.025,
                        "p95": 0.5 + i * 0.03,
                        "n_runs": 5,
                    }
                    for i in range(1, 13)
                },
                "metric_values": [0.45, 0.47, 0.44, 0.46, 0.45],
            },
            "variance_decomposition": {
                "var_epistemic": 0.005,
                "var_within": 0.015,
                "var_total": 0.02,
                "pct_epistemic": 0.25,
                "pct_within": 0.75,
            },
        }

    def test_generates_html(self):
        from simulation.report import generate_report
        html = generate_report(ensemble_result=self._mock_ensemble())
        assert "<!DOCTYPE html>" in html
        assert "dualmirakl" in html
        assert "Plotly" in html

    def test_fan_chart_included(self):
        from simulation.report import generate_report
        html = generate_report(ensemble_result=self._mock_ensemble())
        assert "fan-chart" in html
        assert "Fan Chart" in html

    def test_convergence_chart_included(self):
        from simulation.report import generate_report
        html = generate_report(ensemble_result=self._mock_ensemble())
        assert "convergence-plot" in html

    def test_variance_pie_included(self):
        from simulation.report import generate_report
        html = generate_report(ensemble_result=self._mock_ensemble())
        assert "variance-pie" in html
        assert "25.0%" in html  # pct_epistemic

    def test_empty_report(self):
        from simulation.report import generate_report
        html = generate_report()
        assert "<!DOCTYPE html>" in html

    def test_with_scenario_tree(self):
        from simulation.report import generate_report
        tree = {
            "id": "root", "step": 0, "probability": 1.0,
            "metrics": {"mean": 0.5, "std": 0.1},
            "n_supporting_runs": 10,
            "children": [
                {"id": "root.1", "step": 6, "probability": 0.6,
                 "metrics": {"mean": 0.4, "std": 0.05}, "n_supporting_runs": 6, "children": []},
                {"id": "root.2", "step": 6, "probability": 0.4,
                 "metrics": {"mean": 0.7, "std": 0.08}, "n_supporting_runs": 4, "children": []},
            ],
        }
        html = generate_report(scenario_tree=tree)
        assert "root.1" in html
        assert "root.2" in html

    def test_with_calibration(self):
        from simulation.report import generate_report
        cal = {
            "abc_result": {
                "param_names": ["alpha", "kappa"],
                "posterior_mean": [0.15, 0.05],
                "posterior_std": [0.02, 0.01],
                "n_populations": 3,
                "acceptance_rate": 0.12,
            }
        }
        html = generate_report(calibration_result=cal)
        assert "alpha" in html
        assert "kappa" in html

    def test_with_tornado(self):
        from simulation.report import generate_report
        dynamics = {
            "D_sobol_s2": {
                "S1": {"alpha": 0.4, "kappa": 0.3, "susceptibility": 0.15},
                "ST": {"alpha": 0.5, "kappa": 0.35, "susceptibility": 0.2},
                "S2": {},
            }
        }
        html = generate_report(dynamics_analysis=dynamics)
        assert "tornado-chart" in html
