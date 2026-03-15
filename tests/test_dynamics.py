"""Tests for dynamics.py — coupled ODE, phase analysis, Lyapunov estimation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest


class TestCoupledDynamics:
    def test_zero_coupling_matches_uncoupled(self):
        from simulation.dynamics import coupled_score_update
        from simulation.sim_loop import update_score
        scores = [0.3, 0.5, 0.7]
        signal = 0.6
        coupled = coupled_score_update(0.3, signal, scores, kappa=0.0, alpha=0.2)
        uncoupled = update_score(0.3, signal, alpha=0.2)
        assert abs(coupled - uncoupled) < 1e-6

    def test_positive_coupling_pulls_toward_mean(self):
        from simulation.dynamics import coupled_score_update
        scores = [0.3, 0.5, 0.7]  # mean = 0.5
        # Agent at 0.3 (below mean) should be pulled up
        low = coupled_score_update(0.3, 0.3, scores, kappa=0.2)
        no_coupling = coupled_score_update(0.3, 0.3, scores, kappa=0.0)
        assert low > no_coupling  # pulled toward mean

    def test_negative_coupling_pushes_away(self):
        from simulation.dynamics import coupled_score_update
        scores = [0.3, 0.5, 0.7]  # mean = 0.5
        # Agent at 0.3 (below mean) should be pushed further down
        low = coupled_score_update(0.3, 0.3, scores, kappa=-0.1)
        no_coupling = coupled_score_update(0.3, 0.3, scores, kappa=0.0)
        assert low < no_coupling  # pushed away from mean

    def test_coupling_modes(self):
        from simulation.dynamics import coupled_score_update
        scores = [0.3, 0.5, 0.7]
        for mode in ["linear", "sigmoid", "threshold"]:
            result = coupled_score_update(0.3, 0.5, scores, kappa=0.1,
                                          coupling_mode=mode)
            assert 0.0 <= result <= 1.0

    def test_batch_update_preserves_count(self):
        from simulation.dynamics import coupled_batch_update
        scores = [0.2, 0.4, 0.6, 0.8]
        signals = [0.5, 0.5, 0.5, 0.5]
        new = coupled_batch_update(scores, signals, kappa=0.1)
        assert len(new) == 4

    def test_strong_coupling_reduces_spread(self):
        from simulation.dynamics import coupled_batch_update
        rng = np.random.RandomState(42)
        scores = [0.1, 0.3, 0.7, 0.9]
        # Run many steps with strong coupling
        for _ in range(20):
            signals = [float(rng.uniform(0.3, 0.7)) for _ in range(4)]
            scores = coupled_batch_update(scores, signals, kappa=0.3, alpha=0.15)
        spread = max(scores) - min(scores)
        assert spread < 0.5  # should converge somewhat

    def test_coupling_matrix_mean_field(self):
        from simulation.dynamics import compute_coupling_matrix
        W = compute_coupling_matrix([0.3, 0.5, 0.7], mode="mean_field")
        assert W.shape == (3, 3)
        assert abs(W[0, 0]) < 1e-10  # no self-coupling
        assert abs(W[0, 1] - 0.5) < 1e-10  # uniform

    def test_coupling_matrix_distance(self):
        from simulation.dynamics import compute_coupling_matrix
        W = compute_coupling_matrix([0.1, 0.5, 0.9], mode="distance")
        assert W.shape == (3, 3)
        assert abs(W[0, 0]) < 1e-10  # no self-coupling


class TestPhaseAnalysis:
    def test_extract_phase_trajectory(self):
        from simulation.dynamics import extract_phase_trajectory
        log = [0.3, 0.35, 0.38, 0.40, 0.39]
        pts = extract_phase_trajectory(log)
        assert len(pts) == 5
        assert pts[0].score == 0.3
        assert abs(pts[0].velocity - 0.05) < 1e-10  # 0.35 - 0.3

    def test_phase_acceleration(self):
        from simulation.dynamics import extract_phase_trajectory
        # Constant velocity → zero acceleration
        log = [0.1, 0.2, 0.3, 0.4, 0.5]
        pts = extract_phase_trajectory(log)
        for p in pts[1:-1]:
            assert abs(p.acceleration) < 1e-10

    def test_find_fixed_points(self):
        from simulation.dynamics import find_fixed_points
        # Agent that converges
        stable = [0.3, 0.35, 0.38, 0.39, 0.395, 0.397, 0.398, 0.398]
        # Agent that oscillates
        osc = [0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5]
        results = find_fixed_points([stable, osc])
        assert results[0]["is_fixed_point"] is True
        assert results[1]["is_fixed_point"] is False


class TestBifurcation:
    def test_sweep_runs(self):
        from simulation.dynamics import bifurcation_sweep
        result = bifurcation_sweep(
            "kappa", [0.0, 0.1, 0.2, 0.3],
            base_params={"alpha": 0.15, "dampening": 1.0, "susceptibility": 0.5,
                         "resilience": 0.1, "logistic_k": 6.0, "score_mode": "ema",
                         "coupling_mode": "linear"},
            n_agents=4, n_ticks=20,
        )
        assert len(result["distributions"]) == 4
        assert len(result["means"]) == 4

    def test_detect_bifurcation_points(self):
        from simulation.dynamics import detect_bifurcation_points
        # Synthetic sweep with a jump
        sweep = {
            "param_name": "kappa",
            "param_values": [0.0, 0.1, 0.2, 0.3],
            "means": [0.4, 0.41, 0.42, 0.5],
            "stds": [0.1, 0.11, 0.11, 0.25],
            "n_clusters": [1, 1, 1, 2],
            "distributions": [[], [], [], []],
        }
        bifs = detect_bifurcation_points(sweep, std_jump_threshold=0.05)
        assert len(bifs) >= 1
        assert bifs[0]["cluster_change"] is True


class TestLyapunov:
    def test_stable_trajectory(self):
        from simulation.dynamics import lyapunov_exponent_twin
        # Converging trajectories → negative λ
        log1 = [0.5 * (1 - 0.9 ** t) for t in range(30)]
        log2 = [0.5 * (1 - 0.9 ** t) + 0.01 * 0.9 ** t for t in range(30)]
        lam = lyapunov_exponent_twin(log1, log2)
        assert lam < 0  # should be negative (stable)

    def test_diverging_trajectory(self):
        from simulation.dynamics import lyapunov_exponent_twin
        # Diverging trajectories → positive λ
        log1 = [0.3 + 0.01 * t for t in range(20)]
        log2 = [0.3 + 0.01 * t + 0.001 * 1.1 ** t for t in range(20)]
        lam = lyapunov_exponent_twin(log1, log2)
        assert lam > 0  # should be positive (divergent)

    def test_lyapunov_from_timeseries(self):
        from simulation.dynamics import lyapunov_from_timeseries
        # Logistic map (chaotic at r=3.9)
        r = 3.9
        x = 0.1
        log = []
        for _ in range(100):
            x = r * x * (1 - x)
            log.append(x)
        lam = lyapunov_from_timeseries(log)
        assert lam > 0  # logistic map at r=3.9 is chaotic

    def test_lyapunov_from_timeseries_stable(self):
        from simulation.dynamics import lyapunov_from_timeseries
        # Converging series → negative λ
        log = [0.5 + 0.3 * 0.95 ** t for t in range(60)]
        lam = lyapunov_from_timeseries(log)
        assert lam <= 0.05  # should be non-positive

    def test_system_lyapunov(self):
        from simulation.dynamics import estimate_system_lyapunov
        logs = [
            [0.3 + 0.01 * t for t in range(30)],
            [0.5 - 0.005 * t for t in range(30)],
        ]
        result = estimate_system_lyapunov(logs)
        assert "regime" in result
        assert result["regime"] in ("chaotic", "stable", "marginal")
        assert len(result["per_agent"]) == 2

    def test_system_lyapunov_twin_method(self):
        from simulation.dynamics import estimate_system_lyapunov
        logs = [[0.3 + 0.01 * t for t in range(20)]]
        perturbed = [[0.3 + 0.01 * t + 0.001 for t in range(20)]]
        result = estimate_system_lyapunov(logs, method="twin", perturbed_logs=perturbed)
        assert "max_lyapunov" in result
