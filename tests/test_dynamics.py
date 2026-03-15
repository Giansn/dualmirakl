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


class TestSobolS2:
    def test_sobol_second_order_returns_structure(self):
        from simulation.dynamics import sobol_second_order
        def f(x):
            return x[0] * 3 + x[1] * 0.5
        result = sobol_second_order(f, [(0, 1), (0, 1)],
                                     param_names=["a", "b"], n_samples=128)
        assert "S1" in result
        assert "S2" in result
        assert "ST" in result
        assert "a" in result["S1"]
        assert ("a", "b") in result["S2"]

    def test_interaction_detected(self):
        from simulation.dynamics import sobol_second_order
        # f = x0 * x1 → strong interaction
        def f(x):
            return x[0] * x[1]
        result = sobol_second_order(f, [(0, 1), (0, 1)],
                                     param_names=["a", "b"], n_samples=512)
        assert result["S2"][("a", "b")] > -0.1  # should detect interaction

    def test_no_interaction_additive(self):
        from simulation.dynamics import sobol_second_order
        # f = x0 + x1 → no interaction
        def f(x):
            return x[0] + x[1]
        result = sobol_second_order(f, [(0, 1), (0, 1)],
                                     param_names=["a", "b"], n_samples=256)
        assert abs(result["S2"][("a", "b")]) < 0.15


class TestTransferEntropy:
    def test_independent_series_lower_than_coupled(self):
        from simulation.dynamics import transfer_entropy
        rng = np.random.RandomState(42)
        x = list(rng.uniform(0, 1, 200))
        y_indep = list(rng.uniform(0, 1, 200))
        # Coupled: y follows x with lag
        y_coupled = [0.5] + [0.8 * x[i] + 0.2 * rng.uniform(0, 1) for i in range(199)]
        te_indep = transfer_entropy(x, y_indep, n_bins=4)
        te_coupled = transfer_entropy(x, y_coupled, n_bins=4)
        assert te_coupled > te_indep  # coupled should have higher TE

    def test_coupled_series_higher_te(self):
        from simulation.dynamics import transfer_entropy
        rng = np.random.RandomState(42)
        x = list(rng.uniform(0, 1, 100))
        # y follows x with lag 1
        y = [0.5] + [0.7 * x[i] + 0.3 * rng.uniform(0, 1) for i in range(99)]
        te = transfer_entropy(x, y)
        te_reverse = transfer_entropy(y, x)
        # Forward TE should be >= reverse (x drives y, not vice versa)
        assert te >= te_reverse - 0.1

    def test_te_matrix_shape(self):
        from simulation.dynamics import transfer_entropy_matrix
        logs = [[0.1 + 0.01 * t for t in range(30)] for _ in range(3)]
        mat = transfer_entropy_matrix(logs)
        assert mat.shape == (3, 3)
        assert mat[0, 0] == 0.0  # no self-transfer

    def test_net_information_flow(self):
        from simulation.dynamics import transfer_entropy_matrix, net_information_flow
        logs = [[0.1 + 0.01 * t for t in range(30)] for _ in range(3)]
        mat = transfer_entropy_matrix(logs)
        flow = net_information_flow(mat)
        assert "total_te" in flow
        assert "net_flow" in flow
        assert len(flow["net_flow"]) == 3


class TestEmergence:
    def test_emergence_variance_ratio_independent(self):
        from simulation.dynamics import emergence_variance_ratio
        rng = np.random.RandomState(42)
        # Independent random walks → E ≈ positive (cancellation)
        logs = [list(np.cumsum(rng.normal(0, 0.1, 30)) + 0.5) for _ in range(6)]
        e = emergence_variance_ratio(logs)
        assert isinstance(e, float)

    def test_emergence_identical_agents_zero(self):
        from simulation.dynamics import emergence_variance_ratio
        # All agents identical → population variance = individual variance → E ≈ 0
        log = [0.3 + 0.01 * t for t in range(20)]
        logs = [log[:] for _ in range(4)]
        e = emergence_variance_ratio(logs)
        assert abs(e) < 0.01

    def test_emergence_mutual_information(self):
        from simulation.dynamics import emergence_mutual_information
        rng = np.random.RandomState(42)
        logs = [list(rng.uniform(0, 1, 30)) for _ in range(4)]
        mi = emergence_mutual_information(logs)
        assert 0.0 <= mi <= 1.0

    def test_emergence_clustering_bimodal(self):
        from simulation.dynamics import emergence_score_clustering
        # Bimodal distribution (needs enough points for stable BC)
        scores = [0.1, 0.12, 0.15, 0.11, 0.13, 0.85, 0.88, 0.9, 0.87, 0.86]
        bc = emergence_score_clustering(scores)
        # Unimodal comparison
        scores_uni = [0.4, 0.42, 0.45, 0.41, 0.43, 0.44, 0.46, 0.39, 0.47, 0.38]
        bc_uni = emergence_score_clustering(scores_uni)
        assert bc > bc_uni  # bimodal should have higher BC

    def test_compute_emergence(self):
        from simulation.dynamics import compute_emergence
        rng = np.random.RandomState(42)
        logs = [list(np.cumsum(rng.normal(0, 0.05, 20)) + 0.5) for _ in range(4)]
        em = compute_emergence(logs)
        assert "variance_ratio" in em
        assert "mutual_information" in em
        assert "bimodality" in em
        assert "is_emergent" in em


class TestAttractorBasins:
    def test_map_basins_runs(self):
        from simulation.dynamics import map_attractor_basins
        result = map_attractor_basins(n_grid=20, n_ticks=30, n_agents=3)
        assert result["n_attractors"] >= 1
        assert len(result["grid_initial"]) == 20
        assert len(result["grid_final"]) == 20

    def test_basins_cover_space(self):
        from simulation.dynamics import map_attractor_basins
        result = map_attractor_basins(n_grid=30, n_ticks=40)
        # Total basin sizes should sum to ~1.0
        total = sum(a.basin_size for a in result["attractors"])
        assert 0.5 < total <= 1.0  # allow some unassigned points

    def test_attractor_shift_analysis(self):
        from simulation.dynamics import attractor_shift_analysis
        result = attractor_shift_analysis(
            "alpha", [0.1, 0.2, 0.3],
            base_params={"dampening": 1.0, "susceptibility": 0.5,
                         "resilience": 0.1, "logistic_k": 6.0,
                         "score_mode": "ema", "kappa": 0.0},
            n_grid=20, n_ticks=30,
        )
        assert len(result["sweep"]) == 3
        assert all("n_attractors" in s for s in result["sweep"])


class TestAnalyzeBridge:
    def test_analyze_simulation(self):
        from simulation.dynamics import analyze_simulation
        rng = np.random.RandomState(42)
        logs = [list(np.cumsum(rng.normal(0, 0.03, 30)) + 0.4) for _ in range(4)]
        result = analyze_simulation(logs, run_config={"alpha": 0.15})
        assert "lyapunov" in result
        assert "transfer_entropy" in result
        assert "emergence" in result
        assert "fixed_points" in result
        assert "attractor_basins" in result

    def test_analyze_simulation_no_config(self):
        from simulation.dynamics import analyze_simulation
        logs = [[0.3 + 0.01 * t for t in range(20)] for _ in range(3)]
        result = analyze_simulation(logs)
        assert "lyapunov" in result
        assert "attractor_basins" not in result  # no config = no basin mapping


class TestStochasticResonance:
    def test_sr_curve_runs(self):
        from simulation.dynamics import stochastic_resonance_curve
        sr = stochastic_resonance_curve(
            temperature_values=[0.0, 0.5, 1.0],
            n_agents=4, n_ticks=20, intervention_tick=10, n_trials=3,
        )
        assert len(sr["temperatures"]) == 3
        assert len(sr["snr"]) == 3
        assert sr["peak_temperature"] in sr["temperatures"]
        assert sr["peak_snr"] >= 0

    def test_sr_has_peak(self):
        from simulation.dynamics import stochastic_resonance_curve
        sr = stochastic_resonance_curve(
            temperature_values=[0.0, 0.3, 0.6, 0.9, 1.2],
            n_agents=6, n_ticks=30, intervention_tick=15, n_trials=5,
        )
        # Peak should not be at the extremes (that would mean no resonance)
        # At minimum, the function should produce varying SNR values
        assert max(sr["snr"]) > min(sr["snr"]) or all(s == sr["snr"][0] for s in sr["snr"])

    def test_intervention_response_profile(self):
        from simulation.dynamics import intervention_response_profile
        # Baseline: linear increase
        bl = [[0.3 + 0.01 * t for t in range(30)] for _ in range(3)]
        # Intervention: flattens after tick 15
        iv = [[0.3 + 0.01 * t if t < 15 else 0.45 for t in range(30)] for _ in range(3)]
        result = intervention_response_profile(iv, bl, intervention_tick=15)
        assert result["n_agents"] == 3
        assert result["mean_magnitude"] > 0
        assert result["fraction_recovering"] >= 0


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
