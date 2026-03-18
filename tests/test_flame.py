"""
Tests for the FLAME GPU 2 population dynamics framework.
Tests bridge logic, config, and snapshots without requiring pyflamegpu/GPU.
"""

import json
import math
import os
import tempfile
import pytest

from simulation.flame.bridge import FlameBridge, PopulationSnapshot
from simulation.flame.engine import DEFAULT_CONFIG, _beta_sample


# ---------------------------------------------------------------------------
# FlameBridge tests
# ---------------------------------------------------------------------------

class TestFlameBridge:
    def test_init_defaults(self):
        bridge = FlameBridge(n_influencers=4, space_size=100.0)
        assert bridge.n_influencers == 4
        assert bridge.space_size == 100.0
        assert len(bridge.snapshots) == 0

    def test_influencer_positions_evenly_spaced(self):
        bridge = FlameBridge(n_influencers=4, space_size=100.0)
        positions = bridge._influencer_positions
        assert len(positions) == 4
        # All positions within bounds
        for x, y in positions:
            assert 0 <= x <= 100
            assert 0 <= y <= 100
        # Positions are distinct
        for i in range(4):
            for j in range(i + 1, 4):
                dist = math.dist(positions[i], positions[j])
                assert dist > 1.0, "Influencer positions should be spread apart"

    def test_influencer_positions_symmetry(self):
        bridge = FlameBridge(n_influencers=4, space_size=100.0)
        positions = bridge._influencer_positions
        cx, cy = 50.0, 50.0
        # All should be same distance from center
        distances = [math.dist(p, (cx, cy)) for p in positions]
        for d in distances:
            assert abs(d - distances[0]) < 1e-6

    def test_influencer_positions_scale_with_space(self):
        b1 = FlameBridge(n_influencers=4, space_size=100.0)
        b2 = FlameBridge(n_influencers=4, space_size=200.0)
        r1 = math.dist(b1._influencer_positions[0], (50, 50))
        r2 = math.dist(b2._influencer_positions[0], (100, 100))
        assert abs(r2 / r1 - 2.0) < 1e-6

    def test_push_score_count_mismatch_raises(self):
        bridge = FlameBridge(n_influencers=4)
        with pytest.raises(ValueError, match="Expected 4 scores"):
            bridge.push_influencer_scores(None, [0.5, 0.5])

    def test_reset_clears_snapshots(self):
        bridge = FlameBridge()
        bridge.snapshots.append(
            PopulationSnapshot(
                tick=0, n_population=100, n_influencers=4,
                sub_steps=10, mean_score=0.3, std_score=0.1,
                min_score=0.1, max_score=0.5,
            )
        )
        assert len(bridge.snapshots) == 1
        bridge.reset()
        assert len(bridge.snapshots) == 0


# ---------------------------------------------------------------------------
# PopulationSnapshot tests
# ---------------------------------------------------------------------------

class TestPopulationSnapshot:
    def test_to_dict(self):
        snap = PopulationSnapshot(
            tick=5, n_population=10000, n_influencers=4,
            sub_steps=10, mean_score=0.45, std_score=0.12,
            min_score=0.01, max_score=0.98,
            influencer_scores=[0.3, 0.5, 0.7, 0.2],
            histogram=[500, 800, 1200, 1500, 1800, 1500, 1200, 800, 500, 200],
        )
        d = snap.to_dict()
        assert d["tick"] == 5
        assert d["n_population"] == 10000
        assert d["mean_score"] == 0.45
        assert len(d["histogram"]) == 10
        assert sum(d["histogram"]) == 10000

    def test_export_import_roundtrip(self):
        bridge = FlameBridge(n_influencers=2, space_size=50.0)
        bridge.snapshots.append(
            PopulationSnapshot(
                tick=0, n_population=100, n_influencers=2,
                sub_steps=5, mean_score=0.3, std_score=0.1,
                min_score=0.05, max_score=0.55,
                influencer_scores=[0.4, 0.6],
                histogram=[10, 15, 20, 15, 10, 10, 8, 5, 4, 3],
            )
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            bridge.export_snapshots(path)
            with open(path) as f:
                data = json.load(f)
            assert data["n_influencers"] == 2
            assert data["space_size"] == 50.0
            assert len(data["snapshots"]) == 1
            assert data["snapshots"][0]["mean_score"] == 0.3
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Coupling feedback tests
# ---------------------------------------------------------------------------

class TestCouplingFeedback:
    def _make_bridge_with_stats(self):
        """Helper: creates a bridge and a mock engine-like stats dict."""
        bridge = FlameBridge(n_influencers=4)
        return bridge

    def test_polarization_calculation(self):
        bridge = self._make_bridge_with_stats()
        # Simulate a polarized population: lots at extremes
        class MockEngine:
            def get_population_stats(self):
                return {
                    "mean_score": 0.5,
                    "std_score": 0.3,
                    "count": 100,
                    "histogram": [30, 5, 5, 5, 5, 5, 5, 5, 5, 30],
                }
        feedback = bridge.get_population_coupling_feedback(MockEngine())
        assert feedback["polarization"] == pytest.approx(0.7, abs=0.01)
        assert feedback["extreme_low_frac"] == pytest.approx(0.35, abs=0.01)
        assert feedback["extreme_high_frac"] == pytest.approx(0.35, abs=0.01)

    def test_no_polarization(self):
        bridge = FlameBridge(n_influencers=4)
        class MockEngine:
            def get_population_stats(self):
                return {
                    "mean_score": 0.5,
                    "std_score": 0.05,
                    "count": 100,
                    "histogram": [0, 0, 5, 20, 25, 25, 20, 5, 0, 0],
                }
        feedback = bridge.get_population_coupling_feedback(MockEngine())
        assert feedback["polarization"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Config / defaults tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config_keys(self):
        expected = {
            "n_population", "n_influencers", "space_size",
            "interaction_radius", "alpha", "kappa", "dampening",
            "influencer_weight", "score_mode", "logistic_k",
            "drift_sigma", "move_speed", "sub_steps", "gpu_id", "seed",
        }
        assert set(DEFAULT_CONFIG.keys()) == expected

    def test_default_matches_dualmirakl(self):
        """Verify FLAME defaults match dualmirakl sim_loop defaults."""
        assert DEFAULT_CONFIG["alpha"] == 0.15
        assert DEFAULT_CONFIG["n_influencers"] == 4
        assert DEFAULT_CONFIG["score_mode"] == "ema"
        assert DEFAULT_CONFIG["logistic_k"] == 6.0

    def test_gpu_id_default(self):
        assert DEFAULT_CONFIG["gpu_id"] == 2  # Third GPU


# ---------------------------------------------------------------------------
# Beta sampling test
# ---------------------------------------------------------------------------

class TestBetaSampling:
    def test_beta_range(self):
        import random
        rng = random.Random(42)
        samples = [_beta_sample(rng, 2, 3) for _ in range(1000)]
        assert all(0 <= s <= 1 for s in samples)
        # Beta(2,3) mean = 2/5 = 0.4
        mean = sum(samples) / len(samples)
        assert abs(mean - 0.4) < 0.05

    def test_resilience_distribution(self):
        import random
        rng = random.Random(42)
        samples = [_beta_sample(rng, 2, 5) for _ in range(1000)]
        # Beta(2,5) mean = 2/7 ≈ 0.286
        mean = sum(samples) / len(samples)
        assert abs(mean - 2 / 7) < 0.05
