"""
Tests for W&B tracking module.
All tests work without wandb installed — tracker is a no-op.
"""

from simulation.tracking import SimTracker, _std


class TestSimTracker:
    def test_unavailable_is_noop(self):
        t = SimTracker()
        # All methods should be safe to call even without wandb
        assert not t.active
        t.log_tick(1, [0.3, 0.5])
        t.log_summary({"mean": 0.4})
        t.log_artifact("/tmp/fake", "test")
        t.finish()

    def test_init_without_wandb(self):
        t = SimTracker()
        # init_run returns False if wandb not installed
        result = t.init_run({"alpha": 0.15})
        if not t.available:
            assert result is False
            assert not t.active


class TestStd:
    def test_empty(self):
        assert _std([]) == 0.0

    def test_single(self):
        assert _std([5.0]) == 0.0

    def test_known_values(self):
        import math
        # std of [0, 2] = 1.0
        assert abs(_std([0.0, 2.0]) - 1.0) < 1e-6

    def test_identical(self):
        assert _std([0.5, 0.5, 0.5]) == 0.0
