"""
Tests for FLAME boot sequence and auto-configuration.
All tests work without pyflamegpu, wandb, or optuna installed.
"""

import pytest

from simulation.flame_setup import FlameContext, flame_status


class TestFlameContext:
    def test_inactive_by_default(self):
        ctx = FlameContext()
        assert not ctx.active
        assert ctx.engine is None
        assert ctx.bridge is None
        assert not ctx.wandb_active
        assert ctx.optuna_study is None

    def test_active_with_engine(self):
        ctx = FlameContext()
        ctx.engine = object()  # mock engine
        assert ctx.active

    def test_shutdown_clears_all(self):
        ctx = FlameContext()
        ctx.engine = object()
        ctx.bridge = object()
        ctx.wandb_active = True
        ctx.optuna_study = object()
        ctx.optuna_trial = object()

        ctx.shutdown()

        assert not ctx.active
        assert ctx.engine is None
        assert ctx.bridge is None
        assert ctx.optuna_study is None
        assert ctx.optuna_trial is None

    def test_shutdown_safe_when_empty(self):
        ctx = FlameContext()
        ctx.shutdown()  # should not raise
        assert not ctx.active


class TestFlameStatus:
    def test_returns_dict(self):
        status = flame_status()
        assert isinstance(status, dict)

    def test_has_required_keys(self):
        status = flame_status()
        assert "flame_enabled" in status
        assert "flame_gpu" in status
        assert "flame_n_population" in status
        assert "pyflamegpu" in status
        assert "wandb" in status
        assert "optuna" in status

    def test_pyflamegpu_status(self):
        status = flame_status()
        assert status["pyflamegpu"] in ("installed", "not installed")

    def test_wandb_status(self):
        status = flame_status()
        assert status["wandb"] in ("installed", "not installed")

    def test_optuna_status(self):
        status = flame_status()
        assert status["optuna"] in ("installed", "not installed")

    def test_default_gpu(self):
        status = flame_status()
        assert status["flame_gpu"] == 2

    def test_default_population(self):
        status = flame_status()
        assert status["flame_n_population"] == 10000


class TestFlameBoot:
    def test_boot_without_pyflamegpu(self):
        """Boot should return inactive context when pyflamegpu not installed."""
        from simulation.flame_setup import flame_boot

        sim_config = {
            "alpha": 0.15, "score_mode": "ema",
            "logistic_k": 6.0, "seed": 42,
        }
        ctx = flame_boot(sim_config, n_participants=4)

        # Without pyflamegpu, engine won't start
        # Context should be returned either way (active or not)
        assert isinstance(ctx, FlameContext)
        # If pyflamegpu is not installed, engine is None
        try:
            import pyflamegpu
            # If installed, engine might be active (depends on GPU)
        except ImportError:
            assert not ctx.active

    def test_boot_returns_context_type(self):
        from simulation.flame_setup import flame_boot
        ctx = flame_boot({"alpha": 0.15, "seed": 42})
        assert isinstance(ctx, FlameContext)
