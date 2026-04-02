"""Tests for simulation.forecaster linear fallback (no NeuralProphet required)."""

import os
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestForecasterLinearFallback:
    """Verify TrajectoryForecaster works without NeuralProphet."""

    def test_forecast_linear_when_prophet_disabled(self):
        """Linear fallback produces valid Forecast with trend + predictions."""
        import simulation.forecaster as fm
        with patch.object(fm, "_HAS_PROPHET", False):
            fc = fm.TrajectoryForecaster(horizon=3, min_history=4)
            for tick, score in enumerate([0.3, 0.35, 0.4, 0.45, 0.5, 0.55]):
                fc.update("a0", tick, score)
            result = fc.forecast("a0")
            assert result is not None
            assert result.agent_id == "a0"
            assert len(result.predicted) == 3
            assert result.trend == "rising"
            # Predictions should continue upward
            assert all(p > 0.5 for p in result.predicted)

    def test_get_context_for_observer_empty_when_no_forecasts(self):
        """get_context_for_observer returns {} with insufficient history."""
        import simulation.forecaster as fm
        with patch.object(fm, "_HAS_PROPHET", False):
            fc = fm.TrajectoryForecaster(horizon=3, min_history=6)
            fc.update("a0", 0, 0.5)
            ctx = fc.get_context_for_observer(tick=1)
            assert ctx == {}
