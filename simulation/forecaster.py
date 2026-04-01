"""
Trajectory forecasting for dualmirakl agents.

Uses NeuralProphet for online score trajectory forecasting, changepoint
detection, and trend decomposition. Gives the observer and authority
predictive capability: "agent 3 will cross threshold in ~4 ticks."

Falls back to simple linear extrapolation if NeuralProphet is unavailable.

Usage:
    from simulation.forecaster import TrajectoryForecaster

    fc = TrajectoryForecaster(horizon=4)
    fc.update("agent_0", tick=5, score=0.42)
    forecast = fc.forecast("agent_0")
    # {predicted: [0.45, 0.48, 0.51, 0.53], trend: "rising",
    #  changepoints: [3], threshold_crossing: {0.5: 3}}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_HAS_PROPHET = False
try:
    from neuralprophet import NeuralProphet, set_log_level
    set_log_level("ERROR")
    # Suppress PyTorch Lightning progress bars
    import pytorch_lightning as _pl
    _pl.Trainer.enable_progress_bar = False
    _HAS_PROPHET = True
except ImportError:
    logger.debug("neuralprophet not available; using linear fallback")


@dataclass
class Forecast:
    """Forecast result for a single agent."""
    agent_id: str
    predicted: list[float]
    trend: str  # "rising", "falling", "stable"
    changepoints: list[int]  # tick numbers where behavior shifted
    trend_slope: float  # score change per tick
    confidence_upper: list[float]
    confidence_lower: list[float]
    threshold_crossings: dict[float, Optional[int]]  # {threshold: ticks_until}
    decomposition: Optional[dict] = None  # trend/season/residual components


class TrajectoryForecaster:
    """
    Online trajectory forecaster for agent behavioral scores.

    Maintains per-agent score histories, fits lightweight models on demand,
    and provides forecasts + changepoint detection + decomposition.
    """

    def __init__(
        self,
        horizon: int = 4,
        min_history: int = 6,
        refit_interval: int = 5,
        thresholds: Optional[list[float]] = None,
    ):
        """
        Args:
            horizon: How many ticks ahead to forecast.
            min_history: Minimum ticks before forecasting starts.
            refit_interval: Refit model every N new data points.
            thresholds: Score thresholds to check for crossings.
        """
        self.horizon = horizon
        self.min_history = min_history
        self.refit_interval = refit_interval
        self.thresholds = thresholds or [0.3, 0.5, 0.7, 0.8]

        # Per-agent state
        self._histories: dict[str, list[float]] = {}
        self._models: dict[str, object] = {}
        self._updates_since_fit: dict[str, int] = {}

    def update(self, agent_id: str, tick: int, score: float):
        """Record a new score observation for an agent."""
        if agent_id not in self._histories:
            self._histories[agent_id] = []
            self._updates_since_fit[agent_id] = 0
        self._histories[agent_id].append(score)
        self._updates_since_fit[agent_id] += 1

    def update_batch(self, tick: int, scores: dict[str, float]):
        """Record scores for multiple agents at once."""
        for agent_id, score in scores.items():
            self.update(agent_id, tick, score)

    def forecast(self, agent_id: str) -> Optional[Forecast]:
        """
        Forecast an agent's future trajectory.

        Returns None if insufficient history.
        """
        history = self._histories.get(agent_id)
        if not history or len(history) < self.min_history:
            return None

        scores = np.array(history)

        # Decide whether to use NeuralProphet or linear fallback
        needs_refit = (
            self._updates_since_fit.get(agent_id, 0) >= self.refit_interval
            or agent_id not in self._models
        )

        if _HAS_PROPHET and len(history) >= 10 and needs_refit:
            return self._forecast_prophet(agent_id, scores)
        else:
            return self._forecast_linear(agent_id, scores)

    def forecast_all(self) -> dict[str, Forecast]:
        """Forecast all agents with sufficient history."""
        results = {}
        for agent_id in self._histories:
            fc = self.forecast(agent_id)
            if fc is not None:
                results[agent_id] = fc
        return results

    def detect_changepoints(self, agent_id: str, sensitivity: float = 1.5) -> list[int]:
        """
        Detect behavioral regime changes via CUSUM-like method.

        Args:
            sensitivity: Multiplier on std for detection threshold.

        Returns:
            List of tick indices where changepoints were detected.
        """
        history = self._histories.get(agent_id)
        if not history or len(history) < 5:
            return []

        scores = np.array(history)
        diffs = np.diff(scores)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs) + 1e-10
        threshold = sensitivity * std_diff

        changepoints = []
        cusum_pos = 0.0
        cusum_neg = 0.0

        for i, d in enumerate(diffs):
            cusum_pos = max(0, cusum_pos + d - mean_diff - threshold / 2)
            cusum_neg = max(0, cusum_neg - d + mean_diff - threshold / 2)
            if cusum_pos > threshold or cusum_neg > threshold:
                changepoints.append(i + 1)  # tick index
                cusum_pos = 0.0
                cusum_neg = 0.0

        return changepoints

    def get_context_for_observer(self, tick: int) -> dict:
        """
        Generate forecast summary for observer prompt injection.

        Returns a dict suitable for adding to the observer's context.
        """
        forecasts = self.forecast_all()
        if not forecasts:
            return {}

        context = {
            "forecast_tick": tick,
            "horizon": self.horizon,
            "agents": {},
        }

        for agent_id, fc in forecasts.items():
            agent_ctx = {
                "trend": fc.trend,
                "slope": round(fc.trend_slope, 4),
                "predicted_next": round(fc.predicted[0], 3) if fc.predicted else None,
                "predicted_end": round(fc.predicted[-1], 3) if fc.predicted else None,
            }
            if fc.changepoints:
                agent_ctx["recent_changepoint"] = fc.changepoints[-1]
            if fc.threshold_crossings:
                crossings = {
                    str(k): v for k, v in fc.threshold_crossings.items()
                    if v is not None
                }
                if crossings:
                    agent_ctx["threshold_crossings"] = crossings
            context["agents"][agent_id] = agent_ctx

        # Aggregate warnings
        warnings = []
        for agent_id, fc in forecasts.items():
            for thresh, ticks_until in fc.threshold_crossings.items():
                if ticks_until is not None and ticks_until <= self.horizon:
                    warnings.append(
                        f"{agent_id} predicted to cross {thresh} in ~{ticks_until} ticks"
                    )
        if warnings:
            context["warnings"] = warnings

        return context

    # ── NeuralProphet forecasting ────────────────────────────────────────

    def _forecast_prophet(self, agent_id: str, scores: np.ndarray) -> Forecast:
        """Full NeuralProphet forecast with decomposition."""
        import pandas as pd

        # NeuralProphet expects 'ds' (datetime) and 'y' columns
        # Use integer ticks as dates (1 tick = 1 day for Prophet's internal math)
        n = len(scores)
        df = pd.DataFrame({
            "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
            "y": scores,
        })

        model = NeuralProphet(
            n_forecasts=self.horizon,
            n_lags=min(n // 2, 10),
            changepoints_range=0.8,
            trend_reg=0.1,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            epochs=30,
            batch_size=min(32, n),
            learning_rate=0.1,
        )

        model.fit(df, freq="D", progress=False)
        self._models[agent_id] = model
        self._updates_since_fit[agent_id] = 0

        # Forecast
        future = model.make_future_dataframe(df, periods=self.horizon, n_historic_predictions=n)
        forecast_df = model.predict(future)

        # Extract predictions (last `horizon` rows)
        yhat_cols = [c for c in forecast_df.columns if c.startswith("yhat")]
        if yhat_cols:
            last_row = forecast_df.iloc[-1]
            predicted = [float(last_row[c]) for c in sorted(yhat_cols)]
        else:
            predicted = self._linear_extrapolate(scores, self.horizon)

        predicted = [float(np.clip(p, 0.0, 1.0)) for p in predicted]

        # Trend from model
        trend_slope = self._compute_slope(scores)
        trend = self._classify_trend(trend_slope)

        # Changepoints
        changepoints = self.detect_changepoints(agent_id)

        # Confidence bounds (simple ±2*residual_std)
        residuals = scores - forecast_df["yhat1"].values[:n] if "yhat1" in forecast_df.columns else np.zeros(n)
        res_std = float(np.std(residuals)) if len(residuals) > 0 else 0.05
        confidence_upper = [min(1.0, p + 2 * res_std) for p in predicted]
        confidence_lower = [max(0.0, p - 2 * res_std) for p in predicted]

        # Threshold crossings
        threshold_crossings = self._check_crossings(scores[-1], predicted)

        # Decomposition
        decomposition = None
        if "trend" in forecast_df.columns:
            decomposition = {
                "trend": forecast_df["trend"].values[-self.horizon:].tolist(),
            }
            season_cols = [c for c in forecast_df.columns if "season" in c]
            if season_cols:
                decomposition["seasonality"] = forecast_df[season_cols[0]].values[-self.horizon:].tolist()

        return Forecast(
            agent_id=agent_id,
            predicted=predicted,
            trend=trend,
            changepoints=changepoints,
            trend_slope=trend_slope,
            confidence_upper=confidence_upper,
            confidence_lower=confidence_lower,
            threshold_crossings=threshold_crossings,
            decomposition=decomposition,
        )

    # ── Linear fallback ──────────────────────────────────────────────────

    def _forecast_linear(self, agent_id: str, scores: np.ndarray) -> Forecast:
        """Simple linear extrapolation fallback."""
        predicted = self._linear_extrapolate(scores, self.horizon)
        predicted = [float(np.clip(p, 0.0, 1.0)) for p in predicted]

        trend_slope = self._compute_slope(scores)
        trend = self._classify_trend(trend_slope)
        changepoints = self.detect_changepoints(agent_id)

        # Confidence from recent residual variance
        if len(scores) > 3:
            recent_std = float(np.std(np.diff(scores[-5:])))
        else:
            recent_std = 0.05

        confidence_upper = [min(1.0, p + 2 * recent_std) for p in predicted]
        confidence_lower = [max(0.0, p - 2 * recent_std) for p in predicted]

        threshold_crossings = self._check_crossings(scores[-1], predicted)

        return Forecast(
            agent_id=agent_id,
            predicted=predicted,
            trend=trend,
            changepoints=changepoints,
            trend_slope=trend_slope,
            confidence_upper=confidence_upper,
            confidence_lower=confidence_lower,
            threshold_crossings=threshold_crossings,
        )

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _linear_extrapolate(scores: np.ndarray, horizon: int) -> list[float]:
        """Weighted linear regression extrapolation."""
        n = len(scores)
        x = np.arange(n, dtype=float)
        # Weight recent points more heavily
        weights = np.exp(np.linspace(-1, 0, n))
        w_sum = weights.sum()
        x_mean = np.sum(weights * x) / w_sum
        y_mean = np.sum(weights * scores) / w_sum
        slope = np.sum(weights * (x - x_mean) * (scores - y_mean)) / (
            np.sum(weights * (x - x_mean) ** 2) + 1e-10
        )
        intercept = y_mean - slope * x_mean

        future_x = np.arange(n, n + horizon, dtype=float)
        return (slope * future_x + intercept).tolist()

    @staticmethod
    def _compute_slope(scores: np.ndarray) -> float:
        """Compute recent trend slope (last 5 points or all)."""
        window = scores[-5:] if len(scores) > 5 else scores
        x = np.arange(len(window), dtype=float)
        if len(x) < 2:
            return 0.0
        slope = np.polyfit(x, window, 1)[0]
        return float(slope)

    @staticmethod
    def _classify_trend(slope: float) -> str:
        if slope > 0.01:
            return "rising"
        elif slope < -0.01:
            return "falling"
        return "stable"

    def _check_crossings(
        self, current: float, predicted: list[float]
    ) -> dict[float, Optional[int]]:
        """Check when predicted trajectory crosses each threshold."""
        crossings = {}
        for thresh in self.thresholds:
            crossed_at = None
            for i, p in enumerate(predicted):
                # Rising through threshold or falling through it
                if (current < thresh <= p) or (current > thresh >= p):
                    crossed_at = i + 1  # ticks until crossing
                    break
            crossings[thresh] = crossed_at
        return crossings
