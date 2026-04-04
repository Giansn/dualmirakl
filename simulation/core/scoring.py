"""
Pluggable score engines for dualmirakl.

OOP wrappers around the canonical update_score() in signal_computation.py.
Each engine holds its config and delegates to the single source of truth.
"""

from __future__ import annotations

from simulation.signal.computation import update_score


class ScoreEngine:
    """
    Base class for score update engines.

    Subclasses configure mode-specific parameters; all delegate to
    signal_computation.update_score() for the actual math.
    """

    def __init__(self, alpha: float = 0.15, **params):
        if alpha == 0.0:
            import logging
            logging.getLogger(__name__).warning(
                "ScoreEngine alpha=0.0: scores will never update. "
                "This is likely a configuration error."
            )
        self.alpha = alpha
        self.params = params

    def update(
        self,
        current: float,
        signal: float,
        dampening: float = 1.0,
        susceptibility: float = 1.0,
        resilience: float = 0.0,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def from_config(scoring_config) -> ScoreEngine:
        """
        Factory: create a ScoreEngine from a ScoringConfig object.
        Accepts either a Pydantic model or a plain dict.
        """
        if hasattr(scoring_config, "mode"):
            mode = scoring_config.mode
            params = dict(scoring_config.parameters) if hasattr(scoring_config, "parameters") else {}
        else:
            mode = scoring_config.get("mode", "ema")
            params = dict(scoring_config.get("parameters", {}))

        alpha = params.pop("alpha", 0.15)

        if mode == "ema":
            return EMAScoreEngine(alpha=alpha, **params)
        elif mode == "logistic":
            return LogisticScoreEngine(alpha=alpha, **params)
        else:
            raise ValueError(f"Unknown scoring mode: {mode}. Use 'ema' or 'logistic'.")


class EMAScoreEngine(ScoreEngine):
    """
    Exponential moving average score update.
    Score += d * alpha * (signal - score)
    Analytically tractable, suitable for SA sweeps.
    """

    def update(
        self,
        current: float,
        signal: float,
        dampening: float = 1.0,
        susceptibility: float = 1.0,
        resilience: float = 0.0,
    ) -> float:
        return update_score(
            current, signal, dampening, self.alpha,
            mode="ema", susceptibility=susceptibility, resilience=resilience,
        )


class LogisticScoreEngine(ScoreEngine):
    """
    Sigmoid-transformed score update.
    Captures saturation at extremes: deeply engaged agents resist both
    intervention and further escalation.
    """

    def __init__(self, alpha: float = 0.15, logistic_k: float = 6.0, **params):
        super().__init__(alpha=alpha, **params)
        self.logistic_k = logistic_k

    def update(
        self,
        current: float,
        signal: float,
        dampening: float = 1.0,
        susceptibility: float = 1.0,
        resilience: float = 0.0,
    ) -> float:
        return update_score(
            current, signal, dampening, self.alpha,
            mode="logistic", logistic_k=self.logistic_k,
            susceptibility=susceptibility, resilience=resilience,
        )
