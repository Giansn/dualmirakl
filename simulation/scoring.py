"""
Pluggable score engines for dualmirakl.

Extracts score update logic from sim_loop.py into standalone, configurable
engines. Each engine implements the same interface but uses different dynamics.

Phase 3 of the general-purpose refactoring.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


class ScoreEngine:
    """
    Base class for score update engines.

    Subclasses implement .update() with mode-specific dynamics.
    All engines share:
      - alpha: learning rate
      - susceptibility/resilience: per-agent heterogeneity modifiers
      - dampening: intervention-applied dampening
    """

    def __init__(self, alpha: float = 0.15, **params):
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
        effective_signal = current + susceptibility * (signal - current)
        effective_dampening = dampening * (1.0 - resilience)
        delta = self.alpha * (effective_signal - current)
        return float(np.clip(current + delta * effective_dampening, 0.0, 1.0))


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
        effective_signal = current + susceptibility * (signal - current)
        midpoint = 0.5
        effective_signal = 1.0 / (1.0 + np.exp(-self.logistic_k * (effective_signal - midpoint)))
        effective_dampening = dampening * (1.0 - resilience)
        delta = self.alpha * (effective_signal - current)
        return float(np.clip(current + delta * effective_dampening, 0.0, 1.0))
