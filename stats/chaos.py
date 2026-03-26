"""
Chaos analysis for simulation time series.

Wraps nolds (Lyapunov, Hurst, Correlation Dimension, Sample Entropy).
Complements simulation/dynamics.py which has custom implementations —
this module uses established nolds algorithms for validation/comparison.

Optional dependency: install via `pip install nolds`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChaosMetrics:
    lyapunov_exponent: float      # >0 = chaotic
    hurst_exponent: float         # >0.5 = persistent, <0.5 = anti-persistent
    correlation_dimension: float  # fractal dimension of attractor
    sample_entropy: float         # time series complexity
    is_chaotic: bool
    is_persistent: bool


def analyze_chaos(series: np.ndarray, emb_dim: int = 10) -> ChaosMetrics:
    """
    Chaos metrics for a univariate time series.

    series: shape (T,) — e.g., mean stance over ticks.
    emb_dim: embedding dimension for delay embedding.

    Requires nolds: pip install nolds
    """
    try:
        import nolds
    except ImportError:
        logger.warning("nolds not installed. pip install nolds")
        return ChaosMetrics(
            lyapunov_exponent=float("nan"), hurst_exponent=float("nan"),
            correlation_dimension=float("nan"), sample_entropy=float("nan"),
            is_chaotic=False, is_persistent=False,
        )

    if len(series) < 50:
        logger.warning("analyze_chaos: need >= 50 data points, got %d", len(series))
        return ChaosMetrics(
            lyapunov_exponent=float("nan"), hurst_exponent=float("nan"),
            correlation_dimension=float("nan"), sample_entropy=float("nan"),
            is_chaotic=False, is_persistent=False,
        )

    try:
        lyap = float(nolds.lyap_r(series, emb_dim=emb_dim, lag=1, min_tsep=20))
    except Exception:
        lyap = float("nan")

    try:
        hurst = float(nolds.hurst_rs(series))
    except Exception:
        hurst = float("nan")

    try:
        corr_dim = float(nolds.corr_dim(series, emb_dim=emb_dim))
    except Exception:
        corr_dim = float("nan")

    try:
        samp_en = float(nolds.sampen(series, emb_dim=2))
    except Exception:
        samp_en = float("nan")

    return ChaosMetrics(
        lyapunov_exponent=lyap,
        hurst_exponent=hurst,
        correlation_dimension=corr_dim,
        sample_entropy=samp_en,
        is_chaotic=not np.isnan(lyap) and lyap > 0.0,
        is_persistent=not np.isnan(hurst) and hurst > 0.5,
    )
