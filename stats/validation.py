"""
Statistical validation of simulation results.

Convergence checks, bootstrap confidence intervals,
multi-run consistency.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def convergence_check(
    series: np.ndarray,
    window: int = 20,
    threshold: float = 0.01,
) -> dict:
    """
    Checks whether a time series has converged.

    Compares variance of the last `window` values against total variance.
    """
    if len(series) < window * 2:
        return {"converged": False, "reason": "Too few data points"}

    total_var = float(np.var(series))
    tail_var = float(np.var(series[-window:]))
    ratio = tail_var / total_var if total_var > 0 else 0.0

    # Geweke diagnostic: compare first third vs last third
    n = len(series)
    first = series[:n // 3]
    last = series[-n // 3:]
    t_stat, p_value = sp_stats.ttest_ind(first, last)

    return {
        "converged": ratio < threshold and p_value > 0.05,
        "variance_ratio": ratio,
        "geweke_t": float(t_stat),
        "geweke_p": float(p_value),
        "tail_mean": float(np.mean(series[-window:])),
        "tail_std": float(np.std(series[-window:])),
    }


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    statistic=np.mean,
) -> dict:
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(42)
    n = len(values)
    boot_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = values[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic(sample)

    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": float(statistic(values)),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_width": upper - lower,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
    }


def multi_run_consistency(
    run_results: list[dict],
    key: str,
) -> dict:
    """
    Checks consistency of a metric across multiple sim runs.

    run_results: list of result dicts from different runs.
    key: which metric to compare (e.g., "final_polarization").
    """
    values = np.array([r[key] for r in run_results if key in r])
    n = len(values)

    if n < 3:
        return {"error": f"Need at least 3 runs, have {n}"}

    cv = float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float("inf")
    ci = bootstrap_ci(values)

    _, normality_p = sp_stats.shapiro(values) if n <= 50 else (0, 0)

    return {
        "n_runs": n,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "cv": cv,
        "is_consistent": cv < 0.15,
        "bootstrap_ci": ci,
        "normality_p": float(normality_p),
        "range": [float(np.min(values)), float(np.max(values))],
    }
