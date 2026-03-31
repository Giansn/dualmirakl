"""
Proper scoring rules for probabilistic simulation output.

CRPS (Continuous Ranked Probability Score), Brier score, and
Wasserstein distance for evaluating ensemble forecast quality
against observed data.

Usage:
    from stats.scoring import crps, brier_score, wasserstein_dist, score_ensemble
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def crps(forecasts: np.ndarray, observed: float) -> float:
    """
    Continuous Ranked Probability Score (ensemble version).

    The proper scoring rule for probabilistic forecasts. Lower is better.
    CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent draws from the forecast distribution.

    Args:
        forecasts: Array of ensemble forecast values (n_members,).
        observed: The observed/true value.

    Returns:
        CRPS score (non-negative, 0 = perfect).
    """
    forecasts = np.asarray(forecasts, dtype=float)
    n = len(forecasts)
    if n == 0:
        return float("inf")

    # E|X - y|
    term1 = float(np.mean(np.abs(forecasts - observed)))

    # 0.5 * E|X - X'| via sorted formula
    # For sorted X: 0.5*E|X-X'| = 1/(n^2) * sum_i (2i - n + 1) * X_(i)
    # The sorted formula already includes the 0.5 factor.
    if n == 1:
        spread_term = 0.0
    else:
        sorted_f = np.sort(forecasts)
        weights = 2.0 * np.arange(n) - n + 1.0
        spread_term = float(np.sum(weights * sorted_f)) / (n * n)

    return term1 - spread_term


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Brier score for binary outcome predictions.

    BS = mean((p - o)^2) where p is predicted probability, o is 0/1 outcome.
    Range [0, 1]. Lower is better. 0 = perfect calibration.

    Args:
        probabilities: Predicted probabilities (n_events,) in [0, 1].
        outcomes: Binary outcomes (n_events,) in {0, 1}.

    Returns:
        Brier score.
    """
    p = np.asarray(probabilities, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return float("inf")
    return float(np.mean((p - o) ** 2))


def wasserstein_dist(
    simulated: np.ndarray,
    observed: np.ndarray,
) -> float:
    """
    Wasserstein-1 (Earth Mover's) distance between two distributions.

    Measures how much "work" is needed to transform one distribution
    into another. Lower is better.

    Args:
        simulated: Simulated distribution samples.
        observed: Observed distribution samples.

    Returns:
        Wasserstein distance (non-negative).
    """
    return float(sp_stats.wasserstein_distance(
        np.asarray(simulated, dtype=float),
        np.asarray(observed, dtype=float),
    ))


def score_ensemble(
    ensemble_forecasts: dict[str, np.ndarray],
    observed_data: dict[str, float | np.ndarray],
) -> dict[str, dict]:
    """
    Compute all scoring metrics for an ensemble forecast against observations.

    Args:
        ensemble_forecasts: {metric_name: array of ensemble values}.
        observed_data: {metric_name: observed value or distribution}.

    Returns:
        {metric_name: {crps, wasserstein, ...}} for each shared metric.
    """
    results = {}
    for key in set(ensemble_forecasts) & set(observed_data):
        forecasts = np.asarray(ensemble_forecasts[key])
        obs = observed_data[key]

        entry: dict = {}
        if np.isscalar(obs):
            entry["crps"] = round(crps(forecasts, float(obs)), 6)
            entry["mean_forecast"] = round(float(np.mean(forecasts)), 6)
            entry["bias"] = round(float(np.mean(forecasts) - obs), 6)
        if isinstance(obs, np.ndarray) and len(obs) > 1:
            entry["wasserstein"] = round(wasserstein_dist(forecasts, obs), 6)

        # Brier: if metric is binary (threshold exceedance)
        if np.isscalar(obs):
            # P(forecast > observed)
            p_exceed = float(np.mean(forecasts > obs))
            entry["p_exceedance"] = round(p_exceed, 4)

        results[key] = entry

    return results
