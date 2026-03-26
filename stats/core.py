"""
Core metrics for simulation analysis.

All functions operate on numpy arrays or DuckDB queries.
No LLM involved — pure math.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def stance_drift(stances: np.ndarray) -> dict:
    """
    Measures how much agent positions shift over time.

    stances: shape (T, N) — stance values per tick and agent.
    """
    T, N = stances.shape
    abs_drift = stances[-1] - stances[0]
    tick_changes = np.diff(stances, axis=0)
    volatility = np.sum(np.abs(tick_changes), axis=0)
    sign_changes = np.sum(np.diff(np.sign(tick_changes), axis=0) != 0, axis=0)

    return {
        "abs_drift_mean": float(np.mean(np.abs(abs_drift))),
        "abs_drift_std": float(np.std(abs_drift)),
        "max_drifter": int(np.argmax(np.abs(abs_drift))),
        "max_drift_value": float(np.max(np.abs(abs_drift))),
        "mean_volatility": float(np.mean(volatility)),
        "mean_sign_changes": float(np.mean(sign_changes)),
        "per_agent_drift": abs_drift.tolist(),
    }


def polarization(stances: np.ndarray) -> dict:
    """
    Variance-based polarization index + bimodality test.

    stances: shape (T, N).
    """
    T, N = stances.shape
    pol_series = np.zeros(T)

    for t in range(T):
        pol_series[t] = float(np.var(stances[t]))

    # Sarle's bimodality coefficient at final tick
    final = stances[-1]
    skew = float(sp_stats.skew(final))
    kurt = float(sp_stats.kurtosis(final, fisher=False))  # Pearson
    bc = (skew**2 + 1) / kurt if kurt > 0 else 0.0

    slope, _intercept, _r, p, _se = sp_stats.linregress(np.arange(T), pol_series)

    return {
        "polarization_series": pol_series.tolist(),
        "final_polarization": float(pol_series[-1]),
        "bimodality_coefficient": bc,
        "is_bimodal": bc > 0.555,
        "trend_slope": float(slope),
        "trend_p_value": float(p),
        "trend_direction": ("increasing" if slope > 0 and p < 0.05 else
                            "decreasing" if slope < 0 and p < 0.05 else "stable"),
    }


def opinion_clusters(stances: np.ndarray, max_clusters: int = 5) -> dict:
    """
    K-Means clustering on final stance distribution.
    Identifies opinion groups and their characteristics.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    final = stances[-1].reshape(-1, 1)
    if len(final) < 3:
        return {"n_clusters": 1, "silhouette_score": 0.0, "clusters": [], "labels": []}

    best_k, best_score = 2, -1
    for k in range(2, min(max_clusters + 1, len(final))):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(final)
        score = silhouette_score(final, labels)
        if score > best_score:
            best_k, best_score = k, score

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(final)

    clusters = []
    for i in range(best_k):
        mask = labels == i
        cluster_vals = final[mask].flatten()
        clusters.append({
            "cluster_id": i,
            "size": int(np.sum(mask)),
            "mean_stance": float(np.mean(cluster_vals)),
            "std_stance": float(np.std(cluster_vals)),
        })

    return {
        "n_clusters": best_k,
        "silhouette_score": float(best_score),
        "clusters": sorted(clusters, key=lambda c: c["mean_stance"]),
        "labels": labels.tolist(),
    }


def influence_network(interactions: list[dict]) -> dict:
    """
    Influence network from interaction logs.

    interactions: [{"source": "a1", "target": "a2", "tick": 5, "effect": 0.3}, ...]
    """
    from collections import defaultdict

    edges: dict[tuple, float] = defaultdict(float)
    out_degree: dict[str, float] = defaultdict(float)
    in_degree: dict[str, float] = defaultdict(float)

    for ix in interactions:
        src, tgt = ix["source"], ix["target"]
        effect = abs(ix.get("effect", 0.0))
        edges[(src, tgt)] += effect
        out_degree[src] += effect
        in_degree[tgt] += effect

    all_agents = set(out_degree.keys()) | set(in_degree.keys())
    influencers = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
    influenced = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)

    return {
        "n_agents": len(all_agents),
        "n_edges": len(edges),
        "top_influencers": [{"agent": a, "total_effect": float(e)}
                            for a, e in influencers[:5]],
        "top_influenced": [{"agent": a, "total_effect": float(e)}
                           for a, e in influenced[:5]],
        "density": (len(edges) / (len(all_agents) * (len(all_agents) - 1))
                    if len(all_agents) > 1 else 0.0),
    }
