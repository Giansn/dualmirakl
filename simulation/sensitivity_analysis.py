"""Sensitivity analysis: Morris screening, Sobol indices, threshold calibration.

Extracted from sim_loop.py — pure-math SA routines (no LLM calls) plus
intervention threshold calibration using embedding similarity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from simulation.agent_rolesv3 import INTERVENTION_CODEBOOK, INTERVENTION_THRESHOLD
from simulation.signal_computation import (
    _get_embed, _load_anchors, _compute_signal_from_vec, update_score,
)

logger = logging.getLogger(__name__)


def morris_screening(
    func,
    bounds: list[tuple[float, float]],
    r: int = 10,
    p: int = 4,
    seed: int = 42,
) -> dict[int, dict]:
    rng = np.random.RandomState(seed)
    k = len(bounds)
    delta = 1.0 / (p - 1) if p > 1 else 0.5
    effects = {i: [] for i in range(k)}

    for _ in range(r):
        x_base = rng.randint(0, p, size=k) / (p - 1)
        x_scaled = np.array([lo + x_base[i] * (hi - lo) for i, (lo, hi) in enumerate(bounds)])
        order = rng.permutation(k)
        x_current = x_scaled.copy()
        y_current = func(x_current)
        for i in order:
            lo, hi = bounds[i]
            step = delta * (hi - lo)
            x_next = x_current.copy()
            x_next[i] += step
            if x_next[i] > hi:
                x_next[i] -= 2 * step
            y_next = func(x_next)
            ee = (y_next - y_current) / (step if step != 0 else 1e-8)
            effects[i].append(ee)
            x_current = x_next
            y_current = y_next

    return {i: {
        "mu_star": float(np.mean(np.abs(effects[i]))),
        "sigma": float(np.std(effects[i])),
        "mu": float(np.mean(effects[i])),
    } for i in range(k)}


def sobol_first_order(
    func,
    bounds: list[tuple[float, float]],
    n_samples: int = 1024,
    seed: int = 42,
) -> dict[int, float]:
    rng = np.random.RandomState(seed)
    k = len(bounds)

    def _sample(n):
        raw = rng.uniform(size=(n, k))
        return np.array([[lo + raw[j, i] * (hi - lo) for i, (lo, hi) in enumerate(bounds)]
                         for j in range(n)])

    A = _sample(n_samples)
    B = _sample(n_samples)
    y_A = np.array([func(A[j]) for j in range(n_samples)])
    y_B = np.array([func(B[j]) for j in range(n_samples)])
    var_y = np.var(np.concatenate([y_A, y_B]))

    if var_y < 1e-12:
        return {i: 0.0 for i in range(k)}

    indices = {}
    for i in range(k):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        y_AB_i = np.array([func(AB_i[j]) for j in range(n_samples)])
        s_i = float(np.mean(y_B * (y_AB_i - y_A)) / var_y)
        indices[i] = max(0.0, min(1.0, s_i))
    return indices


# ── SA runner ────────────────────────────────────────────────────────────────

DEFAULT_SA_RESPONSES: list[str] = [
    "I felt calm and handled the situation without difficulty.",        # ~low
    "I pulled back a little, unsure whether to continue.",             # ~withdraw
    "I responded to the situation as it came, nothing special.",        # ~neutral
    "I leaned in and got more involved than I expected.",              # ~engage
    "I could not stop myself, I kept pushing further and further.",     # ~escalate
]


def run_sensitivity_analysis(
    mode: str = "morris",
    n_ticks: int = 6,
    n_participants: int = 4,
    r: int = 10,
    n_samples: int = 256,
    response_templates: Optional[list[str]] = None,
) -> dict:
    """Run Morris or Sobol sensitivity analysis on the score-update model.

    Args:
        response_templates: Domain-specific representative responses spanning
            the behavioral spectrum (low→high). Defaults to generic templates.
            Override with scenario-appropriate text for domain-specific SA.
    """
    bounds = [
        (0.1, 0.4),    # alpha
        (1.0, 6.0),    # K (cast to int)
        (0.60, 0.85),  # intervention threshold
        (0.3, 1.0),    # dampening coefficient
        (0.2, 1.0),    # susceptibility
        (0.0, 0.6),    # resilience
        (3.0, 10.0),   # logistic_k (steepness)
    ]
    param_names = ["alpha", "K", "threshold", "dampening",
                   "susceptibility", "resilience", "logistic_k"]

    templates = response_templates or DEFAULT_SA_RESPONSES
    _sa_signals = None

    def _get_sa_signals():
        nonlocal _sa_signals
        if _sa_signals is None:
            _load_anchors()
            model = _get_embed()
            vecs = model.encode(templates, show_progress_bar=False)
            _sa_signals = [_compute_signal_from_vec(v)[0] for v in vecs]
        return _sa_signals

    def objective(x: np.ndarray) -> float:
        alpha, k_float, threshold, dampening, susceptibility, resilience, logistic_k = x
        rng = np.random.RandomState(42)
        sa_signals = _get_sa_signals()
        scores = []
        score = 0.3
        for t in range(n_ticks):
            base_idx = rng.randint(0, len(sa_signals))
            signal = sa_signals[base_idx] + rng.normal(0, 0.05)
            signal = max(0.0, min(1.0, signal))
            d = dampening if (t % max(1, int(k_float)) == 0) else 1.0
            score = update_score(score, signal, d, alpha,
                                 mode="logistic", logistic_k=logistic_k,
                                 susceptibility=susceptibility,
                                 resilience=resilience)
            scores.append(score)
        return float(np.mean(scores))

    if mode == "morris":
        results = morris_screening(objective, bounds, r=r)
        logger.info("Morris Screening Results:")
        for i, name in enumerate(param_names):
            r_i = results[i]
            logger.info("  %s  \u03bc*=%.4f  \u03c3=%.4f  \u03bc=%.4f",
                        name.ljust(15), r_i["mu_star"], r_i["sigma"], r_i["mu"])
        return {"mode": "morris", "param_names": param_names, "results": results}

    elif mode == "sobol":
        results = sobol_first_order(objective, bounds, n_samples=n_samples)
        logger.info("Sobol First-Order Indices:")
        for i, name in enumerate(param_names):
            logger.info("  %s  S_i=%.4f", name.ljust(15), results[i])
        return {"mode": "sobol", "param_names": param_names, "results": results}

    else:
        raise ValueError(f"Unknown SA mode: {mode}. Use 'morris' or 'sobol'.")


# ── Intervention threshold calibration ───────────────────────────────────────

def calibrate_intervention_threshold(
    run_dirs: list[str],
    labels: Optional[dict[str, bool]] = None,
    theta_range: tuple[float, float] = (0.60, 0.85),
    n_steps: int = 26,
) -> dict:
    """Sweep intervention threshold and find optimal F1.

    If labels is None, uses a heuristic: interventions triggered when any
    agent's score > 0.8 for 2+ consecutive ticks are "warranted".

    Args:
        run_dirs: paths to completed run directories containing observations.json
        labels: optional dict mapping run_dir -> bool (intervention warranted)
        theta_range: range of thresholds to sweep
        n_steps: number of threshold values to test

    Returns:
        dict with optimal_theta, f1_scores, and best_f1
    """
    # Collect observer_b outputs and ground truth labels
    samples = []
    for run_dir in run_dirs:
        rd = Path(run_dir)
        obs_path = rd / "observations.json"
        event_path = rd / "event_stream.json"
        if not obs_path.exists():
            logger.warning(f"Skipping {run_dir}: no observations.json")
            continue

        obs_data = json.loads(obs_path.read_text())

        # Extract observer_b responses from event stream
        observer_texts = []
        if event_path.exists():
            events = json.loads(event_path.read_text())
            observer_texts = [
                e.get("content", "") for e in events
                if e.get("event_type") == "observation"
                and "observer_b" in e.get("agent_id", "")
            ]

        # Determine ground truth: was intervention warranted?
        if labels is not None:
            warranted = labels.get(run_dir, False)
        else:
            # Heuristic: check if any agent had score > 0.8 for 2+ consecutive ticks
            warranted = False
            for agent_obs in obs_data if isinstance(obs_data, list) else obs_data.values():
                if isinstance(agent_obs, dict):
                    agent_obs = [agent_obs]
                scores = [o.get("score_after", 0) for o in agent_obs if isinstance(o, dict)]
                for i in range(1, len(scores)):
                    if scores[i] > 0.8 and scores[i - 1] > 0.8:
                        warranted = True
                        break
                if warranted:
                    break

        for text in observer_texts:
            samples.append({"text": text, "warranted": warranted})

    if not samples:
        logger.warning("No samples found for calibration. Run pilot simulations first.")
        return {"optimal_theta": INTERVENTION_THRESHOLD, "f1_scores": {}, "best_f1": 0.0}

    # Compute cosine similarities against intervention codebook
    model = _get_embed()
    codebook_phrases = []
    for phrases in INTERVENTION_CODEBOOK.values():
        codebook_phrases.extend(phrases)
    codebook_vecs = model.encode(codebook_phrases, show_progress_bar=False)

    sample_vecs = model.encode([s["text"] for s in samples], show_progress_bar=False)
    max_sims = []
    for sv in sample_vecs:
        norms = np.linalg.norm(codebook_vecs, axis=1) * (np.linalg.norm(sv) + 1e-8)
        sims = codebook_vecs @ sv / norms
        max_sims.append(float(np.max(sims)))

    # Sweep thresholds
    thetas = np.linspace(theta_range[0], theta_range[1], n_steps)
    f1_scores = {}
    best_f1, optimal_theta = 0.0, INTERVENTION_THRESHOLD

    for theta in thetas:
        tp = fp = fn = 0
        for i, sample in enumerate(samples):
            predicted = max_sims[i] >= theta
            actual = sample["warranted"]
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[float(theta)] = f1
        if f1 > best_f1:
            best_f1 = f1
            optimal_theta = float(theta)

    logger.info("Threshold Calibration: %d samples, best F1=%.3f at θ=%.3f (current=%.3f)",
                len(samples), best_f1, optimal_theta, INTERVENTION_THRESHOLD)
    if best_f1 > 0:
        logger.info("Recommendation: set INTERVENTION_THRESHOLD=%.2f", optimal_theta)

    return {"optimal_theta": optimal_theta, "f1_scores": f1_scores, "best_f1": best_f1}
