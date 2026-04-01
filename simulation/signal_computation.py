"""Signal computation, embedding, and score update logic.

Extracted from sim_loop.py — owns the embedding model singleton, anchor-based
signal computation, action-modulated signals, batch scoring, and the
two-mode (EMA/logistic) score update function.
"""

from __future__ import annotations

import contextvars
import logging
import math
import os
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from simulation.agent_rolesv3 import ENGAGEMENT_ANCHORS, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_EMBED_PATH = os.environ.get(
    "SIM_EMBED_MODEL",
    os.path.join(os.environ.get("HF_HOME", "/workspace/huggingface"), "hub", "e5-small-v2"),
)

# ── Reproducibility ──────────────────────────────────────────────────────────

_rng: Optional[np.random.RandomState] = None
_rng_ctx: contextvars.ContextVar[Optional[np.random.RandomState]] = contextvars.ContextVar(
    "_rng_ctx", default=None
)


def set_seed(seed: int = 42) -> None:
    """Set the RNG for the current context (or global fallback)."""
    rng = np.random.RandomState(seed)
    _rng_ctx.set(rng)
    global _rng
    _rng = rng
    logger.info(f"Seed set to {seed}")


def _get_rng() -> np.random.RandomState:
    """Get the RNG for the current context, falling back to global."""
    ctx_rng = _rng_ctx.get(None)
    if ctx_rng is not None:
        return ctx_rng
    global _rng
    if _rng is None:
        set_seed(42)
    return _rng


# ── Embedding helper ─────────────────────────────────────────────────────────

_embed_model: Optional[SentenceTransformer] = None


def _get_embed() -> SentenceTransformer:
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        _embed_model = SentenceTransformer(DEFAULT_EMBED_PATH)
        logger.info(f"Loaded embedding model from: {DEFAULT_EMBED_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load from {DEFAULT_EMBED_PATH}: {e}. Falling back to HuggingFace.")
        _embed_model = SentenceTransformer("intfloat/e5-small-v2")
        logger.info("Loaded embedding model from HuggingFace: intfloat/e5-small-v2")
    return _embed_model


def _cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Anchor-based signal computation ─────────────────────────────────────────

_anchor_vecs: Optional[np.ndarray] = None
_anchor_labels: list[str] = []
_anchor_high_mask: Optional[np.ndarray] = None


def _load_anchors():
    global _anchor_vecs, _anchor_labels
    if _anchor_vecs is not None:
        return
    model = _get_embed()
    high = ENGAGEMENT_ANCHORS["high"]
    low  = ENGAGEMENT_ANCHORS["low"]
    _anchor_labels = ["high"] * len(high) + ["low"] * len(low)
    _anchor_vecs = model.encode(high + low)


def _compute_signal_from_vec(vec) -> tuple[float, float]:
    """Vectorized cosine similarity against all anchors.

    SE is propagated from both poles separately via delta method:
      signal = (high_mean - low_mean + 1) / 2
      se(signal) = sqrt(se_high^2 + se_low^2) / 2
    This captures uncertainty in *both* pole estimates rather than
    pooling all 20 similarities into a single noisy scalar.
    """
    global _anchor_high_mask
    _load_anchors()
    if _anchor_high_mask is None:
        _anchor_high_mask = np.array([l == "high" for l in _anchor_labels])
    # Vectorized: cosine sims for all anchors at once
    norms = np.linalg.norm(_anchor_vecs, axis=1) * (np.linalg.norm(vec) + 1e-8)
    sims = _anchor_vecs @ vec / norms
    high_sims = sims[_anchor_high_mask]
    low_sims = sims[~_anchor_high_mask]
    high_sim = float(np.mean(high_sims))
    low_sim  = float(np.mean(low_sims))
    signal = float(np.clip((high_sim - low_sim + 1.0) / 2.0, 0.0, 1.0))
    # Per-pole SE, propagated through the signal transform
    se_high = float(np.std(high_sims) / math.sqrt(len(high_sims)))
    se_low = float(np.std(low_sims) / math.sqrt(len(low_sims)))
    se = float(math.sqrt(se_high**2 + se_low**2) / 2.0)
    return signal, se


# ── Action-based signal override ─────────────────────────────────────────────
# Base signal for each structured action type.
# 5-level system: wider spread + larger modifier range for meaningful variation.
ACTION_BASE_SIGNALS: dict[str, float] = {
    "disengage": 0.10,
    "withdraw": 0.25,
    "respond": 0.50,
    "engage": 0.70,
    "escalate": 0.85,
}
EMBED_MODIFIER_RANGE = 0.20


def _compute_signal_with_action(
    parsed_action: Optional[dict],
    embed_vec,
) -> tuple[float, float]:
    """
    Compute behavioral signal using structured action as base, with
    embedding similarity as a fine-grained modifier.

    Falls back to pure embedding if no recognized action.
    """
    embed_signal, embed_se = _compute_signal_from_vec(embed_vec)

    if parsed_action is None:
        return embed_signal, embed_se

    action_type = str(parsed_action.get("action", "")).lower().strip()

    if action_type in ACTION_BASE_SIGNALS:
        base = ACTION_BASE_SIGNALS[action_type]
        modifier = (embed_signal - 0.5) * 2.0 * EMBED_MODIFIER_RANGE
        signal = float(np.clip(base + modifier, 0.0, 1.0))
        return signal, embed_se

    intensity = parsed_action.get("intensity")
    if intensity is not None:
        try:
            intensity_f = float(intensity)
            if 0.0 <= intensity_f <= 1.0:
                signal = float(np.clip(0.7 * intensity_f + 0.3 * embed_signal, 0.0, 1.0))
                return signal, embed_se
        except (TypeError, ValueError):
            pass

    return embed_signal, embed_se


def embed_score_batch(
    responses: list[str],
    parsed_actions: Optional[list[Optional[dict]]] = None,
) -> list[tuple[float, float]]:
    """[Fix 12] Batch encode all responses in one model call.
    If parsed_actions provided, uses action type as base signal with embed modifier."""
    _load_anchors()
    model = _get_embed()
    vecs = model.encode(responses, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)

    if parsed_actions is None:
        return [_compute_signal_from_vec(v) for v in vecs]

    return [
        _compute_signal_with_action(
            parsed_actions[i] if i < len(parsed_actions) else None, v
        )
        for i, v in enumerate(vecs)
    ]


def update_score(
    current: float,
    signal: float,
    dampening: float = 1.0,
    alpha: float = 0.2,
    mode: str = "ema",
    logistic_k: float = 6.0,
    susceptibility: float = 1.0,
    resilience: float = 0.0,
) -> float:
    """
    Score update with two modes and heterogeneous agent modifiers.

    Modes:
      ema      -- linear EMA: Score += d * alpha * (signal - score). Analytically
                 tractable, suitable for SA sweeps.
      logistic -- sigmoid-transformed signal: captures saturation at extremes.
                 Clinically motivated: deeply engaged users resist both
                 intervention (hard to push below 0.8) and further escalation
                 (ceiling effect). k controls steepness.

    Agent modifiers (heterogeneous agents):
      susceptibility -- scales raw signal before update (higher = more affected
                       by stimuli). Default 1.0 = no modification.
      resilience     -- additive dampening (higher = more resistant to score
                       change). Applied multiplicatively with intervention
                       dampening: effective_d = dampening * (1 - resilience).
    """
    # Apply susceptibility: modulates how strongly signal affects this agent
    effective_signal = current + susceptibility * (signal - current)

    if mode == "logistic":
        midpoint = 0.5
        effective_signal = 1.0 / (1.0 + np.exp(-logistic_k * (effective_signal - midpoint)))

    # Apply resilience: baseline resistance to score change
    effective_dampening = dampening * (1.0 - resilience)

    delta = alpha * (effective_signal - current)
    return float(np.clip(current + delta * effective_dampening, 0.0, 1.0))
