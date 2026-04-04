"""Intervention extraction and management.

Extracted from sim_loop.py — owns the Intervention dataclass, codebook
matching via cosine similarity, and intervention instantiation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from simulation.agent_rolesv3 import INTERVENTION_CODEBOOK, INTERVENTION_THRESHOLD
from simulation.signal_computation import _get_embed

logger = logging.getLogger(__name__)


@dataclass
class Intervention:
    type: str
    description: str
    modifier: dict
    activated_at: int
    duration: int = -1
    source: str = ""


# ── Codebook matching ────────────────────────────────────────────────────────

_codebook_vecs: Optional[np.ndarray] = None
_codebook_keys: list[str] = []
_codebook_phrases: list[str] = []


def _load_codebook():
    global _codebook_vecs, _codebook_keys, _codebook_phrases
    if _codebook_vecs is not None:
        return
    model = _get_embed()
    for key, phrases in INTERVENTION_CODEBOOK.items():
        for phrase in phrases:
            _codebook_keys.append(key)
            _codebook_phrases.append(phrase)
    _codebook_vecs = model.encode(_codebook_phrases)


def extract_interventions(observer_id: str, response: str, tick: int, precomputed_vec=None,
                          threshold: float = INTERVENTION_THRESHOLD) -> list[Intervention]:
    _load_codebook()
    vec = precomputed_vec if precomputed_vec is not None else _get_embed().encode([response])[0]
    triggered = {}
    # Vectorized cosine similarity against codebook
    norms = np.linalg.norm(_codebook_vecs, axis=1) * (np.linalg.norm(vec) + 1e-8)
    sims = _codebook_vecs @ vec / norms
    for idx, (key, phrase) in enumerate(zip(_codebook_keys, _codebook_phrases)):
        s = float(sims[idx])
        if s >= threshold:
            if key not in triggered or s > triggered[key][0]:
                triggered[key] = (s, phrase)

    ivs = []
    for key, (sim, phrase) in triggered.items():
        iv_type, description, modifier = _make_intervention(key)
        ivs.append(Intervention(
            type=iv_type, description=description, modifier=modifier,
            activated_at=tick, source=observer_id,
        ))
        logger.info(f"  [{observer_id}] INTERVENTION: {key} (sim={sim:.2f})")
    return ivs


def _make_intervention(key: str) -> tuple[str, str, dict]:
    defaults = {
        "pause_prompt":        ("participant_nudge",
                                "A pause prompt appeared: 'Take a moment to reflect.'", {}),
        "boundary_warning":    ("environment_constraint",
                                "Flag content or interactions approaching scenario boundaries.", {}),
        "pacing_adjustment":   ("participant_nudge",
                                "Consider adjusting the pace of the current interaction.", {}),
        "dynamics_dampening":  ("score_modifier",
                                "Behavioral dynamics dampened by observer recommendation.",
                                {"dampening": 0.6}),
    }
    return defaults.get(key, ("environment_constraint", key, {}))


