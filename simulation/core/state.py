"""Core data structures for the simulation loop."""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from simulation.signal.intervention import Intervention
from simulation.core.event_stream import EventStream
from simulation.knowledge.agent_memory import AgentMemoryStore
from simulation.core.safety import SafetyGate
from simulation.knowledge.graph_memory import GraphMemory

logger = logging.getLogger(__name__)

__all__ = [
    # Defined in this module
    "ObsEntry",
    "WorldState",
    "_format_stats",
    "MAX_RETRIES",
    "RETRY_DELAY",
    # Re-exported from signal.computation
    "set_seed", "_get_rng", "_get_embed", "_cosine", "_load_anchors",
    "_compute_signal_from_vec", "_compute_signal_with_action",
    "embed_score_batch", "update_score", "DEFAULT_EMBED_PATH",
    "ACTION_BASE_SIGNALS", "EMBED_MODIFIER_RANGE",
    # Re-exported from signal.intervention
    "Intervention", "extract_interventions", "_make_intervention",
    "_load_codebook",
    # Re-exported from signal.preflight
    "CONTEXT_FILE", "CONTEXT_REQUIREMENTS", "FLAME_ENABLED",
    "load_world_context", "detect_missing_context", "preflight_check",
    # Re-exported from signal.sensitivity
    "morris_screening", "sobol_first_order",
    "run_sensitivity_analysis", "calibrate_intervention_threshold",
    # Re-exported from storage.export
    "export_results", "OUTPUT_DIR",
]

# -- Configuration --
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# -- Re-exports (backward compatibility) --
# DEPRECATED: prefer importing directly from the canonical module.
from simulation.signal.computation import (  # noqa: F401
    set_seed, _get_rng, _get_embed, _cosine, _load_anchors,
    _compute_signal_from_vec, _compute_signal_with_action,
    embed_score_batch, update_score, DEFAULT_EMBED_PATH,
    ACTION_BASE_SIGNALS, EMBED_MODIFIER_RANGE,
)
from simulation.signal.intervention import (  # noqa: F401
    Intervention, extract_interventions, _make_intervention,
    _load_codebook,
)
from simulation.signal.preflight import (  # noqa: F401
    CONTEXT_FILE, CONTEXT_REQUIREMENTS, FLAME_ENABLED,
    load_world_context, detect_missing_context, preflight_check,
)
from simulation.signal.sensitivity import (  # noqa: F401
    morris_screening, sobol_first_order,
    run_sensitivity_analysis, calibrate_intervention_threshold,
)
from simulation.storage.export import (  # noqa: F401
    export_results, OUTPUT_DIR,
)

@dataclass
class ObsEntry:
    tick: int
    participant_id: str
    score_before: float
    score_after: float
    stimulus: str
    response: str
    signal: float = 0.0
    signal_se: float = 0.0

    def to_str(self) -> str:
        return (
            f"[T{self.tick}] {self.participant_id} "
            f"(score {self.score_before:.2f}\u2192{self.score_after:.2f}, "
            f"signal={self.signal:.2f}\u00b1{self.signal_se:.3f}) "
            f"stimulus: \"{self.stimulus[:70]}\" | response: \"{self.response[:70]}\""
        )


@dataclass
class WorldState:
    k: int = 3
    _log: list[ObsEntry] = field(default_factory=list)
    _log_by_tick: dict = field(default_factory=dict)  # tick → [ObsEntry]
    active_interventions: list[Intervention] = field(default_factory=list)
    _compliance_log: list[dict] = field(default_factory=list)
    stream: EventStream = field(default_factory=EventStream)
    memory: Optional[AgentMemoryStore] = field(default=None)
    safety_gate: SafetyGate = field(default_factory=SafetyGate)
    graph: Optional[GraphMemory] = field(default=None)

    pom_targets: dict = field(default_factory=lambda: {
        "score_distribution": None,
        "temporal_autocorrelation": None,
        "intervention_response": None,
        "convergence_pattern": None,
    })

    def log(self, entry: ObsEntry) -> None:
        self._log.append(entry)
        self._log_by_tick.setdefault(entry.tick, []).append(entry)

    def full_log(self) -> list[ObsEntry]:
        return list(self._log)

    def compliance_report(self) -> list[dict]:
        return list(self._compliance_log)

    def observer_prompt_window(self, tick: int, n_participants: int) -> str:
        # O(K) lookup instead of O(total_entries)
        window = []
        for t in range(max(1, tick - self.k + 1), tick + 1):
            window.extend(self._log_by_tick.get(t, []))
        if not window:
            return "No observations yet."
        if n_participants > 15:
            scores_before = [e.score_before for e in window]
            scores_after  = [e.score_after  for e in window]
            signals       = [e.signal for e in window]
            crossings     = sum(1 for e in window if e.score_before < 0.7 <= e.score_after)
            return (
                f"Ticks {tick - self.k + 1}\u2013{tick} | {n_participants} participants | "
                f"mean score {np.mean(scores_before):.2f}\u2192{np.mean(scores_after):.2f} "
                f"(\u03c3={np.std(scores_after):.3f}) | "
                f"mean signal {np.mean(signals):.2f} (\u03c3={np.std(signals):.3f}) | "
                f"participants crossing 0.7 threshold: {crossings}"
            )
        return "\n".join(e.to_str() for e in window)

    def apply_interventions(self) -> None:
        still_active = []
        for iv in self.active_interventions:
            if iv.duration == -1:
                still_active.append(iv)
            elif iv.duration > 0:
                iv.duration -= 1
                still_active.append(iv)
        self.active_interventions = still_active

    def environment_constraints(self) -> str:
        return " ".join(iv.description for iv in self.active_interventions
                        if iv.type == "environment_constraint")

    def participant_nudges(self) -> str:
        return " ".join(iv.description for iv in self.active_interventions
                        if iv.type == "participant_nudge")

    def score_dampening(self) -> float:
        d = 1.0
        for iv in self.active_interventions:
            if iv.type == "score_modifier":
                d *= iv.modifier.get("dampening", 1.0)
        return d

    def compute_score_statistics(self, tick: int) -> dict:
        entries = self._log_by_tick.get(tick, [])
        if not entries:
            return {}
        scores = [e.score_after for e in entries]
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores) if n > 1 else 0.0
        skew = float(np.mean(((np.array(scores) - mean) / (std + 1e-8)) ** 3)) if n > 2 else 0.0
        above = sum(1 for s in scores if s >= 0.7)

        prev_entries = self._log_by_tick.get(tick - 1, [])
        autocorr = None
        if prev_entries and len(prev_entries) == len(entries):
            prev_scores = [e.score_after for e in prev_entries]
            combined = np.array(scores + prev_scores)
            mu = np.mean(combined)
            num = sum((s - mu) * (p - mu) for s, p in zip(scores, prev_scores))
            den = sum((s - mu) ** 2 for s in combined)
            autocorr = float(num / (den + 1e-8)) if den > 0 else None

        return {
            "mean": float(mean), "std": float(std), "skewness": float(skew),
            "n_above_threshold": above, "n_total": n,
            "autocorrelation_lag1": autocorr,
        }


def _format_stats(stats: dict) -> str:
    """Format score statistics for observer prompt injection."""
    if not stats:
        return ""
    s = (
        f"\nPopulation statistics: mean={stats['mean']:.3f}, "
        f"\u03c3={stats['std']:.3f}, skew={stats['skewness']:.2f}, "
        f"above 0.7: {stats['n_above_threshold']}/{stats['n_total']}"
    )
    if stats.get("autocorrelation_lag1") is not None:
        s += f", lag-1 autocorr={stats['autocorrelation_lag1']:.3f}"
    return s

