"""
Generalized multi-agent simulation loop — stratified tick architecture (v3).

Merges sim_loopv2.1 base with all v3 patches:
  [Fix 2]  Phase D: sequential A-analyse → B-intervene
  [Fix 5]  Intervention threshold from agent_rolesv3
  [Fix 6]  Score NOT leaked to participant prompt
  [Fix 7]  Persona summary injection every PERSONA_SUMMARY_INTERVAL ticks
  [Fix 9]  Compliance checking after each agent turn
  [Fix 12] Batch embedding in Phase C

Architecture layers (Park et al. 2023; Adaptive-VP, ACL 2025):
  Layer 1 — Scenario Engine:     EnvironmentAgent + run_tick() orchestration
  Layer 2 — Persona Module:      ParticipantAgent (six-component prompt encoding)
  Layer 3 — Feedback/Assessment: ObserverAgent (evaluator-generator separation)
  Layer 4 — State Management:    WorldState (log, interventions, windowed summaries)
  Layer 5 — Memory/Context:      history_window (recency) + embed_score (relevance)
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import os
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from simulation.agent_rolesv3 import (
    AGENT_ROLES,
    INTERVENTION_CODEBOOK,
    ENGAGEMENT_ANCHORS,
    INTERVENTION_THRESHOLD,
    PERSONA_SUMMARY_INTERVAL,
    PERSONA_SUMMARY_TEMPLATE,
    EMBED_BATCH_SIZE,
    check_compliance,
)
from orchestrator import agent_turn, close_client

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_EMBED_PATH = os.environ.get(
    "SIM_EMBED_MODEL",
    os.path.join(os.environ.get("HF_HOME", "/per.volume/huggingface"), "hub", "e5-small-v2"),
)
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


# ── Reproducibility ───────────────────────────────────────────────────────────

_rng: Optional[np.random.RandomState] = None


def set_seed(seed: int = 42) -> None:
    global _rng
    _rng = np.random.RandomState(seed)
    logger.info(f"Global seed set to {seed}")


def _get_rng() -> np.random.RandomState:
    global _rng
    if _rng is None:
        set_seed(42)
    return _rng


# ── Embedding helper ──────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

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


# ── Resilient agent_turn wrapper ──────────────────────────────────────────────

async def _resilient_agent_turn(
    agent_id: str,
    backend: str,
    system_prompt: str,
    user_message: str,
    history: list[dict],
    max_tokens: int,
) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await agent_turn(
                agent_id=agent_id,
                backend=backend,
                system_prompt=system_prompt,
                user_message=user_message,
                history=history,
                max_tokens=max_tokens,
            )
        except Exception as e:
            delay = RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(
                f"[{agent_id}] agent_turn failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(delay)
            else:
                fallback = f"[{agent_id} fallback: agent_turn failed after {MAX_RETRIES} attempts]"
                logger.error(f"[{agent_id}] All retries exhausted. Using fallback response.")
                return fallback


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Intervention:
    type: str
    description: str
    modifier: dict
    activated_at: int
    duration: int = -1
    source: str = ""


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
    active_interventions: list[Intervention] = field(default_factory=list)
    _compliance_log: list[dict] = field(default_factory=list)

    pom_targets: dict = field(default_factory=lambda: {
        "score_distribution": None,
        "temporal_autocorrelation": None,
        "intervention_response": None,
        "convergence_pattern": None,
    })

    def log(self, entry: ObsEntry) -> None:
        self._log.append(entry)

    def full_log(self) -> list[ObsEntry]:
        return list(self._log)

    def compliance_report(self) -> list[dict]:
        return list(self._compliance_log)

    def observer_prompt_window(self, tick: int, n_participants: int) -> str:
        window = [e for e in self._log if e.tick > tick - self.k]
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
        entries = [e for e in self._log if e.tick == tick]
        if not entries:
            return {}
        scores = [e.score_after for e in entries]
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores) if n > 1 else 0.0
        skew = float(np.mean(((np.array(scores) - mean) / (std + 1e-8)) ** 3)) if n > 2 else 0.0
        above = sum(1 for s in scores if s >= 0.7)

        prev_entries = [e for e in self._log if e.tick == tick - 1]
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


# ── Score update ──────────────────────────────────────────────────────────────

_anchor_vecs: Optional[np.ndarray] = None
_anchor_labels: list[str] = []


def _load_anchors():
    global _anchor_vecs, _anchor_labels
    if _anchor_vecs is not None:
        return
    model = _get_embed()
    high = ENGAGEMENT_ANCHORS["high"]
    low  = ENGAGEMENT_ANCHORS["low"]
    _anchor_labels = ["high"] * len(high) + ["low"] * len(low)
    _anchor_vecs = model.encode(high + low)


@functools.lru_cache(maxsize=512)
def _embed_cached(text: str):
    return _get_embed().encode([text])[0]


def _compute_signal_from_vec(vec) -> tuple[float, float]:
    _load_anchors()
    sims = np.array([_cosine(vec, a) for a in _anchor_vecs])
    high_mask = np.array([l == "high" for l in _anchor_labels])
    high_sim = float(np.mean(sims[high_mask]))
    low_sim  = float(np.mean(sims[~high_mask]))
    signal = float(np.clip((high_sim - low_sim + 1.0) / 2.0, 0.0, 1.0))
    se = float(np.std(sims) / math.sqrt(len(sims)))
    return signal, se


def embed_score(response: str) -> tuple[float, float]:
    _load_anchors()
    vec = _get_embed().encode([response])[0]
    return _compute_signal_from_vec(vec)


def embed_score_batch(responses: list[str]) -> list[tuple[float, float]]:
    """[Fix 12] Batch encode all responses in one model call."""
    _load_anchors()
    model = _get_embed()
    vecs = model.encode(responses, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)
    return [_compute_signal_from_vec(v) for v in vecs]


def update_score(
    current: float,
    signal: float,
    dampening: float = 1.0,
    alpha: float = 0.2,
) -> float:
    delta = alpha * (signal - current)
    return float(np.clip(current + delta * dampening, 0.0, 1.0))


# ── Sensitivity analysis ─────────────────────────────────────────────────────

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


# ── Intervention extraction ───────────────────────────────────────────────────

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


def extract_interventions(observer_id: str, response: str, tick: int) -> list[Intervention]:
    _load_codebook()
    vec = _get_embed().encode([response])[0]
    triggered = {}
    for key, anchor_vec, phrase in zip(_codebook_keys, _codebook_vecs, _codebook_phrases):
        sim = _cosine(vec, anchor_vec)
        if sim >= INTERVENTION_THRESHOLD:
            if key not in triggered or sim > triggered[key][0]:
                triggered[key] = (sim, phrase)

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


def _format_stats(stats: dict) -> str:
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


# ── Agent classes ─────────────────────────────────────────────────────────────

class EnvironmentAgent:
    """
    Generates stimuli per participant (Phase A). Singleton.
    DDA: score >= 0.8 -> stabilising, >= 0.5 -> variation, < 0.5 -> baseline.
    """

    def __init__(self, history_window: int = 4):
        self.cfg = AGENT_ROLES["environment"]
        self.history: list[dict] = []
        self.history_window = history_window

    async def decide(self, participant: "ParticipantAgent", world_state: WorldState, max_tokens: int = 128) -> str:
        constraints = world_state.environment_constraints()
        constraint_note = f" ACTIVE CONSTRAINTS: {constraints}" if constraints else ""

        if participant.behavioral_score >= 0.8:
            dda_note = " [HIGH SCORE \u2014 test boundary maintenance, present stabilising stimuli]"
        elif participant.behavioral_score >= 0.5:
            dda_note = " [MODERATE SCORE \u2014 introduce gradual variation]"
        else:
            dda_note = ""

        prompt = (
            f"Participant {participant.agent_id}: behavioral_score={participant.behavioral_score:.2f}."
            f"{constraint_note}{dda_note} "
            f"Last step you presented: \"{participant.last_stimulus[:80]}\". "
            f"They responded: \"{participant.last_response[:80]}\". "
            f"Decide what stimulus to present next. Be specific."
        )
        response = await _resilient_agent_turn(
            agent_id="environment",
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=max_tokens,
        )
        self.history.append({"role": "assistant", "content": response})
        return response

    async def batch_decide(
        self, participants: list["ParticipantAgent"], world_state: WorldState,
        max_tokens: int = 256,
    ) -> dict[str, str]:
        constraints = world_state.environment_constraints()
        constraint_note = f"\nACTIVE CONSTRAINTS: {constraints}" if constraints else ""

        participant_summaries = []
        for p in participants:
            if p.behavioral_score >= 0.8:
                dda = "HIGH"
            elif p.behavioral_score >= 0.5:
                dda = "MODERATE"
            else:
                dda = "BASELINE"
            participant_summaries.append(
                f"  {p.agent_id}: score={p.behavioral_score:.2f} ({dda}) "
                f"last_response=\"{p.last_response[:60]}\""
            )

        prompt = (
            f"Generate a stimulus for each of the following {len(participants)} participants. "
            f"Respond with a JSON object mapping participant_id \u2192 stimulus string.{constraint_note}\n\n"
            + "\n".join(participant_summaries)
        )

        response = await _resilient_agent_turn(
            agent_id="environment",
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=max_tokens * len(participants),
        )
        self.history.append({"role": "assistant", "content": response})

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            stimuli = json.loads(cleaned)
            missing = [p.agent_id for p in participants if p.agent_id not in stimuli]
            if missing:
                raise ValueError(f"Missing participants in batch response: {missing}")
            return stimuli
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"batch_decide parse failed ({e}), falling back to sequential")
            stimuli = {}
            for p in participants:
                stimuli[p.agent_id] = await self.decide(p, world_state, max_tokens=max(64, max_tokens // 2))
            return stimuli


class ParticipantAgent:
    """
    Reacts to environment stimuli (Phase B). N instances, concurrent.
    [Fix 6] Score NOT leaked to participant.
    [Fix 7] Persona summary injected every PERSONA_SUMMARY_INTERVAL ticks.
    """

    def __init__(self, agent_id: str, history_window: int = 4):
        self.agent_id = agent_id
        self.cfg = AGENT_ROLES["participant"]
        self.history: list[dict] = []
        self.history_window = history_window
        self.behavioral_score: float = float(_get_rng().uniform(0.1, 0.5))
        self.score_log: list[float] = []
        self.last_stimulus: str = "nothing yet"
        self.last_response: str = ""
        self.persona_summary: str = ""

    async def _maybe_update_persona_summary(self, tick: int) -> None:
        """[Fix 7] Generate persona summary every PERSONA_SUMMARY_INTERVAL ticks."""
        if tick % PERSONA_SUMMARY_INTERVAL != 0 or tick == 0:
            return
        if len(self.history) < 4:
            return
        summary_prompt = (
            "In 3 concise sentences, summarise this participant's character: "
            "their personality, key emotional states shown, and any consistent "
            "patterns of behaviour (e.g. resistance, eagerness, withdrawal). "
            "This summary will be used to maintain consistency in future turns."
        )
        self.persona_summary = await _resilient_agent_turn(
            agent_id=f"{self.agent_id}_summariser",
            backend=self.cfg["backend"],
            system_prompt="You create concise character summaries from conversation history.",
            user_message=summary_prompt,
            history=self.history[-20:],
            max_tokens=120,
        )

    def _build_system_prompt(self) -> str:
        """[Fix 7] Prepend persona summary if available."""
        base = self.cfg["system"]
        if not self.persona_summary:
            return base
        summary_block = PERSONA_SUMMARY_TEMPLATE.format(
            interval=PERSONA_SUMMARY_INTERVAL,
            summary=self.persona_summary,
        )
        return f"{summary_block}\n\n{base}"

    async def step(self, tick: int, stimulus: str, world_state: WorldState, max_tokens: int = 256) -> str:
        await self._maybe_update_persona_summary(tick)

        nudge = world_state.participant_nudges()
        nudge_note = f" {nudge}" if nudge else ""
        # [Fix 6] Score NOT included in participant prompt
        prompt = (
            f"[Tick {tick}]{nudge_note} "
            f"The environment presents: \"{stimulus[:120]}\". "
            f"How do you respond? What do you do?"
        )

        response = await _resilient_agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=self._build_system_prompt(),
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=max_tokens,
        )

        # [Fix 9] Compliance check
        violations = check_compliance(response, "participant")
        if violations:
            print(f"  [COMPLIANCE] {self.agent_id} tick={tick} violations: {violations}")
            world_state._compliance_log.append({
                "tick": tick, "agent": self.agent_id,
                "role": "participant", "violations": violations,
            })

        self.last_stimulus = stimulus
        self.last_response = response
        self.history.append({"role": "assistant", "content": response})
        return response


class ObserverAgent:
    """
    Analyst agent — fires every K ticks (Phase D).
    [Fix 2] Split into analyse() (A) and intervene() (B).
    """

    def __init__(self, agent_id: str, role: str, history_window: int = 4, max_tokens: int = 256):
        self.agent_id = agent_id
        self.cfg = AGENT_ROLES[role]
        self.history: list[dict] = []
        self.history_window = history_window
        self.max_tokens = max_tokens
        self.analyses: list[str] = []

    async def analyse(self, tick: int, world_state: WorldState, n_participants: int) -> str:
        """[Fix 2] Observer A: analysis only. No codebook extraction."""
        window = world_state.observer_prompt_window(tick, n_participants)
        active = ", ".join(iv.description for iv in world_state.active_interventions) or "none"
        stats = world_state.compute_score_statistics(tick)
        stats_note = _format_stats(stats)

        prompt = (
            f"[Tick {tick}] Observation window (last {world_state.k} ticks):\n"
            f"{window}\n"
            f"{stats_note}\n\n"
            f"Active interventions: {active}\n\n"
            f"Analyse participant behaviour and population dynamics. "
            f"Describe what you see \u2014 do NOT recommend any interventions."
        )
        response = await _resilient_agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=self.max_tokens,
        )
        print(f"[{self.agent_id} ANALYSIS] {response[:120]}...")

        # [Fix 9] observer_a must NOT contain intervention keywords
        violations = check_compliance(response, "observer_a")
        if violations:
            print(f"  [COMPLIANCE] {self.agent_id} used intervention keywords: {violations}")
            world_state._compliance_log.append({
                "tick": tick, "agent": self.agent_id,
                "role": "observer_a", "violations": violations,
            })

        self.history.append({"role": "assistant", "content": response})
        self.analyses.append(response)
        return response

    async def intervene(
        self,
        tick: int,
        world_state: WorldState,
        n_participants: int,
        analysis: str,
    ) -> list[Intervention]:
        """[Fix 2] Observer B: proposes interventions based on observer_a's analysis."""
        window = world_state.observer_prompt_window(tick, n_participants)
        active = ", ".join(iv.description for iv in world_state.active_interventions) or "none"
        stats = world_state.compute_score_statistics(tick)
        stats_note = _format_stats(stats)

        prompt = (
            f"[Tick {tick}] Analyst report from observer_a:\n"
            f"{analysis}\n\n"
            f"Raw observation window (last {world_state.k} ticks):\n"
            f"{window}\n"
            f"{stats_note}\n\n"
            f"Active interventions: {active}\n\n"
            f"Based on the analyst's findings, decide whether to intervene. "
            f"If yes, state which type using the exact phrases from your instructions."
        )
        response = await _resilient_agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=self.max_tokens,
        )
        print(f"[{self.agent_id} INTERVENTION] {response[:120]}...")
        self.history.append({"role": "assistant", "content": response})
        self.analyses.append(response)
        return extract_interventions(self.agent_id, response, tick)

    async def observe(self, tick: int, world_state: WorldState, n_participants: int) -> list[Intervention]:
        """Legacy v2 method — kept for backward compatibility."""
        window = world_state.observer_prompt_window(tick, n_participants)
        active = ", ".join(iv.description for iv in world_state.active_interventions) or "none"
        stats = world_state.compute_score_statistics(tick)
        stats_note = _format_stats(stats)

        prompt = (
            f"[Tick {tick}] Observation window (last {world_state.k} ticks):\n"
            f"{window}\n{stats_note}\n\n"
            f"Active interventions: {active}\n\n"
            f"Analyse the dynamics at both individual and population level. "
            f"Recommend any new interventions if warranted."
        )
        response = await _resilient_agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=self.max_tokens,
        )
        logger.info(f"[{self.agent_id}] {response[:120]}...")
        self.history.append({"role": "assistant", "content": response})
        self.analyses.append(response)
        return extract_interventions(self.agent_id, response, tick)


# ── Tick orchestration ────────────────────────────────────────────────────────

async def run_tick(
    tick: int,
    environment: EnvironmentAgent,
    participants: list[ParticipantAgent],
    observers: list[ObserverAgent],
    world_state: WorldState,
    alpha: float = 0.2,
    max_tokens: int = 256,
) -> None:
    """
    Execute one tick: A (stimulus) -> B (response) -> C (scoring) -> D (observation).
    [Fix 2]  Phase D: sequential A->B
    [Fix 9]  Environment compliance check
    [Fix 12] Phase C: batch embedding
    """
    n = len(participants)

    # -- Phase A -- sequential stimulus generation
    if n <= 10:
        stimuli = {}
        for p in participants:
            stimuli[p.agent_id] = await environment.decide(
                p, world_state, max_tokens=max(64, max_tokens // 2)
            )
            # [Fix 9] Compliance check for environment
            violations = check_compliance(stimuli[p.agent_id], "environment")
            if violations:
                print(f"  [COMPLIANCE] environment tick={tick} violations: {violations}")
                world_state._compliance_log.append({
                    "tick": tick, "agent": "environment",
                    "role": "environment", "violations": violations,
                })
    else:
        stimuli = await environment.batch_decide(
            participants, world_state, max_tokens=max(64, max_tokens // 2)
        )

    # -- Phase B -- concurrent participant responses
    responses = await asyncio.gather(*[
        p.step(tick, stimuli[p.agent_id], world_state, max_tokens=max_tokens)
        for p in participants
    ])

    # -- Phase C -- batch score update [Fix 12]
    dampening = world_state.score_dampening()
    signals_and_ses = embed_score_batch(list(responses))

    for participant, response, (signal, signal_se) in zip(participants, responses, signals_and_ses):
        score_before = participant.behavioral_score
        participant.behavioral_score = update_score(score_before, signal, dampening, alpha)
        participant.score_log.append(participant.behavioral_score)
        world_state.log(ObsEntry(
            tick=tick,
            participant_id=participant.agent_id,
            score_before=score_before,
            score_after=participant.behavioral_score,
            stimulus=stimuli[participant.agent_id],
            response=response,
            signal=signal,
            signal_se=signal_se,
        ))

    # -- Phase D -- SEQUENTIAL observer cycle [Fix 2]
    if tick % world_state.k == 0:
        print(f"\n[Tick {tick}] Observer cycle (sequential A\u2192B)...")
        obs_a, obs_b = observers[0], observers[1]
        analysis = await obs_a.analyse(tick, world_state, n)
        ivs = await obs_b.intervene(tick, world_state, n, analysis)
        world_state.active_interventions.extend(ivs)

    world_state.apply_interventions()


async def run_simulation(
    n_ticks: int = 12,
    n_participants: int = 4,
    k: int = 4,
    alpha: float = 0.15,
    history_window: int = 4,
    max_tokens: int = 192,
    seed: int = 42,
    intervention_threshold: float = INTERVENTION_THRESHOLD,
    persona_summary_interval: int = PERSONA_SUMMARY_INTERVAL,
) -> tuple[list[ParticipantAgent], WorldState]:
    """
    Run the stratified multi-agent simulation.
    [Fix 5] Exposes intervention_threshold as a run parameter for SA sweeps.
    [Fix 7] Exposes persona_summary_interval.
    """
    # Temporarily override module-level threshold for this run
    import simulation.agent_rolesv3 as _cfg
    _original_threshold = _cfg.INTERVENTION_THRESHOLD
    _original_interval = _cfg.PERSONA_SUMMARY_INTERVAL
    _cfg.INTERVENTION_THRESHOLD = intervention_threshold
    _cfg.PERSONA_SUMMARY_INTERVAL = persona_summary_interval

    t_start = time.monotonic()

    set_seed(seed)
    world_state = WorldState(k=k)
    environment = EnvironmentAgent(history_window=history_window)
    participants = [
        ParticipantAgent(f"participant_{i}", history_window=history_window)
        for i in range(n_participants)
    ]
    observers = [
        ObserverAgent("observer_a", "observer_a",
                      history_window=history_window, max_tokens=max_tokens),
        ObserverAgent("observer_b", "observer_b",
                      history_window=history_window, max_tokens=max_tokens),
    ]

    try:
        for tick in range(1, n_ticks + 1):
            pct = tick / n_ticks * 100
            print(f"\n\u2500\u2500 Tick {tick}/{n_ticks}  ({pct:.0f}%) \u2500\u2500")
            await run_tick(tick, environment, participants, observers, world_state, alpha, max_tokens)
            for p in participants:
                print(f"  {p.agent_id}: score={p.behavioral_score:.3f}")
            stats = world_state.compute_score_statistics(tick)
            if stats:
                print(
                    f"  [POM] mean={stats['mean']:.3f} \u03c3={stats['std']:.3f} "
                    f"skew={stats['skewness']:.2f} above_0.7={stats['n_above_threshold']}"
                )
    finally:
        _cfg.INTERVENTION_THRESHOLD = _original_threshold
        _cfg.PERSONA_SUMMARY_INTERVAL = _original_interval

    duration_s = time.monotonic() - t_start

    # [Fix 9] Compliance report
    compliance = world_state.compliance_report()
    if compliance:
        print(f"\n\u2500\u2500 Compliance violations: {len(compliance)} \u2500\u2500")
        for v in compliance:
            print(f"  [T{v['tick']}] {v['agent']} ({v['role']}): {v['violations']}")

    # Export results
    run_config = {
        "n_ticks": n_ticks, "n_participants": n_participants,
        "k": k, "alpha": alpha, "history_window": history_window,
        "max_tokens": max_tokens, "seed": seed,
        "intervention_threshold": intervention_threshold,
        "persona_summary_interval": persona_summary_interval,
    }
    run_dir = export_results(participants, world_state, run_config, duration_s)
    print(f"\n\u2500\u2500 Results exported to {run_dir} \u2500\u2500")

    return participants, world_state


# ── Data export ───────────────────────────────────────────────────────────────

OUTPUT_DIR = os.environ.get("SIM_OUTPUT_DIR", "data")


def export_results(
    participants: list[ParticipantAgent],
    world_state: WorldState,
    config: dict,
    duration_s: float,
    output_dir: str | None = None,
) -> str:
    """
    Export simulation results to JSON files in data/{run_id}/.
    Returns the output directory path.
    """
    output_dir = output_dir or OUTPUT_DIR
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{ts}_s{config.get('seed', 0)}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Config + metadata
    meta = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(duration_s, 2),
        "config": config,
    }
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))

    # Score trajectories
    trajectories = {}
    for p in participants:
        trajectories[p.agent_id] = {
            "initial_score": p.score_log[0] if p.score_log else p.behavioral_score,
            "final_score": p.behavioral_score,
            "score_log": [round(s, 4) for s in p.score_log],
        }
    (run_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2))

    # Full observation log
    obs_log = [
        {
            "tick": e.tick, "participant_id": e.participant_id,
            "score_before": round(e.score_before, 4),
            "score_after": round(e.score_after, 4),
            "signal": round(e.signal, 4),
            "signal_se": round(e.signal_se, 4),
            "stimulus": e.stimulus, "response": e.response,
        }
        for e in world_state.full_log()
    ]
    (run_dir / "observations.json").write_text(json.dumps(obs_log, indent=2))

    # Compliance report
    compliance = world_state.compliance_report()
    if compliance:
        (run_dir / "compliance.json").write_text(json.dumps(compliance, indent=2))

    # Interventions that were active during the run
    interventions = [
        {
            "type": iv.type, "description": iv.description,
            "activated_at": iv.activated_at, "source": iv.source,
        }
        for iv in world_state.active_interventions
    ]
    (run_dir / "interventions.json").write_text(json.dumps(interventions, indent=2))

    logger.info(f"Results exported to {run_dir}")
    return str(run_dir)


# ── Sensitivity analysis runner ───────────────────────────────────────────────

def run_sensitivity_analysis(
    mode: str = "morris",
    n_ticks: int = 6,
    n_participants: int = 4,
    r: int = 10,
    n_samples: int = 256,
) -> dict:
    bounds = [
        (0.1, 0.4),    # alpha
        (1.0, 6.0),    # K (cast to int)
        (0.60, 0.85),  # intervention threshold
        (0.3, 1.0),    # dampening coefficient
    ]
    param_names = ["alpha", "K", "threshold", "dampening"]

    def objective(x: np.ndarray) -> float:
        alpha, k_float, threshold, dampening = x
        rng = np.random.RandomState(42)
        scores = []
        score = 0.3
        for t in range(n_ticks):
            signal = 0.5 + 0.3 * math.sin(t * 0.5) + rng.normal(0, 0.1)
            signal = max(0.0, min(1.0, signal))
            d = dampening if (t % max(1, int(k_float)) == 0) else 1.0
            score = score + d * alpha * (signal - score)
            score = max(0.0, min(1.0, score))
            scores.append(score)
        return float(np.mean(scores))

    if mode == "morris":
        results = morris_screening(objective, bounds, r=r)
        print("\n\u2500\u2500 Morris Screening Results \u2500\u2500")
        for i, name in enumerate(param_names):
            r_i = results[i]
            print(f"  {name:<15} \u03bc*={r_i['mu_star']:.4f}  \u03c3={r_i['sigma']:.4f}  \u03bc={r_i['mu']:.4f}")
        return {"mode": "morris", "param_names": param_names, "results": results}

    elif mode == "sobol":
        results = sobol_first_order(objective, bounds, n_samples=n_samples)
        print("\n\u2500\u2500 Sobol First-Order Indices \u2500\u2500")
        for i, name in enumerate(param_names):
            print(f"  {name:<15} S_i={results[i]:.4f}")
        return {"mode": "sobol", "param_names": param_names, "results": results}

    else:
        raise ValueError(f"Unknown SA mode: {mode}. Use 'morris' or 'sobol'.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _prompt(label: str, default, cast, valid_range: str = "") -> any:
    hint = f"  ({valid_range})" if valid_range else ""
    raw = input(f"  {label:<22} [default: {default}]{hint}: ").strip()
    return cast(raw) if raw else default


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n\u2500\u2500 multi-agent simulation (v3) \u2500\u2500")
    print("Modes: [s]imulation (default) | [m]orris screening | [b]sobol analysis\n")

    mode = input("  Mode [s/m/b]: ").strip().lower() or "s"

    if mode in ("m", "morris"):
        run_sensitivity_analysis(mode="morris")

    elif mode in ("b", "sobol"):
        run_sensitivity_analysis(mode="sobol")

    else:
        # Defaults from env vars (set in .env), overridable via interactive prompt
        _e = lambda k, d: os.environ.get(k, d)
        n_ticks        = _prompt("Ticks to simulate",      int(_e("SIM_N_TICKS", "12")),       int,   "1\u2013168")
        n_participants = _prompt("Number of participants",  int(_e("SIM_N_PARTICIPANTS", "4")), int,   "1\u201350")
        k              = _prompt("Observer frequency K",    int(_e("SIM_OBSERVER_K", "4")),     int,   "1\u2013n_ticks")
        alpha          = _prompt("EMA alpha",               float(_e("SIM_ALPHA", "0.15")),     float, "0.1\u20130.4")
        history_window = _prompt("History window (turns)",  int(_e("SIM_HISTORY_WINDOW", "4")), int,   "1\u201312")
        max_tokens     = _prompt("Context size (tokens)",   int(_e("SIM_MAX_TOKENS", "192")),   int,   "64\u20138192")
        seed           = _prompt("Random seed",             int(_e("SIM_SEED", "42")),          int,   "any int")

        print(
            f"\n  n_ticks={n_ticks} | n_participants={n_participants} | K={k} | "
            f"alpha={alpha} | history_window={history_window} | "
            f"max_tokens={max_tokens} | seed={seed}\n"
        )

        async def main():
            try:
                participants, world_state = await run_simulation(
                    n_ticks=n_ticks, n_participants=n_participants,
                    k=k, alpha=alpha, history_window=history_window,
                    max_tokens=max_tokens, seed=seed,
                )
                print("\n\u2500\u2500 Final scores \u2500\u2500")
                for p in participants:
                    print(f"  {p.agent_id}: {p.behavioral_score:.3f}")
                print(f"\n\u2500\u2500 Full log: {len(world_state.full_log())} entries \u2500\u2500")

                final_stats = world_state.compute_score_statistics(n_ticks)
                if final_stats:
                    print(f"\n\u2500\u2500 POM Summary \u2500\u2500")
                    print(f"  Mean: {final_stats['mean']:.3f}")
                    print(f"  Std:  {final_stats['std']:.3f}")
                    print(f"  Skew: {final_stats['skewness']:.2f}")
                    if final_stats.get("autocorrelation_lag1") is not None:
                        print(f"  Lag-1 autocorrelation: {final_stats['autocorrelation_lag1']:.3f}")
            finally:
                await close_client()

        asyncio.run(main())
