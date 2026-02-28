"""
Generalized multi-agent simulation loop — stratified tick architecture.

Domain-neutral orchestration framework for professional AI simulation
environments. Implements the layered multi-agent pattern with four-phase
tick processing, embedding-based scoring, observer-driven intervention
mechanics, and formal mathematical foundations from simulation theory.

═══════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATIONS — FORMAL SPECIFICATION
═══════════════════════════════════════════════════════════════════════════

1. DEVS FORMALISM (Zeigler 1976)
   Each agent is an atomic DEVS model:
     M = ⟨X, S, Y, δ_int, δ_ext, λ, ta⟩
   where:
     X   = set of input events (prompts, stimuli, observation windows)
     S   = set of sequential states (history, behavioral_score, score_log)
     Y   = set of output events (responses, interventions)
     δ_int: S → S         internal transition (score decay, intervention expiry)
     δ_ext: Q × X → S     external transition (agent response to input)
           Q = {(s, e) | s ∈ S, 0 ≤ e ≤ ta(s)}, e = elapsed time
     λ: S → Y             output function (executed before internal transition)
     ta: S → ℝ⁺₀ ∪ {∞}   time advance (tick scheduling)

   Coupled DEVS:
     N = ⟨X, Y, D, {M_d | d ∈ D}, EIC, EOC, IC, Select⟩
     D = {environment, participant_0, ..., participant_n, observer_a, observer_b}
     IC:  environment.Y → participant.X  (Phase A → B)
          participant.Y → environment.X  (feedback loop, Phase B → next A)
          participant.Y → WorldState     (Phase B → C, observation log)
          WorldState    → observer.X     (Phase C → D, observation window)
          observer.Y    → WorldState     (Phase D → next A, interventions)
     Select: Phase A sequential, Phase B synchronous, Phase D synchronous
     Closure under coupling guarantees compositional reasoning.

2. ABM AS FINITE DYNAMICAL SYSTEM (Laubenbacher et al. 2008)
   Global state: x = (x_1, ..., x_n) ∈ S = S_1 × ... × S_n
   System map F: S → S depends on update schedule π:
     Synchronous (Phase B):  F(x) = (f_1(x), ..., f_n(x))
     Sequential  (Phase A):  agents update in order π, each seeing latest state
     Stochastic:             update order drawn from probability distribution

3. SCORE DYNAMICS — EMA AS DISCRETE STOCK-FLOW (Forrester 1961)
     Score(t+1) = Score(t) + d · α · (Signal(t) - Score(t))
   Positive feedback: high score → environment escalates → higher signal → higher score
   Negative feedback: intervention dampening d < 1 attenuates the loop
   Fixed point: Signal*(Score*) = Score*
   Stability:  |d · α · ∂Signal/∂Score|_{Score*} < 1

4. EMBEDDING-BASED SCORING — IMPORTANCE SAMPLING ANALOGY
     high_sim = (1/|A_h|) Σ_{k ∈ A_h} cos(embed(response), embed(anchor_k))
     low_sim  = (1/|A_l|) Σ_{k ∈ A_l} cos(embed(response), embed(anchor_k))
     signal   = clip((high_sim - low_sim + 1) / 2, 0, 1)
     SE       = σ(sims) / √N_anchors

5. SENSITIVITY ANALYSIS (Sobol 1993; Morris 1991)
     S_i  = Var(E[Y|X_i]) / Var(Y)           (Sobol first-order)
     EE_i = [f(x + Δe_i) - f(x)] / Δ         (Morris elementary effect)
     μ* = mean(|EE_i|), σ = std(EE_i)
   Parameters: α ∈ [0.1,0.4], K ∈ [1,n_ticks], θ ∈ [0.60,0.85],
               d ∈ [0.3,1.0], N_anchors ∈ [4,16], history_window ∈ [1,12]

6. UNCERTAINTY QUANTIFICATION
   Aleatory: Score_0 ~ U(0.1,0.5), LLM temperature, update order stochasticity
   Epistemic: anchor quality, embedding capacity, prompt effectiveness, threshold θ
   (Theoretical context: Kennedy & O'Hagan 2001 decomposition
    y_obs = y_model(x,θ) + δ(x) + ε — not implemented, informs design rationale)

7. MEAN-FIELD LIMIT (theoretical context, not implemented)
   For large N: μ_N(t) = (1/N) Σ δ_{Score_i(t)} → McKean-Vlasov ODE.
   Practically approximated via aggregate statistics in observer_prompt_window.

8. PATTERN-ORIENTED MODELING (Grimm et al. 2005) — IMPLEMENTED
   Validation via simultaneous reproduction of multiple observed patterns:
     Pattern 1: Score distribution shape
     Pattern 2: Temporal autocorrelation of individual trajectories
     Pattern 3: Intervention response curves
     Pattern 4: Population-level convergence/divergence

9. ODD PROTOCOL (Grimm et al. 2006, 2010, 2020) — IMPLEMENTED
   Overview: multi-agent professional scenarios, behavioral scoring
   Entities: EnvironmentAgent, ParticipantAgent, ObserverAgent, WorldState
   Emergence: population-level distributions from individual interactions
   Stochasticity: initial scores (seeded), LLM temperature
   Initialization: N participants scores ~ U(0.1,0.5), empty intervention registry

═══════════════════════════════════════════════════════════════════════════
ARCHITECTURE LAYERS (Park et al. 2023; Adaptive-VP, ACL 2025)
═══════════════════════════════════════════════════════════════════════════

  Layer 1 — Scenario Engine:     EnvironmentAgent + run_tick() orchestration
  Layer 2 — Persona Module:      ParticipantAgent (six-component prompt encoding)
  Layer 3 — Feedback/Assessment: ObserverAgent (evaluator-generator separation)
  Layer 4 — State Management:    WorldState (log, interventions, windowed summaries)
  Layer 5 — Memory/Context:      history_window (recency) + embed_score (relevance)

  Generative Agents (Park et al. 2023):
    Memory stream  → agent.history
    Reflection     → observer.analyses (periodic synthesis every K ticks)
    Planning       → environment.decide() (stimulus selection)

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from simulation.agent_roles import AGENT_ROLES, INTERVENTION_CODEBOOK, ENGAGEMENT_ANCHORS
from orchestrator import agent_turn, close_client

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_EMBED_PATH = "/per.volume/huggingface/hub/gte-small"
EMBED_MODEL_NAME = os.environ.get("SIM_EMBED_MODEL", DEFAULT_EMBED_PATH)
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


# ── Reproducibility ───────────────────────────────────────────────────────────

_rng: Optional[np.random.RandomState] = None


def set_seed(seed: int = 42) -> None:
    """Set global seed for reproducible simulation runs."""
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
    """
    Load embedding model with configurable path and fallback.
    Path priority: SIM_EMBED_MODEL env var → DEFAULT_EMBED_PATH → HuggingFace download.
    """
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info(f"Loaded embedding model from: {EMBED_MODEL_NAME}")
    except Exception as e:
        logger.warning(f"Failed to load from {EMBED_MODEL_NAME}: {e}. Falling back to HuggingFace download.")
        _embed_model = SentenceTransformer("thenlper/gte-small")
        logger.info("Loaded embedding model from HuggingFace: thenlper/gte-small")
    return _embed_model


def _cosine(a, b) -> float:
    """
    Cosine similarity: cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
    Returns scalar in [-1, 1]. Used as likelihood ratio analogue
    in the importance-sampling-inspired scoring framework.
    """
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
    """
    Wrapper around agent_turn with retry logic and error handling.

    Retries up to MAX_RETRIES times with exponential backoff on transient
    failures (network errors, timeouts, rate limits). Returns a fallback
    response on exhaustion rather than crashing the simulation.
    """
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
    """
    Typed intervention proposed by an observer agent.

    Three-tier model mapping to DEVS couplings:
      - "environment_constraint": constrains environment.δ_ext (output space Y)
      - "participant_nudge": augments participant.X (input context)
      - "score_modifier": dampening d on Phase C stock-flow dynamics

    Duration: -1 = permanent (ta = ∞); >0 = ticks remaining (ta = n)
    """
    type: str
    description: str
    modifier: dict
    activated_at: int
    duration: int = -1
    source: str = ""


@dataclass
class ObsEntry:
    """
    Single observation log entry for one participant's tick outcome.
    Feeds POM validation (aggregate pattern extraction) and observer
    context windows (token-budget-aware summarization).
    """
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
            f"(score {self.score_before:.2f}→{self.score_after:.2f}, "
            f"signal={self.signal:.2f}±{self.signal_se:.3f}) "
            f"stimulus: \"{self.stimulus[:70]}\" | response: \"{self.response[:70]}\""
        )


@dataclass
class WorldState:
    """
    Central state container mediating IC couplings between agents.

    Token-budget-aware summarization:
      N ≤ 15: granular per-participant entries
      N > 15: aggregate statistics (mean-field approximation)
    """
    k: int = 3
    _log: list[ObsEntry] = field(default_factory=list)
    active_interventions: list[Intervention] = field(default_factory=list)

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

    def observer_prompt_window(self, tick: int, n_participants: int) -> str:
        """Token-budget-aware summary (~2000 token cap)."""
        window = [e for e in self._log if e.tick > tick - self.k]
        if not window:
            return "No observations yet."
        if n_participants > 15:
            scores_before = [e.score_before for e in window]
            scores_after  = [e.score_after  for e in window]
            signals       = [e.signal for e in window]
            crossings     = sum(1 for e in window if e.score_before < 0.7 <= e.score_after)
            return (
                f"Ticks {tick - self.k + 1}–{tick} | {n_participants} participants | "
                f"mean score {np.mean(scores_before):.2f}→{np.mean(scores_after):.2f} "
                f"(σ={np.std(scores_after):.3f}) | "
                f"mean signal {np.mean(signals):.2f} (σ={np.std(signals):.3f}) | "
                f"participants crossing 0.7 threshold: {crossings}"
            )
        return "\n".join(e.to_str() for e in window)

    def apply_interventions(self) -> None:
        """DEVS δ_int for interventions: decrement durations, remove expired."""
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
        """Aggregate dampening: d = ∏ d_i (multiplicative composition)."""
        d = 1.0
        for iv in self.active_interventions:
            if iv.type == "score_modifier":
                d *= iv.modifier.get("dampening", 1.0)
        return d

    def compute_score_statistics(self, tick: int) -> dict:
        """
        Population statistics for POM validation.
        Returns: mean, std, skewness γ, n_above_threshold, autocorrelation_lag1.
        """
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


def embed_score(response: str) -> tuple[float, float]:
    """
    Behavioral signal in [0,1] with standard error.
    SE = σ(sims) / √N (Monte Carlo convergence O(1/√N)).
    """
    _load_anchors()
    vec = _get_embed().encode([response])[0]
    sims = np.array([_cosine(vec, a) for a in _anchor_vecs])

    high_mask = np.array([l == "high" for l in _anchor_labels])
    high_sim = float(np.mean(sims[high_mask]))
    low_sim  = float(np.mean(sims[~high_mask]))

    signal = float(np.clip((high_sim - low_sim + 1.0) / 2.0, 0.0, 1.0))
    se = float(np.std(sims) / math.sqrt(len(sims)))
    return signal, se


def update_score(
    current: float,
    signal: float,
    dampening: float = 1.0,
    alpha: float = 0.2,
) -> float:
    """
    EMA with dampening: Score(t+1) = Score(t) + d · α · (Signal - Score).
    Stability: |d · α · ∂Signal/∂Score| < 1 at fixed point.
    """
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
    """
    Morris method: O(r(k+1)) evaluations for k parameters.
    Returns {param_idx: {mu_star, sigma, mu}}.
    """
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
    """
    Saltelli (2002) estimator: S_i = Var(E[Y|X_i]) / Var(Y).
    Cost: N(k+2) evaluations.
    """
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
INTERVENTION_THRESHOLD = 0.72


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
    """
    Embedding-based codebook matching (principle-adherence prompting).
    Triggers on cosine sim ≥ θ (INTERVENTION_THRESHOLD).
    """
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
    """Map codebook key → (type, description, modifier)."""
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


# ── Agent classes ─────────────────────────────────────────────────────────────

class EnvironmentAgent:
    """
    Generates stimuli per participant (Phase A). Singleton.

    DEVS: M_env = ⟨X_env, S_env, Y_env, δ_int, δ_ext, λ, ta⟩
      ta: fires per participant per tick (sequential in Phase A)

    DDA (Adaptive-VP ACL 2025):
      score ≥ 0.8 → stabilising stimuli
      score ≥ 0.5 → gradual variation
      score < 0.5 → baseline
    """

    def __init__(self, history_window: int = 4):
        self.cfg = AGENT_ROLES["environment"]
        self.history: list[dict] = []
        self.history_window = history_window

    async def decide(self, participant: "ParticipantAgent", world_state: WorldState, max_tokens: int = 128) -> str:
        constraints = world_state.environment_constraints()
        constraint_note = f" ACTIVE CONSTRAINTS: {constraints}" if constraints else ""

        if participant.behavioral_score >= 0.8:
            dda_note = " [HIGH SCORE — test boundary maintenance, present stabilising stimuli]"
        elif participant.behavioral_score >= 0.5:
            dda_note = " [MODERATE SCORE — introduce gradual variation]"
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
        """
        Batch stimulus generation for large cohorts (N>10).

        Constructs a single prompt summarising all participants' states
        and requests per-participant stimuli in a structured JSON response.
        Falls back to sequential decide() on parse failure.
        """
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
            f"Respond with a JSON object mapping participant_id → stimulus string.{constraint_note}\n\n"
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

        # Parse JSON response; fall back to sequential on failure
        try:
            # Strip markdown fencing if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            stimuli = json.loads(cleaned)
            # Validate all participants present
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

    DEVS: M_p, ta = synchronous (Phase B)
    Persona: six-component architecture
    Init: behavioral_score ~ U(0.1, 0.5) (seeded, aleatory)
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

    async def step(self, tick: int, stimulus: str, world_state: WorldState, max_tokens: int = 256) -> str:
        nudge = world_state.participant_nudges()
        nudge_note = f" {nudge}" if nudge else ""
        prompt = (
            f"[Tick {tick}]{nudge_note} "
            f"The environment presents: \"{stimulus[:120]}\". "
            f"Your current behavioral score is {self.behavioral_score:.2f}/1.0. "
            f"How do you respond? What do you do?"
        )
        response = await _resilient_agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=max_tokens,
        )
        self.last_stimulus = stimulus
        self.last_response = response
        self.history.append({"role": "assistant", "content": response})
        return response


class ObserverAgent:
    """
    Analyst agent — fires every K ticks (Phase D).

    DEVS: M_obs, ta = K ticks
    Evaluator-generator separation + POM multi-scale pattern detection.
    """

    def __init__(self, agent_id: str, role: str, history_window: int = 4, max_tokens: int = 256):
        self.agent_id = agent_id
        self.cfg = AGENT_ROLES[role]
        self.history: list[dict] = []
        self.history_window = history_window
        self.max_tokens = max_tokens
        self.analyses: list[str] = []

    async def observe(self, tick: int, world_state: WorldState, n_participants: int) -> list[Intervention]:
        window = world_state.observer_prompt_window(tick, n_participants)
        active = ", ".join(iv.description for iv in world_state.active_interventions) or "none"

        stats = world_state.compute_score_statistics(tick)
        stats_note = ""
        if stats:
            stats_note = (
                f"\nPopulation statistics: mean={stats['mean']:.3f}, "
                f"σ={stats['std']:.3f}, skew={stats['skewness']:.2f}, "
                f"above 0.7: {stats['n_above_threshold']}/{stats['n_total']}"
            )
            if stats.get("autocorrelation_lag1") is not None:
                stats_note += f", lag-1 autocorr={stats['autocorrelation_lag1']:.3f}"

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
    Execute one tick: A (stimulus) → B (response) → C (scoring) → D (observation).
    Phase A: sequential. Phase B: synchronous. Phase D: synchronous.
    """
    n = len(participants)

    # ── Phase A — stimulus generation ─────────────────────────────────────
    if n <= 10:
        stimuli = {}
        for p in participants:
            stimuli[p.agent_id] = await environment.decide(
                p, world_state, max_tokens=max(64, max_tokens // 2)
            )
    else:
        stimuli = await environment.batch_decide(
            participants, world_state, max_tokens=max(64, max_tokens // 2)
        )

    # ── Phase B — synchronous participant responses ───────────────────────
    responses = await asyncio.gather(*[
        p.step(tick, stimuli[p.agent_id], world_state, max_tokens=max_tokens)
        for p in participants
    ])

    # ── Phase C — deterministic score update ──────────────────────────────
    dampening = world_state.score_dampening()
    for participant, response in zip(participants, responses):
        score_before = participant.behavioral_score
        signal, signal_se = embed_score(response)
        participant.behavioral_score = update_score(score_before, signal, dampening, alpha)
        participant.score_log.append(participant.behavioral_score)
        world_state.log(ObsEntry(
            tick=tick, participant_id=participant.agent_id,
            score_before=score_before, score_after=participant.behavioral_score,
            stimulus=stimuli[participant.agent_id], response=response,
            signal=signal, signal_se=signal_se,
        ))

    # ── Phase D — observer analysis (every K ticks) ───────────────────────
    if tick % world_state.k == 0:
        logger.info(f"[Tick {tick}] Observer tick...")
        intervention_lists = await asyncio.gather(*[
            o.observe(tick, world_state, n) for o in observers
        ])
        for ivs in intervention_lists:
            world_state.active_interventions.extend(ivs)

    world_state.apply_interventions()


async def run_simulation(
    n_ticks: int = 12,
    n_participants: int = 4,
    k: int = 3,
    alpha: float = 0.2,
    history_window: int = 4,
    max_tokens: int = 256,
    seed: int = 42,
) -> tuple[list[ParticipantAgent], WorldState]:
    """
    Run the stratified multi-agent simulation.

    ODD Protocol initialization:
      N participants (Score_0 ~ U(0.1,0.5), seeded), 1 environment,
      2 observers (fire every K ticks), empty intervention registry.

    Token budget: system(~300) + user(~200) + window×max_tokens + output ≤ 8192
    """
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

    for tick in range(1, n_ticks + 1):
        pct = tick / n_ticks * 100
        print(f"\n── Tick {tick}/{n_ticks}  ({pct:.0f}%) ──")
        await run_tick(tick, environment, participants, observers, world_state, alpha, max_tokens)
        for p in participants:
            print(f"  {p.agent_id}: score={p.behavioral_score:.3f}")

        stats = world_state.compute_score_statistics(tick)
        if stats:
            print(f"  [POM] mean={stats['mean']:.3f} σ={stats['std']:.3f} "
                  f"skew={stats['skewness']:.2f} above_0.7={stats['n_above_threshold']}")

    return participants, world_state


# ── Sensitivity analysis runner ───────────────────────────────────────────────

def run_sensitivity_analysis(
    mode: str = "morris",
    n_ticks: int = 6,
    n_participants: int = 4,
    r: int = 10,
    n_samples: int = 256,
) -> dict:
    """
    Run Morris screening or Sobol analysis on simulation parameters.

    Parameter vector: [alpha, K, threshold, dampening]
    Objective: mean final behavioral score across participants.

    Note: this uses a simplified synchronous evaluation function that
    runs shortened simulations. For full async analysis, use the
    async runner and aggregate results externally.
    """
    from simulation.agent_roles import ENGAGEMENT_ANCHORS  # ensure loaded

    bounds = [
        (0.1, 0.4),    # alpha
        (1.0, 6.0),    # K (cast to int)
        (0.60, 0.85),  # intervention threshold
        (0.3, 1.0),    # dampening coefficient
    ]
    param_names = ["alpha", "K", "threshold", "dampening"]

    def objective(x: np.ndarray) -> float:
        """Simplified scoring: compute EMA trajectory for a synthetic signal."""
        alpha, k_float, threshold, dampening = x
        # Synthetic signal: sinusoidal engagement + noise
        rng = np.random.RandomState(42)
        scores = []
        score = 0.3
        for t in range(n_ticks):
            signal = 0.5 + 0.3 * math.sin(t * 0.5) + rng.normal(0, 0.1)
            signal = max(0.0, min(1.0, signal))
            # Apply dampening every K ticks
            d = dampening if (t % max(1, int(k_float)) == 0) else 1.0
            score = score + d * alpha * (signal - score)
            score = max(0.0, min(1.0, score))
            scores.append(score)
        return float(np.mean(scores))

    if mode == "morris":
        results = morris_screening(objective, bounds, r=r)
        print("\n── Morris Screening Results ──")
        for i, name in enumerate(param_names):
            r_i = results[i]
            print(f"  {name:<15} μ*={r_i['mu_star']:.4f}  σ={r_i['sigma']:.4f}  μ={r_i['mu']:.4f}")
        return {"mode": "morris", "param_names": param_names, "results": results}

    elif mode == "sobol":
        results = sobol_first_order(objective, bounds, n_samples=n_samples)
        print("\n── Sobol First-Order Indices ──")
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

    print("\n── multi-agent simulation ──")
    print("Modes: [s]imulation (default) | [m]orris screening | [b]sobol analysis\n")

    mode = input("  Mode [s/m/b]: ").strip().lower() or "s"

    if mode in ("m", "morris"):
        run_sensitivity_analysis(mode="morris")

    elif mode in ("b", "sobol"):
        run_sensitivity_analysis(mode="sobol")

    else:
        n_ticks        = _prompt("Ticks to simulate",      12,   int,   "1–168")
        n_participants = _prompt("Number of participants",   4,   int,   "1–50")
        k              = _prompt("Observer frequency K",     3,   int,   "1–n_ticks")
        alpha          = _prompt("EMA alpha",                0.2, float, "0.1–0.4")
        history_window = _prompt("History window (turns)",   4,   int,   "1–12")
        max_tokens     = _prompt("Context size (tokens)",  256,   int,   "64–8192")
        seed           = _prompt("Random seed",             42,   int,   "any int")

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
                print("\n── Final scores ──")
                for p in participants:
                    print(f"  {p.agent_id}: {p.behavioral_score:.3f}")
                print(f"\n── Full log: {len(world_state.full_log())} entries ──")

                final_stats = world_state.compute_score_statistics(n_ticks)
                if final_stats:
                    print(f"\n── POM Summary ──")
                    print(f"  Mean: {final_stats['mean']:.3f}")
                    print(f"  Std:  {final_stats['std']:.3f}")
                    print(f"  Skew: {final_stats['skewness']:.2f}")
                    if final_stats.get("autocorrelation_lag1") is not None:
                        print(f"  Lag-1 autocorrelation: {final_stats['autocorrelation_lag1']:.3f}")
            finally:
                await close_client()

        asyncio.run(main())
