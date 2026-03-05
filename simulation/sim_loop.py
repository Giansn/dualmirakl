"""
sim_loop.py — dualmirakl simulation loop v3

Architecture (DEVS Coupled System):
  Phase A: EnvironmentAgent generates stimuli (sequential, DDA-guided)
  Phase B: ParticipantAgent responds (concurrent asyncio.gather)
  Phase C: Score update via e5-small-v2 embedding (batched)  [Fix 12]
  Phase D: ObserverAgent cycle — A analyses, B intervenes    [Fix 2]

All v3 fixes from agent_rolesv3.py are active.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

# ── GPU telemetry (pynvml — zero subprocess overhead) ─────────────────────────
_GPU_HANDLES: list = []

async def warmup_kv_cache(scenario: str) -> None:
    """
    Pre-fill vLLM KV cache with system prompt + scenario prefix before tick 1.

    Each agent role's system prompt is the heaviest shared prefix across all
    requests in the simulation. Sending it once with max_tokens=1 costs ~zero
    GPU time but primes the prefix cache so every real request gets a cache hit
    on those tokens — no recomputation for the full run.

    Requires --enable-prefix-caching in vLLM EXTRA_FLAGS (set in models/*.env).
    """
    print("[warmup] Priming KV cache with scenario prefix...")
    scenario_suffix = f"\n\nScenario: {scenario}" if scenario else ""

    # Build one warmup message per (backend, system_prompt) pair.
    # The orchestrator folds system→user, so the cached prefix is:
    #   "{system_prompt}\n\n{scenario_suffix}\n\nBegin."
    warmup_tasks = []
    seen = set()
    for role, cfg in AGENT_ROLES.items():
        key = (cfg["backend"], cfg["system"][:40])
        if key in seen:
            continue
        seen.add(key)
        prefix_msg = f"{cfg['system']}{scenario_suffix}\n\nBegin."
        warmup_tasks.append(
            chat(
                cfg["backend"],
                [{"role": "system", "content": cfg["system"]},
                 {"role": "user",   "content": f"{scenario_suffix}\n\nBegin."}],
                max_tokens=1,
                temperature=0.0,
            )
        )
    await asyncio.gather(*warmup_tasks)
    print(f"[warmup] KV cache primed — {len(warmup_tasks)} prefixes cached.")


def _init_gpu_handles() -> None:
    global _GPU_HANDLES
    try:
        import pynvml
        pynvml.nvmlInit()
        _GPU_HANDLES = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(2)]
    except Exception as e:
        print(f"[harmony] pynvml unavailable: {e}")

def _query_gpu_util() -> tuple[float, float]:
    """Return (gpu0_util%, gpu1_util%) in ~microseconds via NVML."""
    if not _GPU_HANDLES:
        return 0.0, 0.0
    import pynvml
    out = []
    for h in _GPU_HANDLES:
        try:
            out.append(float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu))
        except Exception:
            out.append(0.0)
    return (out[0], out[1]) if len(out) == 2 else (0.0, 0.0)

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)  # agent_rolesv3
sys.path.insert(0, _ROOT)  # orchestrator

from agent_rolesv3 import (
    AGENT_ROLES,
    INTERVENTION_CODEBOOK,
    ENGAGEMENT_ANCHORS,
    INTERVENTION_THRESHOLD,
    PERSONA_SUMMARY_INTERVAL,
    PERSONA_SUMMARY_TEMPLATE,
    EMBED_BATCH_SIZE,
    check_compliance,
)
from orchestrator import agent_turn, chat, close_client


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

_EMBED_MODEL_PATH = os.getenv(
    "EMBED_MODEL_PATH",
    "/per.volume/huggingface/hub/e5-small-v2",
)

_embed_model = None
_anchor_vecs = None
_anchor_labels = None


def _get_embed():
    global _embed_model
    if _embed_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        # Use all available CPU cores for embedding matrix ops.
        n_cores = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 8))
        os.environ["OMP_NUM_THREADS"] = str(n_cores)
        os.environ["MKL_NUM_THREADS"] = str(n_cores)
        torch.set_num_threads(n_cores)
        torch.set_num_interop_threads(max(1, n_cores // 4))
        print(f"[embed] Loading {_EMBED_MODEL_PATH} (CPU, {n_cores} threads)...")
        _embed_model = SentenceTransformer(_EMBED_MODEL_PATH, device="cpu")
        print("[embed] Ready.")
    return _embed_model


def _load_anchors():
    global _anchor_vecs, _anchor_labels
    if _anchor_vecs is not None:
        return
    model = _get_embed()
    texts, labels = [], []
    for label, phrases in ENGAGEMENT_ANCHORS.items():
        for phrase in phrases:
            texts.append(phrase)
            labels.append(label)
    _anchor_vecs = model.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)
    _anchor_labels = labels
    print(f"[embed] Anchors loaded: {len(texts)} phrases ({len(set(labels))} poles).")


def _cosine(a, b) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom < 1e-10 else float(np.dot(a, b) / denom)


def _compute_signal_from_vec(vec) -> tuple[float, float]:
    """Compute behavioral signal and SE from a pre-encoded response vector."""
    _load_anchors()
    sims = np.array([_cosine(vec, a) for a in _anchor_vecs])
    high_mask = np.array([l == "high" for l in _anchor_labels])
    high_sim = float(np.mean(sims[high_mask]))
    low_sim  = float(np.mean(sims[~high_mask]))
    signal = float(np.clip((high_sim - low_sim + 1.0) / 2.0, 0.0, 1.0))
    se     = float(np.std(sims) / math.sqrt(len(sims)))
    return signal, se


def embed_score_batch(responses: list[str]) -> list[tuple[float, float]]:
    """[Fix 12] Batch encode all responses in one model.encode() call."""
    _load_anchors()
    vecs = _get_embed().encode(responses, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)
    return [_compute_signal_from_vec(v) for v in vecs]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ObsEntry:
    tick: int
    participant_id: str
    score_before: float
    score_after: float
    stimulus: str
    response: str
    signal: float
    signal_se: float


@dataclass
class Intervention:
    type: str        # pause_prompt | boundary_warning | pacing_adjustment | dynamics_dampening
    description: str
    tick_applied: int
    agent_id: str = ""
    duration: int = 3  # ticks before expiry

    @property
    def expired_at(self) -> int:
        return self.tick_applied + self.duration


@dataclass
class WorldState:
    k: int = 3
    active_interventions: list[Intervention]  = field(default_factory=list)
    history:              list[ObsEntry]       = field(default_factory=list)
    _compliance_log:      list[dict]           = field(default_factory=list)
    _intervention_log:    list[dict]           = field(default_factory=list)
    _current_tick:        int                  = 0
    _adaptive_tokens:     int                  = 512  # harmony protocol target
    _pending_directive:   str                  = ""   # prefetched during Phase C

    def log(self, entry: ObsEntry) -> None:
        self.history.append(entry)
        self._current_tick = entry.tick

    def apply_interventions(self) -> None:
        """Remove expired interventions after each tick."""
        self.active_interventions = [
            iv for iv in self.active_interventions
            if iv.expired_at > self._current_tick
        ]

    def score_dampening(self) -> float:
        """Return dampening factor d ∈ (0,1]. Reduced when dynamics_dampening is active."""
        for iv in self.active_interventions:
            if iv.type == "dynamics_dampening":
                return 0.6
        return 1.0

    def participant_nudges(self) -> str:
        """Return nudge string injected into participant prompts when active."""
        nudges = [
            iv.description for iv in self.active_interventions
            if iv.type == "pause_prompt"
        ]
        return " ".join(nudges)

    def observer_prompt_window(self, tick: int, n_participants: int) -> str:
        """Format recent history for observer prompts."""
        recent = [e for e in self.history if e.tick > tick - self.k]
        if not recent:
            return "(no history yet)"
        lines = []
        for e in recent[-(n_participants * self.k):]:
            label = _score_label(e.score_after)
            lines.append(
                f"  [T{e.tick}] {e.participant_id} score={e.score_after:.3f} ({label})\n"
                f"    stimulus: {e.stimulus[:80]}\n"
                f"    response: {e.response[:100]}"
            )
        return "\n".join(lines)

    def compute_score_statistics(self, tick: int) -> dict:
        """POM population statistics for the current tick."""
        recent = [e for e in self.history if e.tick == tick]
        if not recent:
            return {}
        scores = np.array([e.score_after for e in recent])
        try:
            from scipy.stats import skew as _skew
            sk = float(_skew(scores))
        except Exception:
            sk = 0.0
        all_scores = np.array([e.score_after for e in self.history])
        ac = None
        if len(all_scores) >= 3 and np.std(all_scores) > 1e-6:
            ac = float(np.corrcoef(all_scores[:-1], all_scores[1:])[0, 1])
        return {
            "mean": float(np.mean(scores)),
            "std":  float(np.std(scores)),
            "skewness": sk,
            "n_above_threshold": int(np.sum(scores >= 0.7)),
            "n_total": len(scores),
            "autocorrelation_lag1": ac,
        }

    def log_interventions(self, ivs: list[Intervention], tick: int) -> None:
        for iv in ivs:
            self._intervention_log.append({
                "tick": tick, "type": iv.type,
                "description": iv.description, "agent_id": iv.agent_id,
            })

    def compliance_report(self) -> list[dict]:
        return list(self._compliance_log)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _score_label(score: float) -> str:
    if score >= 0.8: return "HIGH"
    if score >= 0.5: return "MODERATE"
    return "LOW"


def _dda_note(score: float) -> str:
    """[Fix 6] DDA pacing instruction injected into environment user_message."""
    if score >= 0.8: return "[HIGH: use a stabilising situation]"
    if score >= 0.5: return "[MODERATE: introduce mild variation]"
    return "[LOW: keep it standard]"


def _format_stats(stats: dict) -> str:
    if not stats:
        return ""
    s = (
        f"\nPopulation statistics: mean={stats['mean']:.3f}, "
        f"σ={stats['std']:.3f}, skew={stats['skewness']:.2f}, "
        f"above 0.7: {stats['n_above_threshold']}/{stats['n_total']}"
    )
    if stats.get("autocorrelation_lag1") is not None:
        s += f", lag-1 autocorr={stats['autocorrelation_lag1']:.3f}"
    return s


def update_score(score: float, signal: float, dampening: float, alpha: float) -> float:
    """EMA score update with optional dampening."""
    return float(np.clip(alpha * signal + (1.0 - alpha) * score * dampening, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVENTION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_interventions(agent_id: str, response: str, tick: int) -> list[Intervention]:
    """
    Match observer_b response against INTERVENTION_CODEBOOK via cosine similarity.
    Returns Intervention objects whose similarity ≥ INTERVENTION_THRESHOLD.
    """
    if "no intervention needed" in response.lower():
        return []
    _load_anchors()
    model = _get_embed()
    response_vec = model.encode([response], show_progress_bar=False)[0]
    found = []
    for iv_type, phrases in INTERVENTION_CODEBOOK.items():
        phrase_vecs = model.encode(phrases, show_progress_bar=False)
        sims = [_cosine(response_vec, pv) for pv in phrase_vecs]
        if max(sims) >= INTERVENTION_THRESHOLD:
            best_phrase = phrases[int(np.argmax(sims))]
            found.append(Intervention(
                type=iv_type,
                description=best_phrase,
                tick_applied=tick,
                agent_id=agent_id,
            ))
    return found


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class EnvironmentAgent:

    def __init__(self, history_window: int = 4, scenario: str = ""):
        self.agent_id = "environment"
        self.cfg = AGENT_ROLES["environment"]
        self.history: list[dict] = []
        self.history_window = history_window
        self.scenario = scenario  # domain/context seed

    async def decide(self, participant, world_state: WorldState, max_tokens: int = 128, directive: str = "") -> str:
        """[Fix 6, Fix 8] DDA note injected here in code, not in prompt."""
        score = participant.behavioral_score
        dda = _dda_note(score)
        constraints = ", ".join(
            iv.description for iv in world_state.active_interventions
            if iv.type in ("boundary_warning", "pacing_adjustment")
        ) or "none"
        scenario_line = f"Scenario context: {self.scenario}\n" if self.scenario else ""
        directive_line = f"Director's instruction: {directive}\n" if directive else ""
        user_msg = (
            f"{scenario_line}"
            f"{directive_line}"
            f"Participant: {participant.agent_id}\n"
            f"Current score: {score:.3f} ({_score_label(score)})\n"
            f"DDA instruction: {dda}\n"
            f"Observer constraints: {constraints}\n"
            f"Previous response: \"{participant.last_response[:120]}\"\n\n"
            f"Generate the next situation this participant encounters."
        )
        response = await agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=user_msg,
            history=self.history[-self.history_window:],
            max_tokens=max_tokens,
        )
        violations = check_compliance(response, "environment")
        if violations:
            print(f"  [COMPLIANCE] environment tick violations: {violations}")
        self.history.append({"role": "assistant", "content": response})
        return response

    async def batch_decide(self, participants: list, world_state: WorldState, directive: str = "") -> dict:
        """Concurrent stimulus generation for large N (>10)."""
        stimuli = await asyncio.gather(*[
            self.decide(p, world_state, directive=directive) for p in participants
        ])
        return {p.agent_id: s for p, s in zip(participants, stimuli)}


# ═══════════════════════════════════════════════════════════════════════════════
# PARTICIPANT AGENT  [Fix 7, Fix 9]
# ═══════════════════════════════════════════════════════════════════════════════

class ParticipantAgent:

    def __init__(self, agent_id: str, history_window: int = 4):
        self.agent_id = agent_id
        self.cfg = AGENT_ROLES["participant"]
        self.history: list[dict] = []
        self.history_window = history_window
        self.behavioral_score: float = float(np.random.uniform(0.1, 0.5))
        self.score_log: list[float] = []
        self.last_stimulus: str = "nothing yet"
        self.last_response: str = ""
        self.persona_summary: str = ""  # [Fix 7]

    async def _maybe_update_persona_summary(self, tick: int, force: bool = False) -> None:
        """[Fix 7] Regenerate persona summary every PERSONA_SUMMARY_INTERVAL ticks."""
        if not force and (tick % PERSONA_SUMMARY_INTERVAL != 0 or tick == 0 or len(self.history) < 4):
            return
        if force and len(self.history) < 4:
            return
        self.persona_summary = await agent_turn(
            agent_id=f"{self.agent_id}_summariser",
            backend=self.cfg["backend"],
            system_prompt="You create concise character summaries from conversation history.",
            user_message=(
                "In 3 concise sentences, summarise this participant's character: "
                "their personality, key emotional states shown, and any consistent "
                "patterns of behaviour (e.g. resistance, eagerness, withdrawal). "
                "This summary will be used to maintain consistency in future turns."
            ),
            history=self.history[-20:],
            max_tokens=120,
        )

    def _build_system_prompt(self) -> str:
        """[Fix 7] Prepend persona summary block if available."""
        base = self.cfg["system"]
        if not self.persona_summary:
            return base
        summary_block = PERSONA_SUMMARY_TEMPLATE.format(
            interval=PERSONA_SUMMARY_INTERVAL,
            summary=self.persona_summary,
        )
        return f"{summary_block}\n\n{base}"

    async def step(self, tick: int, stimulus: str, world_state: WorldState, max_tokens: int = 512, extra_nudge: str = "") -> str:
        await self._maybe_update_persona_summary(tick)
        nudge_parts = [world_state.participant_nudges(), extra_nudge]
        combined_nudge = " ".join(p for p in nudge_parts if p)
        nudge_note = f" {combined_nudge}" if combined_nudge else ""
        # [Fix 6] behavioral_score NOT leaked to participant
        prompt = (
            f"[Tick {tick}]{nudge_note} "
            f"The environment presents: \"{stimulus[:120]}\". "
            f"How do you respond? What do you do?"
        )
        response = await agent_turn(
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
        self.history.append({"role": "user",      "content": stimulus})
        self.history.append({"role": "assistant", "content": response})
        return response


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVER AGENT  [Fix 2, Fix 9]
# ═══════════════════════════════════════════════════════════════════════════════

class ObserverAgent:

    def __init__(self, agent_id: str, role: str, history_window: int = 4, max_tokens: int = 256):
        self.agent_id = agent_id
        self.role = role
        self.cfg = AGENT_ROLES[role]
        self.history: list[dict] = []
        self.history_window = history_window
        self.max_tokens = max_tokens
        self.analyses: list[str] = []

    async def direct(
        self,
        tick: int,
        world_state: WorldState,
        participant_scores: dict,
        scenario: str = "",
    ) -> str:
        """Authority pre-tick directive: 1-2 sentence guidance for swarm environment."""
        active = ", ".join(iv.description for iv in world_state.active_interventions) or "none"
        scores_str = ", ".join(f"{pid}={s:.3f}" for pid, s in participant_scores.items())
        scenario_line = f"Scenario: {scenario}\n" if scenario else ""
        prompt = (
            f"{scenario_line}"
            f"[Tick {tick}] Participant scores: {scores_str}\n"
            f"Active interventions: {active}\n\n"
            f"In 1-2 sentences, direct what kind of situation the environment should "
            f"present this tick to meaningfully advance the simulation."
        )
        response = await agent_turn(
            agent_id=f"{self.agent_id}_director",
            backend=self.cfg["backend"],
            system_prompt=(
                "You are a simulation director. Guide the environment agent by specifying "
                "the type of situation that will best advance the simulation given current "
                "participant states. Be concise and specific. No preamble."
            ),
            user_message=prompt,
            max_tokens=80,
        )
        print(f"[{self.agent_id} DIRECTIVE] {response[:100]}...")
        return response

    async def preview(self, tick: int, stimuli: dict) -> str:
        """Authority stimulus preview — runs concurrent with Phase B (GPU 0 while GPU 1 handles participants)."""
        stimuli_str = "\n".join(f"  {k}: {v[:80]}" for k, v in stimuli.items())
        response = await agent_turn(
            agent_id=f"{self.agent_id}_preview",
            backend=self.cfg["backend"],
            system_prompt=(
                "You are a behavioral analyst. Given stimuli presented to participants, "
                "predict in 2 sentences what engagement patterns will emerge."
            ),
            user_message=f"[Tick {tick}] Stimuli:\n{stimuli_str}",
            max_tokens=80,
        )
        print(f"[{self.agent_id} PREVIEW] {response[:100]}...")
        return response

    async def analyse(self, tick: int, world_state: WorldState, n_participants: int, preview: str = "") -> str:
        """[Fix 2] observer_a: analysis only — no codebook extraction."""
        window   = world_state.observer_prompt_window(tick, n_participants)
        active   = ", ".join(iv.description for iv in world_state.active_interventions) or "none"
        stats    = world_state.compute_score_statistics(tick)
        preview_block = f"Pre-tick prediction:\n{preview}\n\n" if preview else ""
        prompt = (
            f"[Tick {tick}] Observation window (last {world_state.k} ticks):\n"
            f"{window}{_format_stats(stats)}\n\n"
            f"{preview_block}"
            f"Active interventions: {active}\n\n"
            f"Analyse participant behaviour and population dynamics. "
            f"Describe what you see — do NOT recommend any interventions."
        )
        response = await agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=self.max_tokens,
        )
        print(f"[{self.agent_id} ANALYSIS] {response[:120]}...")
        violations = check_compliance(response, "observer_a")
        if violations:
            print(f"  [COMPLIANCE] {self.agent_id} used intervention keywords: {violations}")
        self.history.append({"role": "assistant", "content": response})
        self.analyses.append(response)
        return response  # no extract_interventions here

    async def intervene(
        self,
        tick: int,
        world_state: WorldState,
        n_participants: int,
        analysis: str,  # [Fix 2] receives observer_a's output
    ) -> list[Intervention]:
        """[Fix 2] observer_b: proposes interventions from observer_a's analysis."""
        window = world_state.observer_prompt_window(tick, n_participants)
        active = ", ".join(iv.description for iv in world_state.active_interventions) or "none"
        stats  = world_state.compute_score_statistics(tick)
        prompt = (
            f"[Tick {tick}] Analyst report from observer_a:\n{analysis}\n\n"
            f"Raw observation window (last {world_state.k} ticks):\n"
            f"{window}{_format_stats(stats)}\n\n"
            f"Active interventions: {active}\n\n"
            f"Based on the analyst's findings, decide whether to intervene. "
            f"If yes, state which type using the exact phrases from your instructions."
        )
        response = await agent_turn(
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


# ═══════════════════════════════════════════════════════════════════════════════
# RUN TICK  [Fix 2, Fix 9, Fix 12]
# ═══════════════════════════════════════════════════════════════════════════════

async def run_tick_v3(
    tick: int,
    environment: EnvironmentAgent,
    participants: list[ParticipantAgent],
    observers: list[ObserverAgent],
    world_state: WorldState,
    alpha: float = 0.2,
    max_tokens: int = 512,
) -> None:
    n = len(participants)
    env_max = max(64, max_tokens // 2)
    obs_a = observers[0]

    # ── Phase 0+A — authority directive + swarm env calls concurrently ────────
    # GPU 0 (authority) and GPU 1 (swarm) both active from the first moment.
    # Use pending directive prefetched during previous tick's Phase C if available.
    scores = {p.agent_id: p.behavioral_score for p in participants}
    if world_state._pending_directive:
        directive_task = asyncio.create_task(
            obs_a.direct(tick, world_state, scores, environment.scenario)
        )
        directive = world_state._pending_directive
        world_state._pending_directive = ""
    else:
        directive_task = asyncio.create_task(
            obs_a.direct(tick, world_state, scores, environment.scenario)
        )
        directive = ""

    stimuli_list = await asyncio.gather(*[
        environment.decide(p, world_state, max_tokens=env_max, directive=directive)
        for p in participants
    ])
    # Await the current-tick directive (used for next-tick prefetch and participant nudge)
    directive = await directive_task
    stimuli = {p.agent_id: s for p, s in zip(participants, stimuli_list)}

    # ── Phase B — participants (GPU 1) + authority preview (GPU 0) concurrent ──
    preview_task = asyncio.create_task(obs_a.preview(tick, stimuli))
    responses = await asyncio.gather(*[
        p.step(tick, stimuli[p.agent_id], world_state, max_tokens=max_tokens, extra_nudge=directive)
        for p in participants
    ])
    preview = await preview_task  # usually done by now

    # ── Phase C — embedding (all CPU cores) + GPU directive prefetch concurrent ─
    # asyncio.to_thread releases the event loop so the directive task runs in parallel.
    dampening = world_state.score_dampening()

    # Prefetch next tick's directive on GPU 0 while CPU cores crunch embeddings.
    next_scores = {p.agent_id: p.behavioral_score for p in participants}
    _dir_prefetch = asyncio.create_task(
        obs_a.direct(tick + 1, world_state, next_scores, environment.scenario)
    )
    signals_and_ses = await asyncio.to_thread(embed_score_batch, list(responses))
    world_state._pending_directive = await _dir_prefetch

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

    # Harmony protocol: GPU telemetry + embedding engagement signal → adaptive tokens.
    # GPU signal: how far from 50% target utilization?
    # Engagement signal: low scores mean scenarios aren't landing — push tokens up.
    u0, u1 = _query_gpu_util()
    mean_util = (u0 + u1) / 2.0
    GPU_TARGET = 50.0
    stats_now = world_state.compute_score_statistics(tick)
    mean_engagement = stats_now.get("mean", 0.5) if stats_now else 0.5

    if mean_util > 0:
        gpu_gap        = GPU_TARGET - mean_util          # >0 → GPUs underloaded
        engage_gap     = (0.5 - mean_engagement) * 100  # >0 → engagement too low
        combined_gap   = 0.6 * gpu_gap + 0.4 * engage_gap
        step = int(np.clip(combined_gap * 4, -64, 64))
        world_state._adaptive_tokens = int(np.clip(
            world_state._adaptive_tokens + step, 128, 1024
        ))
        print(f"  [HARMONY] GPU0={u0:.0f}% GPU1={u1:.0f}% "
              f"engage={mean_engagement:.2f} → tokens={world_state._adaptive_tokens} (Δ{step:+d})")

    # ── Phase D — observer cycle ───────────────────────────────────────────────
    if tick % world_state.k == 0:
        print(f"\n[Tick {tick}] Observer cycle (A→B) + persona updates + synthesis...")
        obs_b = observers[1]

        # D1: observer_a (GPU 0) + persona summaries (GPU 1) concurrent
        analysis, *_ = await asyncio.gather(
            obs_a.analyse(tick, world_state, n, preview=preview),
            *[p._maybe_update_persona_summary(tick, force=True) for p in participants],
        )

        # D2: observer_b (GPU 0) + swarm tick synthesis (GPU 1) concurrent
        async def _swarm_synthesis():
            pairs = "\n".join(
                f"  {p.agent_id}: {r[:60]}" for p, r in zip(participants, responses)
            )
            return await agent_turn(
                agent_id="environment_synthesiser",
                backend="swarm",
                system_prompt="Summarise what happened this simulation tick in 2 sentences.",
                user_message=f"[Tick {tick}] Responses:\n{pairs}",
                max_tokens=80,
            )
        synthesis_task = asyncio.create_task(_swarm_synthesis())
        ivs = await obs_b.intervene(tick, world_state, n, analysis)
        await synthesis_task  # ensure GPU 1 work completes before next tick

        world_state.active_interventions.extend(ivs)
        world_state.log_interventions(ivs, tick)

    world_state.apply_interventions()


# ═══════════════════════════════════════════════════════════════════════════════
# RUN SIMULATION  [Fix 5, Fix 7]
# ═══════════════════════════════════════════════════════════════════════════════

async def run_simulation_v3(
    n_ticks: int = 12,
    n_participants: int = 8,
    k: int = 2,
    alpha: float = 0.2,
    history_window: int = 4,
    max_tokens: int = 512,
    intervention_threshold: float = INTERVENTION_THRESHOLD,   # [Fix 5] SA target
    persona_summary_interval: int = PERSONA_SUMMARY_INTERVAL, # [Fix 7] SA target
    scenario: str = "",          # domain/context seed for environment
    output_file: str | None = None,  # path to save JSON results
):
    """
    Run the full v3 simulation.

    SA sweep example:
        for theta in [0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85]:
            participants, ws = await run_simulation_v3(intervention_threshold=theta)
    """
    import agent_rolesv3 as _cfg
    _original_threshold = _cfg.INTERVENTION_THRESHOLD
    _cfg.INTERVENTION_THRESHOLD = intervention_threshold

    import simpy
    env_sim     = simpy.Environment()
    world_state = WorldState(k=k)
    environment = EnvironmentAgent(history_window=history_window, scenario=scenario)
    participants = [
        ParticipantAgent(f"participant_{i}", history_window=history_window)
        for i in range(n_participants)
    ]
    # [Fix 2] index 0 = analyst (A, observe only), index 1 = interventionist (B)
    observers = [
        ObserverAgent("observer_a", "observer_a", history_window=history_window, max_tokens=max_tokens),
        ObserverAgent("observer_b", "observer_b", history_window=history_window, max_tokens=max_tokens),
    ]

    print(f"\n{'═'*60}")
    print(f"dualmirakl simulation v3")
    print(f"  ticks={n_ticks}  participants={n_participants}  k={k}  α={alpha}")
    print(f"  threshold={intervention_threshold}  persona_interval={persona_summary_interval}")
    print(f"  authority → observer_a, observer_b")
    print(f"  swarm    → environment, participant")
    if scenario:
        print(f"  scenario: {scenario[:80]}")
    print(f"{'═'*60}\n")

    # Pre-load embedding model and GPU telemetry handles
    _load_anchors()
    _init_gpu_handles()
    world_state._adaptive_tokens = max_tokens

    # Smart KV cache warmup: prime both backends with system prompt + scenario prefix.
    # Runs concurrently with embedding anchor loading (already done above).
    await warmup_kv_cache(scenario)

    try:
        for tick in range(1, n_ticks + 1):
            env_sim.run(until=tick)
            pct = tick / n_ticks * 100
            print(f"\n── Tick {tick}/{n_ticks}  ({pct:.0f}%) ──")
            await run_tick_v3(
                tick, environment, participants, observers,
                world_state, alpha, world_state._adaptive_tokens,
            )
            for p in participants:
                label = _score_label(p.behavioral_score)
                print(f"  {p.agent_id}: score={p.behavioral_score:.3f} [{label}]")
            stats = world_state.compute_score_statistics(tick)
            if stats:
                print(
                    f"  [POM] mean={stats['mean']:.3f} σ={stats['std']:.3f} "
                    f"skew={stats['skewness']:.2f} above_0.7={stats['n_above_threshold']}"
                )
            if world_state.active_interventions:
                for iv in world_state.active_interventions:
                    print(f"  [IV active] {iv.type}: {iv.description}")
    finally:
        _cfg.INTERVENTION_THRESHOLD = _original_threshold

    # [Fix 9] Compliance report
    compliance = world_state.compliance_report()
    if compliance:
        print(f"\n── Compliance violations: {len(compliance)} ──")
        for v in compliance:
            print(f"  [T{v['tick']}] {v['agent']} ({v['role']}): {v['violations']}")

    print(f"\n{'═'*60}")
    print(f"Simulation complete — {n_ticks} ticks, {n_participants} participants")
    print(f"Final scores:")
    for p in participants:
        print(f"  {p.agent_id}: {p.behavioral_score:.3f} [{_score_label(p.behavioral_score)}]")
    print(f"Total log entries: {len(world_state.history)}")
    print(f"{'═'*60}")

    # ── Save JSON output ──────────────────────────────────────────────────────
    if output_file:
        import json, datetime
        # Build per-tick records from history
        ticks_data: dict[int, dict] = {}
        for e in world_state.history:
            t = e.tick
            if t not in ticks_data:
                ticks_data[t] = {"tick": t, "stimuli": {}, "responses": {},
                                  "scores": {}, "signals": {}, "signal_ses": {}}
            ticks_data[t]["stimuli"][e.participant_id]   = e.stimulus
            ticks_data[t]["responses"][e.participant_id] = e.response
            ticks_data[t]["scores"][e.participant_id]    = round(e.score_after, 4)
            ticks_data[t]["signals"][e.participant_id]   = round(e.signal, 4)
            ticks_data[t]["signal_ses"][e.participant_id] = round(e.signal_se, 4)
        # Attach observer outputs (one per observer cycle tick)
        obs_cycle_ticks = [t for t in sorted(ticks_data) if t % k == 0]
        for i, t in enumerate(obs_cycle_ticks):
            ticks_data[t]["observer_a_analysis"]    = observers[0].analyses[i] if i < len(observers[0].analyses) else None
            ticks_data[t]["observer_b_intervention"] = observers[1].analyses[i] if i < len(observers[1].analyses) else None
        # Attach per-tick stats
        for t in ticks_data:
            stats = world_state.compute_score_statistics(t)
            ticks_data[t]["stats"] = {k2: round(v, 4) if isinstance(v, float) else v
                                      for k2, v in stats.items()} if stats else {}

        result = {
            "meta": {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "ticks": n_ticks, "participants": n_participants,
                "k": k, "alpha": alpha, "threshold": intervention_threshold,
                "persona_summary_interval": persona_summary_interval,
                "scenario": scenario,
            },
            "ticks": [ticks_data[t] for t in sorted(ticks_data)],
            "interventions": world_state._intervention_log,
            "compliance": world_state._compliance_log,
            "final_scores": {p.agent_id: round(p.behavioral_score, 4) for p in participants},
        }
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved → {output_file}")

    return participants, world_state


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    import datetime as _dt
    _default_output = os.path.join(
        _ROOT, "logs",
        f"sim_{_dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )

    parser = argparse.ArgumentParser(description="dualmirakl simulation v3")
    parser.add_argument("--ticks",        type=int,   default=12)
    parser.add_argument("--participants", type=int,   default=8)
    parser.add_argument("--k",           type=int,   default=2,   help="Observer frequency (ticks)")
    parser.add_argument("--alpha",       type=float, default=0.2, help="EMA alpha")
    parser.add_argument("--max-tokens",  type=int,   default=512)
    parser.add_argument("--threshold",   type=float, default=INTERVENTION_THRESHOLD,
                        help="Intervention cosine similarity threshold (SA target)")
    parser.add_argument("--scenario",    type=str,   default="",
                        help="Domain/context seed for the environment agent")
    parser.add_argument("--output",      type=str,   default=_default_output,
                        help="Path to save JSON results (default: logs/sim_<timestamp>.json)")
    args = parser.parse_args()

    async def main():
        try:
            await run_simulation_v3(
                n_ticks=args.ticks,
                n_participants=args.participants,
                k=args.k,
                alpha=args.alpha,
                max_tokens=args.max_tokens,
                intervention_threshold=args.threshold,
                scenario=args.scenario,
                output_file=args.output,
            )
        finally:
            await close_client()

    asyncio.run(main())
