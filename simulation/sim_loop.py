"""
Multi-agent media addiction simulation — stratified tick architecture.

Tick structure (per simulated hour):
  Phase A — PlatformAgent decides content per user (sequential, closes feedback loop)
  Phase B — MediaUserAgents react to content (concurrent via asyncio.gather)
  Phase C — Score update via gte-small embedding similarity (CPU, deterministic)
  Phase D — ObserverAgents analyse and propose interventions (every K ticks, concurrent)

Inter-agent communication flows:
  platform_ai → media_user  : content_description (what the platform shows)
  media_user  → platform_ai : last_response (engagement signal, feeds next Phase A)
  media_user  → WorldState  : observation log entry
  WorldState  → observers   : K-tick window (aggregate stats when N>15)
  observers   → WorldState  : typed interventions via embedding codebook extraction

Score dynamics: EMA with alpha=0.2 (high inertia, calibrated to PSMU longitudinal
autocorrelation r~0.75-0.85 from Vannucci et al. and Bergen Scale studies).
alpha is a thesis-critical parameter — sweep [0.1, 0.4] in sensitivity analysis.
"""

from __future__ import annotations

import asyncio
import json
import numpy as np
import simpy
from dataclasses import dataclass, field
from typing import Optional

from simulation.agent_roles import AGENT_ROLES, INTERVENTION_CODEBOOK, ENGAGEMENT_ANCHORS
from orchestrator import agent_turn, close_client

# ── Embedding helper (gte-small loaded in gateway; here we load directly for sim) ──
from sentence_transformers import SentenceTransformer

_embed_model: Optional[SentenceTransformer] = None

def _get_embed():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("/per.volume/huggingface/hub/gte-small")
    return _embed_model

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Data structures ────────────────────────────────────────────────────────────────

@dataclass
class Intervention:
    type: str                    # "platform_constraint" | "user_nudge" | "score_modifier"
    description: str             # injected into agent prompts
    modifier: dict               # {"dampening": 0.5} or {"prompt_inject": "..."}
    activated_at: int            # tick number
    duration: int = -1           # ticks remaining; -1 = permanent
    source: str = ""             # which observer proposed it


@dataclass
class ObsEntry:
    hour: int
    user_id: str
    score_before: float
    score_after: float
    platform_content: str
    user_response: str

    def to_str(self) -> str:
        return (
            f"[H{self.hour}] {self.user_id} (score {self.score_before:.2f}→{self.score_after:.2f}) "
            f"saw: \"{self.platform_content[:80]}\" | did: \"{self.user_response[:80]}\""
        )


@dataclass
class WorldState:
    k: int = 3                              # observer fire frequency (configurable)
    _log: list[ObsEntry] = field(default_factory=list)
    active_interventions: list[Intervention] = field(default_factory=list)

    def log(self, entry: ObsEntry) -> None:
        self._log.append(entry)

    def full_log(self) -> list[ObsEntry]:
        """Full granular log — for thesis data export, not LLM prompts."""
        return list(self._log)

    def observer_prompt_window(self, hour: int, n_users: int) -> str:
        """Token-budget-aware summary for observer agents (~2000 token cap)."""
        window = [e for e in self._log if e.hour > hour - self.k]
        if not window:
            return "No observations yet."
        if n_users > 15:
            # Aggregate stats only
            scores_before = [e.score_before for e in window]
            scores_after  = [e.score_after  for e in window]
            crossings     = sum(1 for e in window if e.score_before < 0.7 <= e.score_after)
            return (
                f"Hours {hour - self.k + 1}–{hour} | {n_users} users | "
                f"mean score {np.mean(scores_before):.2f}→{np.mean(scores_after):.2f} | "
                f"users crossing 0.7 threshold: {crossings}"
            )
        return "\n".join(e.to_str() for e in window)

    def apply_interventions(self) -> None:
        """Decrement finite-duration interventions; remove expired ones."""
        still_active = []
        for iv in self.active_interventions:
            if iv.duration == -1:
                still_active.append(iv)
            elif iv.duration > 0:
                iv.duration -= 1
                still_active.append(iv)
        self.active_interventions = still_active

    def platform_constraints(self) -> str:
        parts = [iv.description for iv in self.active_interventions
                 if iv.type == "platform_constraint"]
        return " ".join(parts)

    def user_nudges(self) -> str:
        parts = [iv.description for iv in self.active_interventions
                 if iv.type == "user_nudge"]
        return " ".join(parts)

    def score_dampening(self) -> float:
        d = 1.0
        for iv in self.active_interventions:
            if iv.type == "score_modifier":
                d *= iv.modifier.get("dampening", 1.0)
        return d


# ── Score update ───────────────────────────────────────────────────────────────────

# Pre-encoded at first call
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

def embed_score(response: str) -> float:
    """Return engagement signal in [0,1] via cosine similarity against anchor phrases."""
    _load_anchors()
    vec = _get_embed().encode([response])[0]
    sims = [_cosine(vec, a) for a in _anchor_vecs]
    high_sim = np.mean([s for s, l in zip(sims, _anchor_labels) if l == "high"])
    low_sim  = np.mean([s for s, l in zip(sims, _anchor_labels) if l == "low"])
    # Normalise to [0,1]: 1 = fully high-engagement, 0 = fully low-engagement
    return float(np.clip((high_sim - low_sim + 1.0) / 2.0, 0.0, 1.0))

def update_score(
    current: float,
    signal: float,
    dampening: float = 1.0,
    alpha: float = 0.2,   # EMA responsiveness — thesis sensitivity parameter
) -> float:
    """
    Exponential moving average score update with intervention dampening.
    alpha=0.2 → high inertia, calibrated to PSMU longitudinal autocorrelation.
    Dampening applies to the *change*, not the absolute value.
    """
    base_new = alpha * signal + (1 - alpha) * current
    delta = base_new - current
    return float(np.clip(current + delta * dampening, 0.0, 1.0))


# ── Intervention extraction ────────────────────────────────────────────────────────

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

def extract_interventions(observer_id: str, response: str, hour: int) -> list[Intervention]:
    """
    Embed the observer's response and compare against intervention codebook.
    Returns Intervention objects for any match above threshold.
    Robust to phrasing variation — no structured tag parsing required.
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
            type=iv_type,
            description=description,
            modifier=modifier,
            activated_at=hour,
            source=observer_id,
        ))
        print(f"  [{observer_id}] INTERVENTION triggered: {key} (sim={sim:.2f})")
    return ivs

def _make_intervention(key: str) -> tuple[str, str, dict]:
    defaults = {
        "screen_time_limit":    ("user_nudge",          "You have received a screen time warning.", {}),
        "content_warning":      ("platform_constraint", "You must add content warnings to flagged posts.", {}),
        "cooldown_prompt":      ("user_nudge",          "A cooldown prompt appeared: 'Take a break?'", {}),
        "algorithm_dampening":  ("score_modifier",      "Platform engagement optimisation reduced.",
                                 {"dampening": 0.6}),
    }
    return defaults.get(key, ("platform_constraint", key, {}))


# ── Agent classes ──────────────────────────────────────────────────────────────────

class PlatformAgent:
    """Decides content per user (Phase A). Singleton — one platform serves all users."""

    def __init__(self):
        self.cfg = AGENT_ROLES["platform_ai"]
        self.history: list[dict] = []

    async def decide(self, user: "MediaUserAgent", world_state: WorldState) -> str:
        constraints = world_state.platform_constraints()
        constraint_note = f" ACTIVE CONSTRAINTS: {constraints}" if constraints else ""
        prompt = (
            f"User {user.agent_id}: addiction_score={user.addiction_score:.2f}."
            f"{constraint_note} "
            f"Last hour you showed them: \"{user.last_platform_content[:80]}\". "
            f"They responded: \"{user.last_response[:80]}\". "
            f"Decide what content to show them next. Be specific."
        )
        response = await agent_turn(
            agent_id="platform_ai",
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=prompt,
            history=self.history[-4:],
            max_tokens=128,   # content decisions are short
        )
        self.history.append({"role": "assistant", "content": response})
        return response

    async def batch_decide(self, users: list["MediaUserAgent"], world_state: WorldState) -> dict[str, str]:
        """Stub for N>10 cohort-level decisions. Not yet implemented."""
        raise NotImplementedError("batch_decide: implement cohort-level strategy for N>10")


class MediaUserAgent:
    """Reacts to platform content (Phase B). N instances run concurrently."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.cfg = AGENT_ROLES["media_user"]
        self.history: list[dict] = []
        self.addiction_score: float = float(np.random.uniform(0.1, 0.5))
        self.engagement_log: list[float] = []
        self.last_platform_content: str = "nothing yet"
        self.last_response: str = ""

    async def step(self, hour: int, content: str, world_state: WorldState) -> str:
        nudge = world_state.user_nudges()
        nudge_note = f" {nudge}" if nudge else ""
        prompt = (
            f"[Hour {hour}]{nudge_note} "
            f"The platform just showed you: \"{content[:120]}\". "
            f"Your current engagement score is {self.addiction_score:.2f}/1.0. "
            f"How do you respond? What do you do?"
        )
        response = await agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=prompt,
            history=self.history[-4:],
        )
        self.last_platform_content = content
        self.last_response = response
        self.history.append({"role": "assistant", "content": response})
        return response


class ObserverAgent:
    """Researcher or policy_analyst — fires every K ticks (Phase D)."""

    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.cfg = AGENT_ROLES[role]
        self.history: list[dict] = []
        self.analyses: list[str] = []

    async def observe(self, hour: int, world_state: WorldState, n_users: int) -> list[Intervention]:
        window = world_state.observer_prompt_window(hour, n_users)
        active = ", ".join(iv.description for iv in world_state.active_interventions) or "none"
        prompt = (
            f"[Hour {hour}] Observation window (last {world_state.k} hours):\n"
            f"{window}\n\n"
            f"Active interventions: {active}\n\n"
            f"Analyse the dynamics. Recommend any new interventions if warranted."
        )
        response = await agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=prompt,
            history=self.history[-4:],
            max_tokens=256,
        )
        print(f"[{self.agent_id}] {response[:120]}...")
        self.history.append({"role": "assistant", "content": response})
        self.analyses.append(response)
        return extract_interventions(self.agent_id, response, hour)


# ── Tick orchestration ─────────────────────────────────────────────────────────────

async def run_tick(
    hour: int,
    platform: PlatformAgent,
    users: list[MediaUserAgent],
    observers: list[ObserverAgent],
    world_state: WorldState,
    alpha: float = 0.2,
) -> None:
    n = len(users)

    # Phase A — platform decides per user (sequential; closes user→platform feedback loop)
    if n <= 10:
        contents = {}
        for user in users:
            contents[user.agent_id] = await platform.decide(user, world_state)
    else:
        contents = await platform.batch_decide(users, world_state)

    # Phase B — all users react concurrently
    responses = await asyncio.gather(*[
        u.step(hour, contents[u.agent_id], world_state) for u in users
    ])

    # Phase C — score update + observation log (CPU, no LLM)
    dampening = world_state.score_dampening()
    for user, response in zip(users, responses):
        score_before = user.addiction_score
        signal = embed_score(response)
        user.addiction_score = update_score(score_before, signal, dampening, alpha)
        user.engagement_log.append(user.addiction_score)
        world_state.log(ObsEntry(
            hour=hour,
            user_id=user.agent_id,
            score_before=score_before,
            score_after=user.addiction_score,
            platform_content=contents[user.agent_id],
            user_response=response,
        ))

    # Phase D — observers fire every K ticks (concurrent on GPU0)
    if hour % world_state.k == 0:
        print(f"\n[Hour {hour}] Observer tick...")
        intervention_lists = await asyncio.gather(*[
            o.observe(hour, world_state, n) for o in observers
        ])
        for ivs in intervention_lists:
            world_state.active_interventions.extend(ivs)

    world_state.apply_interventions()


async def run_simulation(
    n_hours: int = 12,
    n_users: int = 4,
    k: int = 3,
    alpha: float = 0.2,
) -> tuple[list[MediaUserAgent], WorldState]:
    """
    Run the stratified multi-agent simulation.

    Args:
        n_hours: Simulated hours to run.
        n_users: Number of media_user agents.
        k:       Observer fire frequency (every k ticks). Thesis sensitivity parameter.
        alpha:   EMA responsiveness for score dynamics. Thesis sensitivity parameter.
                 Calibrated to PSMU longitudinal autocorrelation (sweep [0.1, 0.4]).
    """
    env = simpy.Environment()
    world_state = WorldState(k=k)
    platform = PlatformAgent()
    users = [MediaUserAgent(f"user_{i}") for i in range(n_users)]
    observers = [
        ObserverAgent("researcher",    "researcher"),
        ObserverAgent("policy_analyst","policy_analyst"),
    ]

    for hour in range(1, n_hours + 1):
        env.run(until=hour)
        print(f"\n── Hour {hour} ──")
        await run_tick(hour, platform, users, observers, world_state, alpha)
        for u in users:
            print(f"  {u.agent_id}: score={u.addiction_score:.3f}")

    return users, world_state


if __name__ == "__main__":
    async def main():
        print("Running 12-hour simulation | 4 users | K=3 | alpha=0.2")
        try:
            users, world_state = await run_simulation(n_hours=12, n_users=4)
            print("\n── Final scores ──")
            for u in users:
                print(f"  {u.agent_id}: {u.addiction_score:.3f}")
            print(f"\n── Full log: {len(world_state.full_log())} entries ──")
        finally:
            await close_client()

    asyncio.run(main())
