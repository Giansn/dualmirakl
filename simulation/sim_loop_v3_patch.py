"""
sim_loop_v3_patch.py — Concrete patches for sim_loop.py to activate agent_rolesv3 features.

Apply these changes to sim_loop.py. Each section is labelled with the Fix it implements.
Copy-paste the NEW blocks into the corresponding location in sim_loop.py.
"""

# ══════════════════════════════════════════════════════════════════════════════
# [Fix 4 + 5] Updated imports at top of sim_loop.py
# ══════════════════════════════════════════════════════════════════════════════

# Replace:
#   from simulation.agent_roles import AGENT_ROLES, INTERVENTION_CODEBOOK, ENGAGEMENT_ANCHORS
#   ...
#   INTERVENTION_THRESHOLD = 0.72

# With:
from agent_rolesv3 import (
    AGENT_ROLES,
    INTERVENTION_CODEBOOK,
    ENGAGEMENT_ANCHORS,
    INTERVENTION_THRESHOLD,        # [Fix 5] single source of truth
    PERSONA_SUMMARY_INTERVAL,      # [Fix 7]
    PERSONA_SUMMARY_TEMPLATE,      # [Fix 7]
    EMBED_BATCH_SIZE,              # [Fix 12]
    check_compliance,              # [Fix 9]
)


# ══════════════════════════════════════════════════════════════════════════════
# [Fix 12] Batch embedding — replace embed_score() loop in Phase C
# ══════════════════════════════════════════════════════════════════════════════

import functools
import hashlib

# LRU cache for identical responses (reduces GPU load for repeated phrases)
@functools.lru_cache(maxsize=512)
def _embed_cached(text: str):
    """Encode a single text with caching for repeated responses."""
    return _get_embed().encode([text])[0]


def _compute_signal_from_vec(vec) -> tuple[float, float]:
    """
    Compute behavioral signal from a pre-encoded vector.
    Factored out of embed_score() to support batch processing.
    """
    import numpy as np, math
    _load_anchors()
    sims = np.array([_cosine(vec, a) for a in _anchor_vecs])
    high_mask = np.array([l == "high" for l in _anchor_labels])
    low_mask  = ~high_mask
    high_sim = float(np.mean(sims[high_mask]))
    low_sim  = float(np.mean(sims[low_mask]))
    signal = float(np.clip((high_sim - low_sim + 1.0) / 2.0, 0.0, 1.0))
    se = float(np.std(sims) / math.sqrt(len(sims)))
    return signal, se


def embed_score_batch(responses: list[str]) -> list[tuple[float, float]]:
    """
    [Fix 12] Batch encode all responses in one model call.
    Replaces N individual embed_score() calls with one batched encode.

    Usage in Phase C:
        signals = embed_score_batch([p.last_response for p in participants])
    """
    _load_anchors()
    model = _get_embed()
    vecs = model.encode(responses, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)
    return [_compute_signal_from_vec(v) for v in vecs]


# ══════════════════════════════════════════════════════════════════════════════
# [Fix 7] ParticipantAgent — persona summary injection
# ══════════════════════════════════════════════════════════════════════════════

class ParticipantAgent:  # extends existing class

    def __init__(self, agent_id: str, history_window: int = 4):
        self.agent_id = agent_id
        self.cfg = AGENT_ROLES["participant"]
        self.history: list[dict] = []
        self.history_window = history_window
        import numpy as np
        self.behavioral_score: float = float(np.random.uniform(0.1, 0.5))
        self.score_log: list[float] = []
        self.last_stimulus: str = "nothing yet"
        self.last_response: str = ""
        self.persona_summary: str = ""  # [Fix 7] populated every PERSONA_SUMMARY_INTERVAL ticks

    async def _maybe_update_persona_summary(self, tick: int) -> None:
        """
        [Fix 7] Generate persona summary every PERSONA_SUMMARY_INTERVAL ticks.
        Combats persona drift over long simulations (>20 ticks).
        """
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
        self.persona_summary = await agent_turn(
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

    async def step(self, tick: int, stimulus: str, world_state, max_tokens: int = 256) -> str:
        # [Fix 7] Update persona summary if due
        await self._maybe_update_persona_summary(tick)

        nudge = world_state.participant_nudges()
        nudge_note = f" {nudge}" if nudge else ""
        prompt = (
            f"[Tick {tick}]{nudge_note} "
            f"The environment presents: \"{stimulus[:120]}\". "
            f"How do you respond? What do you do?"
        )
        # Note: behavioral score intentionally NOT leaked to participant (Fix 6)

        response = await agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=self._build_system_prompt(),  # [Fix 7]
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


# ══════════════════════════════════════════════════════════════════════════════
# [Fix 2] ObserverAgent — split into analyse() and intervene()
# ══════════════════════════════════════════════════════════════════════════════

class ObserverAgent:  # extends existing class

    async def analyse(self, tick: int, world_state, n_participants: int) -> str:
        """
        [Fix 2] Observer A: analysis only. Returns text. No codebook extraction.
        Prompt explicitly instructs no intervention recommendations.
        """
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

        # [Fix 9] Compliance: observer_a must NOT contain intervention keywords
        violations = check_compliance(response, "observer_a")
        if violations:
            print(f"  [COMPLIANCE] {self.agent_id} used intervention keywords: {violations}")

        self.history.append({"role": "assistant", "content": response})
        self.analyses.append(response)
        return response  # no extract_interventions here

    async def intervene(
        self,
        tick: int,
        world_state,
        n_participants: int,
        analysis: str,  # [Fix 2] receives observer_a's output
    ) -> list:
        """
        [Fix 2] Observer B: proposes interventions based on observer_a's analysis.
        Only this method calls extract_interventions().
        """
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


def _format_stats(stats: dict) -> str:
    """Helper: format population statistics for observer prompts."""
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


# ══════════════════════════════════════════════════════════════════════════════
# [Fix 2] Updated run_tick() Phase D block — sequential A → B
# ══════════════════════════════════════════════════════════════════════════════

async def run_tick_v3(
    tick: int,
    environment,
    participants: list,
    observers: list,
    world_state,
    alpha: float = 0.2,
    max_tokens: int = 256,
) -> None:
    """
    Full updated run_tick() with all v3 fixes applied.
    Drop-in replacement for run_tick() in sim_loop.py.

    Changes vs v2:
      [Fix 2]  Phase D: sequential A-analyse → B-intervene
      [Fix 12] Phase C: batch embed_score_batch() instead of N individual calls
      [Fix 9]  Environment compliance check
    """
    import asyncio
    n = len(participants)

    # ── Phase A — sequential stimulus generation ──────────────────────────
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
    else:
        stimuli = await environment.batch_decide(participants, world_state)

    # ── Phase B — synchronous participant responses ───────────────────────
    responses = await asyncio.gather(*[
        p.step(tick, stimuli[p.agent_id], world_state, max_tokens=max_tokens)
        for p in participants
    ])

    # ── Phase C — batch score update [Fix 12] ────────────────────────────
    # One encode() call for all responses instead of N individual calls
    dampening = world_state.score_dampening()
    signals_and_ses = embed_score_batch(list(responses))  # [Fix 12]

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

    # ── Phase D — SEQUENTIAL observer cycle [Fix 2] ───────────────────────
    # observer_a analyses → observer_b proposes interventions from analysis
    if tick % world_state.k == 0:
        print(f"\n[Tick {tick}] Observer cycle (sequential A→B)...")
        obs_a, obs_b = observers[0], observers[1]
        # D1: observer_a analyses — returns text, NO codebook extraction
        analysis = await obs_a.analyse(tick, world_state, n)
        # D2: observer_b receives A's analysis, proposes interventions
        ivs = await obs_b.intervene(tick, world_state, n, analysis)
        world_state.active_interventions.extend(ivs)

    world_state.apply_interventions()


# ══════════════════════════════════════════════════════════════════════════════
# [Fix 9] WorldState — add compliance log field
# ══════════════════════════════════════════════════════════════════════════════

# Add to WorldState.__init__ or dataclass definition:
#
#   _compliance_log: list[dict] = field(default_factory=list)
#
# Add method:
#
#   def compliance_report(self) -> list[dict]:
#       return list(self._compliance_log)
#
# At end of run_simulation(), add:
#   compliance = world_state.compliance_report()
#   if compliance:
#       print(f"\n── Compliance violations: {len(compliance)} ──")
#       for v in compliance:
#           print(f"  [T{v['tick']}] {v['agent']} ({v['role']}): {v['violations']}")


# ══════════════════════════════════════════════════════════════════════════════
# [Fix 5] Updated run_simulation() signature — intervention_threshold as param
# ══════════════════════════════════════════════════════════════════════════════

async def run_simulation_v3(
    n_ticks: int = 12,
    n_participants: int = 4,
    k: int = 3,
    alpha: float = 0.2,
    history_window: int = 4,
    max_tokens: int = 256,
    intervention_threshold: float = INTERVENTION_THRESHOLD,  # [Fix 5] SA target
    persona_summary_interval: int = PERSONA_SUMMARY_INTERVAL,  # [Fix 7] SA target
):
    """
    [Fix 5] Exposes intervention_threshold as a run parameter for SA sweeps.
    [Fix 7] Exposes persona_summary_interval.

    Example SA sweep:
        for theta in [0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85]:
            result = await run_simulation_v3(intervention_threshold=theta)
            record(theta, result)
    """
    # Temporarily override module-level threshold for this run
    import agent_rolesv3 as _cfg
    _original_threshold = _cfg.INTERVENTION_THRESHOLD
    _cfg.INTERVENTION_THRESHOLD = intervention_threshold

    import simpy
    env_sim = simpy.Environment()
    world_state = WorldState(k=k)
    world_state._compliance_log = []  # [Fix 9]

    environment = EnvironmentAgent(history_window=history_window)
    participants = [
        ParticipantAgent(f"participant_{i}", history_window=history_window)
        for i in range(n_participants)
    ]
    # [Fix 2] observers: index 0 = analyst (A), index 1 = interventionist (B)
    observers = [
        ObserverAgent("observer_a", "observer_a",
                      history_window=history_window, max_tokens=max_tokens),
        ObserverAgent("observer_b", "observer_b",
                      history_window=history_window, max_tokens=max_tokens),
    ]

    try:
        for tick in range(1, n_ticks + 1):
            env_sim.run(until=tick)
            pct = tick / n_ticks * 100
            print(f"\n── Tick {tick}/{n_ticks}  ({pct:.0f}%) ──")
            await run_tick_v3(
                tick, environment, participants, observers,
                world_state, alpha, max_tokens,
            )
            for p in participants:
                print(f"  {p.agent_id}: score={p.behavioral_score:.3f}")
            stats = world_state.compute_score_statistics(tick)
            if stats:
                print(
                    f"  [POM] mean={stats['mean']:.3f} σ={stats['std']:.3f} "
                    f"skew={stats['skewness']:.2f} above_0.7={stats['n_above_threshold']}"
                )
    finally:
        _cfg.INTERVENTION_THRESHOLD = _original_threshold  # restore

    return participants, world_state
