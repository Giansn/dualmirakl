"""
GPU Harmony v2 — phase-overlapped dual-GPU execution.

Key insight: Phase C (embedding) is CPU-only. During Phase C, both GPUs
are idle. We fill this gap by generating T(n+1) stimuli on authority
while CPU embeds T(n) responses.

Four concurrent workers connected by async queues:

  ┌──────────────────┐     stim_q      ┌──────────────────┐
  │  Stimulus Worker  │ ──────────────► │   Swarm Worker    │
  │  (GPU 0: Phase A) │                 │  (GPU 0+1: B)     │
  └──────────────────┘                  └────────┬─────────┘
                                                  │ resp_q
  ┌──────────────────┐                           ▼
  │  Score Worker     │ ◄────────────────────────┘
  │  (CPU: Phase C)   │ ────► score_q
  └──────────────────┘              │
  ┌──────────────────┐              ▼
  │  Observer Worker  │ ◄───────────┘
  │  (GPU 0: Phase D) │
  └──────────────────┘

Timeline (12-tick, K=4):

  Tick:    T1          T2          T3          T4(obs)     T5
  GPU 0: [A1][B1×N/2]  [A2][B2×N/2]  [A3][B3×N/2]  [A4][B4×N/2][D4]  [A5]...
  GPU 1:     [B1×N/2]      [B2×N/2]      [B3×N/2]      [B4×N/2]      [B5]...
  CPU:            [C1]          [C2]          [C3]          [C4]

  A(n+1) starts as soon as B(n) responses are queued → fills the Phase C gap.
  On observer ticks, D runs after C completes (authority handles observer_a+b).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class GPUHarmony:
    """
    Phase-overlapped coordinator: stimulus prefetch fills the CPU-embedding gap.

    Stimulus worker (GPU 0/authority): generates stimuli, runs 1-2 ticks ahead.
    Swarm worker (GPU 0+1): processes participant responses on both GPUs.
    Score worker (CPU): embedding + score update, signals observer when done.
    Observer worker (GPU 0): observer_a + observer_b on observer ticks.

    The stimulus worker generates T(n+1) stimuli during T(n)'s Phase C
    (CPU embedding), keeping authority busy when it would otherwise be idle.
    """

    def __init__(
        self,
        environment,
        participants: list,
        observers: list,
        world_state,
        *,
        alpha: float = 0.15,
        max_tokens: int = 256,
        score_mode: str = "ema",
        logistic_k: float = 6.0,
        flame_engine=None,
        flame_bridge=None,
        topology_manager=None,
        topology_configs=None,
        bandit=None,
    ):
        self.env = environment
        self.participants = participants
        self.observers = observers
        self.ws = world_state
        self.alpha = alpha
        self.max_tokens = max_tokens
        self.score_mode = score_mode
        self.logistic_k = logistic_k
        self.bandit = bandit

        # Pipeline queues (maxsize=2 allows 2-tick lookahead)
        self._stim_q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._resp_q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._score_q: asyncio.Queue = asyncio.Queue(maxsize=2)

        # Completed tick results for caller
        self._tick_done_q: asyncio.Queue = asyncio.Queue()

    async def run(self, n_ticks: int):
        """Launch all workers concurrently — pipeline starts immediately."""
        t0 = time.monotonic()
        await asyncio.gather(
            self._stimulus_worker(n_ticks),
            self._swarm_worker(n_ticks),
            self._score_worker(n_ticks),
            self._observer_worker(n_ticks),
        )
        elapsed = time.monotonic() - t0
        logger.info("GPU Harmony v2: %d ticks in %.1fs (%.1fs/tick)", n_ticks, elapsed, elapsed / n_ticks)

    # ── Worker 1: Stimulus generation (authority GPU, runs ahead) ──────

    async def _stimulus_worker(self, n_ticks: int):
        """
        Generates stimuli on authority and pushes to stim_q.
        Runs AHEAD: produces T(n+1) stimuli during T(n)'s Phase C (CPU).
        Backpressure: blocks on stim_q.put if 2 stimuli are already queued.
        """
        from simulation.sim_loop import check_compliance, STIMULUS, COMPLIANCE

        for tick in range(1, n_ticks + 1):
            # Pre-phase: persona summaries (concurrent, lightweight)
            await asyncio.gather(*[
                p._maybe_update_persona_summary(tick) for p in self.participants
            ])

            # Phase A: batch stimuli (authority GPU)
            stimuli = await self.env.batch_decide(
                self.participants, self.ws,
                max_tokens=max(64, self.max_tokens // 2),
            )

            # Emit + compliance
            for p in self.participants:
                self.ws.stream.emit(tick, "A", STIMULUS, p.agent_id, {
                    "content": stimuli.get(p.agent_id, ""),
                })
                _sv = stimuli.get(p.agent_id, "")
                if not isinstance(_sv, str):
                    _sv = str(_sv)
                violations = check_compliance(_sv, "environment")
                if violations:
                    self.ws._compliance_log.append({
                        "tick": tick, "agent": "environment",
                        "role": "environment", "violations": violations,
                    })

            await self._stim_q.put((tick, stimuli))
            logger.debug(f"[Tick {tick}] Phase A done → stimuli queued")

    # ── Worker 2: Participant responses (both GPUs) ────────────────────

    async def _swarm_worker(self, n_ticks: int):
        """
        Consumes stimuli, runs all participants concurrently (split across
        both GPUs via backend_override on each ParticipantAgent).
        """
        for _ in range(n_ticks):
            tick, stimuli = await self._stim_q.get()

            # Bandit selection
            _bandit_sel = {}
            if self.bandit is not None:
                for p in self.participants:
                    if p.agent_id in getattr(self.bandit, 'agents', {}):
                        strategy, _ = self.bandit.select(p.agent_id)
                        _bandit_sel[p.agent_id] = strategy
                    else:
                        _bandit_sel[p.agent_id] = ""

            # Phase B: concurrent responses (agents split across both GPUs)
            responses = await asyncio.gather(*[
                p.step(tick, stimuli[p.agent_id], self.ws,
                       max_tokens=self.max_tokens,
                       strategy_constraint=_bandit_sel.get(p.agent_id, ""))
                for p in self.participants
            ])

            await self._resp_q.put((tick, stimuli, responses))
            logger.debug(f"[Tick {tick}] Phase B done → responses queued")

    # ── Worker 3: Scoring (CPU — embedding + score update) ─────────────

    async def _score_worker(self, n_ticks: int):
        """
        Phase C: CPU-only embedding + score update.
        While this runs on CPU, the stimulus worker can generate T(n+1)
        stimuli on authority GPU — that's the key overlap.
        """
        from simulation.sim_loop import (
            embed_score_batch, update_score,
            extract_narrative, ObsEntry,
            RESPONSE, SCORE,
        )

        ws = self.ws
        participants = self.participants

        for _ in range(n_ticks):
            tick, stimuli, responses = await self._resp_q.get()
            ivs_before = len(ws.active_interventions)
            is_observer_tick = (tick % ws.k == 0)

            # Emit responses
            for p, resp in zip(participants, responses):
                payload = {"content": resp}
                if hasattr(p, '_last_parsed') and p._last_parsed:
                    payload["structured"] = p._last_parsed
                if hasattr(p, '_last_prompt_hash') and p._last_prompt_hash:
                    payload["prompt_hash"] = p._last_prompt_hash
                if hasattr(p, '_batched_from') and p._batched_from:
                    payload["batched_from"] = p._batched_from
                ws.stream.emit(tick, "B", RESPONSE, p.agent_id, payload)

            # Phase C: embedding (CPU) — no GPU needed
            dampening = ws.score_dampening()
            texts = [
                extract_narrative(getattr(p, '_last_parsed', None), r)
                for p, r in zip(participants, responses)
            ]
            parsed = [getattr(p, '_last_parsed', None) for p in participants]
            signals_and_ses = embed_score_batch(texts, parsed_actions=parsed)

            # Score update
            for participant, response, (signal, signal_se) in zip(participants, responses, signals_and_ses):
                score_before = participant.behavioral_score
                participant.behavioral_score = update_score(
                    score_before, signal, dampening, self.alpha,
                    mode=self.score_mode, logistic_k=self.logistic_k,
                    susceptibility=participant.susceptibility,
                    resilience=participant.resilience,
                )
                participant.score_log.append(participant.behavioral_score)
                ws.log(ObsEntry(
                    tick=tick, participant_id=participant.agent_id,
                    score_before=score_before, score_after=participant.behavioral_score,
                    stimulus=stimuli[participant.agent_id], response=response,
                    signal=signal, signal_se=signal_se,
                ))
                ws.stream.emit(tick, "C", SCORE, participant.agent_id, {
                    "score_before": score_before, "score_after": participant.behavioral_score,
                    "signal": signal, "signal_se": signal_se, "dampening": dampening,
                })

            # Bandit reward
            if self.bandit is not None:
                for p in participants:
                    if p.agent_id not in getattr(self.bandit, 'agents', {}):
                        continue
                    last_entry = ws._log[-1] if ws._log else None
                    reward = last_entry.signal if last_entry and last_entry.participant_id == p.agent_id else 0.5
                    self.bandit.update(p.agent_id, "", reward)

            # Pass to observer worker (or signal tick done directly)
            await self._score_q.put((tick, stimuli, responses, signals_and_ses, is_observer_tick, ivs_before))

    # ── Worker 4: Observers (authority GPU, observer ticks only) ───────

    async def _observer_worker(self, n_ticks: int):
        """
        Phase D: observer_a analysis + observer_b intervention.
        Only does GPU work on observer ticks (every K ticks).
        On non-observer ticks, just signals tick completion.
        """
        from simulation.sim_loop import OBSERVATION, EV_INTERVENTION

        ws = self.ws
        participants = self.participants
        observers = self.observers
        n = len(participants)

        for _ in range(n_ticks):
            tick, stimuli, responses, signals_and_ses, is_observer_tick, ivs_before = await self._score_q.get()

            if is_observer_tick:
                obs_a, obs_b = observers[0], observers[1]
                if hasattr(obs_a, 'set_participants'):
                    obs_a.set_participants(participants)

                # Phase D1: observer_a analysis (authority GPU)
                analysis = await obs_a.analyse(tick, ws, n)

                obs_payload = {"content": analysis}
                if hasattr(obs_a, '_last_parsed') and obs_a._last_parsed:
                    obs_payload["structured"] = obs_a._last_parsed
                ws.stream.emit(tick, "D", OBSERVATION, "observer_a", obs_payload)

                # Phase D2: observer_b intervention (authority GPU)
                ivs = await obs_b.intervene(tick, ws, n, analysis)
                ws.active_interventions.extend(ivs)

                for iv in ivs:
                    ws.stream.emit(tick, "D", EV_INTERVENTION, "observer_b", {
                        "type": iv.type, "description": iv.description,
                        "modifier": iv.modifier, "activated_at": iv.activated_at,
                        "duration": iv.duration, "source": iv.source,
                    })

            # Signal tick complete to caller
            await self._tick_done_q.put((tick, participants, ivs_before))
            logger.debug(f"[Tick {tick}] {'Phase C+D' if is_observer_tick else 'Phase C'} done → tick complete")
