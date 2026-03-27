"""
GPU Harmony — tick-pipelined dual-GPU execution.

Three concurrent workers connected by async queues:

  ┌──────────────────┐     stim_q      ┌──────────────────┐
  │  Stimulus Worker  │ ──────────────► │   Swarm Worker    │
  │  (GPU 0: Phase A) │                 │  (GPU 1: Phase B) │
  └──────────────────┘                  └────────┬─────────┘
                                                  │ resp_q
  ┌──────────────────┐                           ▼
  │  Observer Worker  │ ◄────────────────────────┘
  │  (GPU 0: Phase D) │
  └──────────────────┘

Timeline with pipeline:
  GPU 0: [T1:A][T2:A*]........[T1:D][T3:A*]........[T2:D]
  GPU 1: ......[T1:B=========].......[T2:B=========]......
  CPU:   ..................[T1:C]...............[T2:C].....

  * = speculative prefetch using pre-tick world state.
  T(n+1):A runs while T(n):B is in progress — both GPUs active.
  T(n):D overlaps with T(n+1):B — both GPUs active again.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class GPUHarmony:
    """
    Tick-pipelined coordinator with three async workers and queue-based IPC.

    Stimulus worker (GPU 0): generates stimuli, runs AHEAD of other phases.
    Swarm worker (GPU 1): processes participant responses.
    Observer worker (GPU 0): scoring + observers, consumes responses.

    The stimulus worker runs 1 tick ahead, so GPU 0 generates T(n+1) stimuli
    while GPU 1 is still processing T(n) participants.
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

        # Queues: stimulus worker → swarm worker → observer worker
        self._stim_q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._resp_q: asyncio.Queue = asyncio.Queue(maxsize=2)

        # Completed tick results for caller
        self._tick_done_q: asyncio.Queue = asyncio.Queue()

    async def run(self, n_ticks: int):
        """Launch all workers concurrently — pipeline starts immediately."""
        t0 = time.monotonic()
        await asyncio.gather(
            self._stimulus_worker(n_ticks),
            self._swarm_worker(n_ticks),
            self._observer_worker(n_ticks),
        )
        elapsed = time.monotonic() - t0
        logger.info("GPU Harmony: %d ticks in %.1fs (%.1fs/tick)", n_ticks, elapsed, elapsed / n_ticks)

    # ── Worker 1: Stimulus generation (Phase A on env backend) ───────────

    async def _stimulus_worker(self, n_ticks: int):
        """
        Generates stimuli and pushes to stim_q.
        Runs AHEAD: produces T(n+1) stimuli while T(n) is still in Phase B.
        """
        from simulation.sim_loop import check_compliance, STIMULUS, COMPLIANCE

        for tick in range(1, n_ticks + 1):
            # Pre-phase: persona summaries
            await asyncio.gather(*[
                p._maybe_update_persona_summary(tick) for p in self.participants
            ])

            # Phase A: batch stimuli
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

            # Push stimuli into pipeline — blocks if swarm is still busy (backpressure)
            await self._stim_q.put((tick, stimuli))
            logger.debug(f"[Tick {tick}] Phase A done → stimuli queued")

    # ── Worker 2: Participant responses (Phase B on swarm + authority) ────

    async def _swarm_worker(self, n_ticks: int):
        """
        Consumes stimuli, runs all participants concurrently (split across GPUs),
        pushes responses to resp_q.
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

    # ── Worker 3: Scoring + Observers (Phase C+D on authority GPU) ───────

    async def _observer_worker(self, n_ticks: int):
        """
        Consumes responses, runs embedding (CPU) + observers (authority GPU).
        While this runs, stimulus_worker is already generating NEXT tick's stimuli
        and swarm_worker will start the next Phase B — that's the pipeline overlap.
        """
        from simulation.sim_loop import (
            embed_score_batch, update_score,
            extract_narrative, ObsEntry,
            RESPONSE, SCORE, OBSERVATION, EV_INTERVENTION,
        )

        ws = self.ws
        participants = self.participants
        observers = self.observers
        n = len(participants)

        for _ in range(n_ticks):
            tick, stimuli, responses = await self._resp_q.get()
            is_observer_tick = (tick % ws.k == 0)
            ivs_before = len(ws.active_interventions)

            # Emit responses
            for p, resp in zip(participants, responses):
                payload = {"content": resp}
                if hasattr(p, '_last_parsed') and p._last_parsed:
                    payload["structured"] = p._last_parsed
                ws.stream.emit(tick, "B", RESPONSE, p.agent_id, payload)

            # Phase C+D
            dampening = ws.score_dampening()

            if is_observer_tick:
                obs_a, obs_b = observers[0], observers[1]
                if hasattr(obs_a, 'set_participants'):
                    obs_a.set_participants(participants)

                async def _embed():
                    texts = [
                        extract_narrative(getattr(p, '_last_parsed', None), r)
                        for p, r in zip(participants, responses)
                    ]
                    parsed = [getattr(p, '_last_parsed', None) for p in participants]
                    return embed_score_batch(texts, parsed_actions=parsed)

                async def _observe_a():
                    return await obs_a.analyse(tick, ws, n)

                (signals_and_ses, analysis) = await asyncio.gather(
                    _embed(), _observe_a()
                )

                obs_payload = {"content": analysis}
                if hasattr(obs_a, '_last_parsed') and obs_a._last_parsed:
                    obs_payload["structured"] = obs_a._last_parsed
                ws.stream.emit(tick, "D", OBSERVATION, "observer_a", obs_payload)

                ivs = await obs_b.intervene(tick, ws, n, analysis)
                ws.active_interventions.extend(ivs)

                for iv in ivs:
                    ws.stream.emit(tick, "D", EV_INTERVENTION, "observer_b", {
                        "type": iv.type, "description": iv.description,
                        "modifier": iv.modifier, "activated_at": iv.activated_at,
                        "duration": iv.duration, "source": iv.source,
                    })
            else:
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

            # Signal tick complete to caller
            await self._tick_done_q.put((tick, participants, ivs_before))
            logger.debug(f"[Tick {tick}] Phase C+D done → tick complete")
