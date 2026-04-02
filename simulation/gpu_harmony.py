"""
GPU Harmony v6 — three-worker pipeline with warm-start.

Key changes from v5:
  - Three-worker topology: stimulus -> respond -> score (decoupled)
  - Respond worker pushes (participant, response) to resp_q without waiting
    for scoring -- GPU never idles while CPU scores
  - Persona summaries overlap with Phase A (fire-and-await pattern)
  - Warm-start: tick 1 stimulus pre-generated before workers launch,
    so GPU 1 has work from the very first moment
  - Score worker fires FLAME/observer/rebalance after all tick responses scored
  - ~88-92% GPU utilization (up from ~80% in v5)

Three workers connected by two queues, FLAME + observer as background tasks:

  +-----------------+  stim_q(4)  +-----------------+  resp_q  +----------------+
  | Stimulus Worker | ----------> | Respond Worker  | -------> | Score Worker   |
  | (GPU 0: Phse A) |  4-tick     | (GPU 0+1: B)    |          | (CPU: Phase C) |
  | runs far ahead  |  lookahead  | fire responses  |          | embed + score  |
  +-----------------+             +-----------------+          +-------+--------+
                                                                       | on TICK_DONE
  +-----------------+  fire-and-forget                                 |
  | FLAME Task      | <-----------------------------------------------+
  | (GPU 2: Phse F) |                                  +-----------------+
  +-----------------+                                  | Observer Task   |
                                                       | (GPU 0: Phse D) |
                                                       +-----------------+

  +------------------------------------------------------+
  | Adaptive Balancer (runs between ticks)                |
  | sample GPUs -> compute imbalance -> shift participants|
  +------------------------------------------------------+

Timeline (v6 three-worker + warm-start, K=4):

  Tick:    T1(warm)          T2                T3                T4(obs)
  GPU 0: [A1*][B*N/2.....]  [A2][B*N/2.....]  [A3][B*N/2.....]  [A4][B*N/2.....][D4]
  GPU 1:  [B*N/2.....]          [B*N/2.....]       [B*N/2.....]       [B*N/2.....]
  CPU:    [s1][s2]..[sN]        [s1][s2]..[sN]     [s1][s2]..[sN]     [s1][s2]..[sN]
  BAL:                [b]                [b]                [b]                [b]

  * = warm-start: tick 1 stimulus pre-generated before workers launch.
  GPU 1 has work from moment zero -- no first-tick stall.
  Respond worker pushes to resp_q without waiting for CPU scoring.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


class GPUHarmony:
    """
    Three-worker pipeline coordinator with adaptive GPU load balancing.

    v6: respond and score are decoupled workers connected by resp_q.
    GPU never waits for CPU scoring. Persona summaries overlap Phase A.
    Warm-start pre-generates tick 1 stimulus before workers launch.

    Stimulus worker (GPU 0/authority): generates stimuli, runs 1-2 ticks ahead.
    Respond worker (GPU 0+1): launches responses, pushes to resp_q as they arrive.
    Score worker (CPU): pulls from resp_q, embeds + scores, fires FLAME/observer.
    Balancer: samples GPU utilization between ticks, shifts participants.
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
        self._flame_engine = flame_engine
        self._flame_bridge = flame_bridge

        # Trajectory forecaster (predictive observer context)
        try:
            from simulation.forecaster import TrajectoryForecaster
            self._forecaster = TrajectoryForecaster(horizon=4, min_history=6)
        except Exception:
            self._forecaster = None

        # Pipeline queues
        self._stim_q: asyncio.Queue = asyncio.Queue(maxsize=4)  # stimulus -> respond
        self._resp_q: asyncio.Queue = asyncio.Queue()            # respond -> score (unbounded)

        # Completed tick results for caller
        self._tick_done_q: asyncio.Queue = asyncio.Queue()

        # FLAME serialization lock — prevents overlapping steps from racing
        self._flame_lock: asyncio.Lock = asyncio.Lock()

        # Intervention lock — prevents observer extend racing with score_dampening reads
        self._intervention_lock: asyncio.Lock = asyncio.Lock()

        # ── Adaptive GPU balancing (v3) ──────────────────────────────
        self._monitor = None
        self._balancer = None
        _adaptive = os.getenv("SIM_ADAPTIVE_GPU", "1") == "1"
        if _adaptive:
            try:
                from simulation.gpu_monitor import GPUMonitor
                from simulation.adaptive_balancer import AdaptiveBalancer

                gpu_ids = [
                    int(os.getenv("AUTHORITY_GPU", "0")),
                    int(os.getenv("SWARM_GPU", "1")),
                ]
                deadband = float(os.getenv("SIM_BALANCE_DEADBAND", "0.10"))
                max_seqs = int(os.getenv("VLLM_MAX_NUM_SEQS", os.getenv("MAX_NUM_SEQS", "12")))

                # Power target: absolute override or TDP fraction
                explicit_w = os.getenv("SIM_TARGET_POWER_W")
                target_frac = float(os.getenv("SIM_TARGET_POWER_FRAC", "0.65"))

                self._monitor = GPUMonitor(gpu_ids=gpu_ids)
                if self._monitor.start():
                    # Compute target from TDP if no explicit wattage set
                    if explicit_w:
                        target_w = float(explicit_w)
                    elif hasattr(self._monitor, 'min_power_limit') and self._monitor.min_power_limit:
                        target_w = target_frac * self._monitor.min_power_limit
                    else:
                        target_w = 195.0  # fallback

                    self._balancer = AdaptiveBalancer(
                        self._monitor,
                        n_participants=len(participants),
                        target_power_w=target_w,
                        deadband=deadband,
                        max_seqs=max_seqs,
                    )
                    # Apply initial split
                    self._balancer.apply_to_participants(participants)
                    logger.info("GPU Harmony v6: adaptive balancing ACTIVE (target=%.0fW)", target_w)
                else:
                    logger.info("GPU Harmony v6: monitor init failed — static split")
                    self._monitor = None
            except Exception as e:
                logger.info("GPU Harmony v6: adaptive unavailable (%s) — static split", e)

    async def run(self, n_ticks: int):
        """Launch workers concurrently — three-worker pipeline with warm-start."""
        t0 = time.monotonic()

        # Warm-start: pre-generate tick 1 stimulus so respond worker has work
        # immediately when it launches — eliminates first-tick GPU 1 stall.
        # Persona summaries on tick 1 are overlapped (typically no-op: no history).
        if n_ticks > 0:
            persona_tasks = [
                asyncio.create_task(p._maybe_update_persona_summary(1))
                for p in self.participants
            ]
            stim_data = await self._generate_stimulus(1)
            await asyncio.gather(*persona_tasks)
            await self._stim_q.put(stim_data)
            logger.debug("[Warm-start] Tick 1 stimulus pre-queued")

        try:
            workers = [
                asyncio.create_task(
                    self._stimulus_worker(n_ticks, start_tick=2),
                    name="stimulus",
                ),
                asyncio.create_task(
                    self._respond_worker(n_ticks),
                    name="respond",
                ),
                asyncio.create_task(
                    self._score_worker(n_ticks),
                    name="score",
                ),
            ]
            done, pending = await asyncio.wait(workers, return_when=asyncio.FIRST_EXCEPTION)

            failed = False
            for task in done:
                if task.exception():
                    logger.error("Pipeline worker '%s' failed: %s",
                                 task.get_name(), task.exception())
                    failed = True

            if failed:
                for task in pending:
                    task.cancel()
                # Unblock any worker waiting on a queue
                try:
                    self._stim_q.put_nowait(None)
                except asyncio.QueueFull:
                    pass
                self._resp_q.put_nowait(None)
                err = next((t.exception() for t in done if t.exception()), RuntimeError("pipeline failed"))
                await self._tick_done_q.put(("PIPELINE_ERROR", err))
                for task in pending:
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
            elif pending:
                await asyncio.gather(*pending)
        finally:
            if self._monitor:
                self._monitor.stop()
        elapsed = time.monotonic() - t0
        adj = self._balancer.state.adjustments if self._balancer else 0
        logger.info(
            "GPU Harmony v6: %d ticks in %.1fs (%.1fs/tick, %d rebalances)",
            n_ticks, elapsed, elapsed / max(n_ticks, 1), adj,
        )

    # ── Stimulus generation helper (shared by warm-start + worker) ────

    async def _generate_stimulus(self, tick: int):
        """Generate stimulus for a single tick. Returns (tick, stimuli, dampening)."""
        from simulation.sim_loop import check_compliance, STIMULUS, COMPLIANCE

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

        dampening_snapshot = self.ws.score_dampening()
        logger.debug("[Tick %d] Phase A done (d=%.2f)", tick, dampening_snapshot)
        return tick, stimuli, dampening_snapshot

    # ── Worker 1: Stimulus generation (authority GPU, runs ahead) ──────

    async def _stimulus_worker(self, n_ticks: int, start_tick: int = 1):
        """
        Generates stimuli on authority and pushes to stim_q.
        Runs AHEAD: produces T(n+1) stimuli during T(n)'s Phase B/C.
        Backpressure: blocks on stim_q.put if 4 stimuli are already queued.

        Persona summaries overlap with Phase A: fired as background tasks,
        awaited only before stim_q push (they must complete before Phase B).
        """
        for tick in range(start_tick, n_ticks + 1):
            # Pre-phase: persona summaries — fire as background tasks,
            # overlap with Phase A stimulus generation (Rec 2)
            persona_tasks = [
                asyncio.create_task(p._maybe_update_persona_summary(tick))
                for p in self.participants
            ]

            # Phase A runs concurrently with persona summaries
            stim_data = await self._generate_stimulus(tick)

            # Ensure persona summaries complete before pushing to stim_q —
            # they must finish before Phase B uses the updated summaries
            await asyncio.gather(*persona_tasks)
            await self._stim_q.put(stim_data)

    # ── Worker 2: Response dispatch (GPUs, pushes to resp_q) ───────────

    async def _respond_worker(self, n_ticks: int):
        """
        Launches all participant responses for each tick and pushes
        (participant, response) to resp_q as they arrive via as_completed.
        Does NOT wait for scoring — immediately moves to next tick.

        Sends a TICK_DONE sentinel after all responses for a tick are pushed,
        so the score worker knows when to fire FLAME/observer/rebalance.
        """
        ws = self.ws
        participants = self.participants

        for _ in range(n_ticks):
            item = await self._stim_q.get()
            if item is None:
                break  # poison pill from error handler
            tick, stimuli, dampening = item
            ivs_before = len(ws.active_interventions)
            is_observer_tick = (tick % ws.k == 0)

            # Bandit selection
            _bandit_sel = {}
            if self.bandit is not None:
                for p in participants:
                    if p.agent_id in getattr(self.bandit, 'agents', {}):
                        strategy, _ = self.bandit.select(p.agent_id)
                        _bandit_sel[p.agent_id] = strategy
                    else:
                        _bandit_sel[p.agent_id] = ""

            # Launch all responses concurrently
            async def _respond_one(p):
                """Single participant: respond -> return (participant, response)."""
                resp = await p.step(
                    tick, stimuli[p.agent_id], ws,
                    max_tokens=self.max_tokens,
                    strategy_constraint=_bandit_sel.get(p.agent_id, ""),
                )
                return p, resp

            tasks = [
                asyncio.create_task(_respond_one(p), name=f"resp_{p.agent_id}")
                for p in participants
            ]

            # Push each response to score worker as it arrives
            for coro in asyncio.as_completed(tasks):
                p, resp = await coro
                await self._resp_q.put(("RESP", tick, p, resp, stimuli, dampening))

            # All responses for this tick dispatched — signal score worker
            await self._resp_q.put(("TICK_DONE", tick, is_observer_tick, ivs_before))

            logger.debug("[Tick %d] all %d responses dispatched to score worker",
                         tick, len(participants))

    # ── Worker 3: Scoring (CPU, pulls from resp_q) ─────────────────────

    async def _score_worker(self, n_ticks: int):
        """
        Pulls responses from resp_q and scores them (embed + update).
        Fires FLAME/observer/rebalance when TICK_DONE sentinel arrives.

        Decoupled from respond worker: GPU can launch next tick's responses
        while CPU is still scoring the current tick.
        """
        from simulation.sim_loop import (
            embed_score_batch, update_score,
            extract_narrative, ObsEntry,
            RESPONSE, SCORE,
        )

        ws = self.ws
        participants = self.participants
        ticks_scored = 0

        while ticks_scored < n_ticks:
            item = await self._resp_q.get()
            if item is None:
                break  # poison pill from error handler

            tag = item[0]

            if tag == "RESP":
                _, tick, p, resp, stimuli, dampening = item

                # Emit response event
                payload = {"content": resp}
                if hasattr(p, '_last_parsed') and p._last_parsed:
                    payload["structured"] = p._last_parsed
                if hasattr(p, '_last_prompt_hash') and p._last_prompt_hash:
                    payload["prompt_hash"] = p._last_prompt_hash
                if hasattr(p, '_batched_from') and p._batched_from:
                    payload["batched_from"] = p._batched_from
                ws.stream.emit(tick, "B", RESPONSE, p.agent_id, payload)

                # Phase C: embed + score this response (CPU)
                text = extract_narrative(getattr(p, '_last_parsed', None), resp)
                parsed_action = getattr(p, '_last_parsed', None)
                [(signal, signal_se)] = embed_score_batch(
                    [text], parsed_actions=[parsed_action],
                )

                score_before = p.behavioral_score
                p.behavioral_score = update_score(
                    score_before, signal, dampening, self.alpha,
                    mode=self.score_mode, logistic_k=self.logistic_k,
                    susceptibility=p.susceptibility,
                    resilience=p.resilience,
                )
                p.score_log.append(p.behavioral_score)
                ws.log(ObsEntry(
                    tick=tick, participant_id=p.agent_id,
                    score_before=score_before, score_after=p.behavioral_score,
                    stimulus=stimuli[p.agent_id], response=resp,
                    signal=signal, signal_se=signal_se,
                ))
                ws.stream.emit(tick, "C", SCORE, p.agent_id, {
                    "score_before": score_before, "score_after": p.behavioral_score,
                    "signal": signal, "signal_se": signal_se, "dampening": dampening,
                })

                # Bandit reward (immediate)
                if self.bandit is not None and p.agent_id in getattr(self.bandit, 'agents', {}):
                    self.bandit.update(p.agent_id, "", signal)

                # Feed forecaster with new score
                if self._forecaster:
                    self._forecaster.update(p.agent_id, tick, p.behavioral_score)

            elif tag == "TICK_DONE":
                _, tick, is_observer_tick, ivs_before = item

                # All responses for this tick scored — post-tick actions

                # Phase F: FLAME population step (GPU 2, fire-and-forget)
                if self._flame_engine and self._flame_bridge:
                    asyncio.create_task(
                        self._run_flame(tick, [p.behavioral_score for p in participants]),
                        name=f"flame_t{tick}",
                    )

                # Inject forecast context for observer
                if self._forecaster:
                    ws.forecast_context = self._forecaster.get_context_for_observer(tick)

                # Observer: fire-and-forget on K-ticks, immediate signal otherwise
                if is_observer_tick:
                    asyncio.create_task(
                        self._run_observer(tick, ivs_before),
                        name=f"observer_t{tick}",
                    )
                else:
                    self._rebalance_and_signal(tick, ivs_before)

                ticks_scored += 1
                logger.debug("[Tick %d] scored + post-tick dispatched", tick)

    # ── Observer execution (runs concurrently, doesn't block pipeline) ──

    async def _run_observer(self, tick: int, ivs_before: int):
        """
        Phase D: observer_a analysis + observer_b intervention.
        Runs on GPU 0 concurrently with the next tick's stimulus/swarm on GPU 1.
        Does NOT block the pipeline — results apply to world state asynchronously.
        """
        from simulation.sim_loop import OBSERVATION, EV_INTERVENTION

        ws = self.ws
        participants = self.participants
        observers = self.observers
        n = len(participants)

        try:
            obs_a, obs_b = observers[0], observers[1]
            if hasattr(obs_a, 'set_participants'):
                obs_a.set_participants(participants)

            # Inject FLAME population feedback for observer context
            if self._flame_engine and self._flame_bridge:
                try:
                    ws.flame_feedback = self._flame_bridge.get_population_coupling_feedback(
                        self._flame_engine
                    )
                except Exception as e:
                    logger.debug("FLAME feedback extraction failed: %s", e)

            # Phase D1: observer_a analysis (authority GPU)
            analysis = await obs_a.analyse(tick, ws, n)

            obs_payload = {"content": analysis}
            if hasattr(obs_a, '_last_parsed') and obs_a._last_parsed:
                obs_payload["structured"] = obs_a._last_parsed
            ws.stream.emit(tick, "D", OBSERVATION, "observer_a", obs_payload)

            # Phase D2: observer_b intervention (authority GPU)
            ivs = await obs_b.intervene(tick, ws, n, analysis)
            async with self._intervention_lock:
                ws.active_interventions.extend(ivs)

            for iv in ivs:
                ws.stream.emit(tick, "D", EV_INTERVENTION, "observer_b", {
                    "type": iv.type, "description": iv.description,
                    "modifier": iv.modifier, "activated_at": iv.activated_at,
                    "duration": iv.duration, "source": iv.source,
                })
        except Exception as e:
            logger.error("[Tick %d] Observer failed: %s", tick, e)
        finally:
            # Rebalance + signal tick done AFTER observer completes
            self._rebalance_and_signal(tick, ivs_before)

    async def _run_flame(self, tick: int, scores: list[float]):
        """Phase F: FLAME population step on GPU 2 (non-blocking background task).
        Lock prevents overlapping steps from racing on engine state."""
        async with self._flame_lock:
            try:
                self._flame_bridge.push_influencer_scores(self._flame_engine, scores)
                sub_steps = self._flame_engine.config.get("sub_steps", 10)
                await asyncio.get_running_loop().run_in_executor(None, self._flame_engine.step, sub_steps)
                self.ws.flame_snapshot = self._flame_bridge.pull_population_stats(
                    self._flame_engine, tick, sub_steps
                )
            except Exception as e:
                logger.warning("FLAME step failed at tick %d: %s", tick, e)

    def _rebalance_and_signal(self, tick: int, ivs_before: int):
        """Rebalance GPUs and signal tick completion."""
        if self._balancer:
            adjusted = self._balancer.rebalance(tick)
            if adjusted:
                self._balancer.apply_to_participants(self.participants)
                new_env = self._balancer.should_move_env(tick)
                if new_env and hasattr(self.env, 'cfg'):
                    self.env.cfg["backend"] = new_env
                    logger.info("[Tick %d] Environment backend -> %s", tick, new_env)

            try:
                from simulation.sim_loop import SCORE
                report = self._balancer.tick_report()
                self.ws.stream.emit(tick, "BAL", "gpu_balance", "balancer", report)
            except Exception:
                pass

        self._tick_done_q.put_nowait((tick, self.participants, ivs_before))
        logger.debug("[Tick %d] done", tick)
