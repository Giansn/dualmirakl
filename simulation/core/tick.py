"""Tick orchestration for the simulation loop."""

from __future__ import annotations

import asyncio
import logging
import numpy as np
from typing import Optional

from simulation.config.legacy_roles import check_compliance
from simulation.core.event_stream import (
    STIMULUS, RESPONSE, SCORE, OBSERVATION, INTERVENTION as EV_INTERVENTION,
    COMPLIANCE, FLAME_SNAPSHOT, GRAPH_UPDATE,
)
from simulation.config.action_schema import extract_narrative
from simulation.signal.computation import embed_score_batch, update_score
from simulation.signal.intervention import Intervention
from simulation.core.topology import TopologyManager, combine_stimuli

from simulation.core.state import WorldState, ObsEntry
from simulation.core.agents_impl import (
    EnvironmentAgent, ParticipantAgent, ObserverAgent,
    resilient_agent_turn,
)

logger = logging.getLogger(__name__)

__all__ = [
    "run_tick",
    "_batch_phase_b",
]

async def _batch_phase_b(
    tick: int,
    participants: list[ParticipantAgent],
    world_state: WorldState,
    stimuli: dict[str, str],
    archetype_groups: dict[str, list[str]],
    min_group_size: int,
    max_tokens: int,
    bandit_selections: dict[str, str],
) -> list[str]:
    """Representative-mode archetype batching for Phase B.

    For each archetype group >= min_group_size: call LLM once via the first
    agent in the group, reuse its response for all others. Scoring diverges
    via per-agent susceptibility/resilience.

    Small groups and ungrouped agents fall back to normal concurrent calls.
    """
    agent_lookup = {p.agent_id: p for p in participants}
    ordered_ids = [p.agent_id for p in participants]
    responses_map: dict[str, str] = {}
    batched_from_map: dict[str, str] = {}

    # Identify which agents get batched vs normal
    batched_agent_ids: set[str] = set()
    representative_tasks = []

    for _profile_id, agent_ids in archetype_groups.items():
        group_agents = [agent_lookup[aid] for aid in agent_ids if aid in agent_lookup]
        if len(group_agents) >= min_group_size:
            # Representative: first agent in group
            rep = group_agents[0]
            representative_tasks.append((rep, agent_ids))
            batched_agent_ids.update(agent_ids)

    # Concurrent calls: representatives + unbatched agents
    unbatched = [p for p in participants if p.agent_id not in batched_agent_ids]

    async def _call_agent(p):
        return p.agent_id, await p.step(
            tick, stimuli[p.agent_id], world_state, max_tokens=max_tokens,
            strategy_constraint=bandit_selections.get(p.agent_id, ""),
        )

    # Call representatives
    rep_coros = [_call_agent(rep) for rep, _ in representative_tasks]
    # Call unbatched agents normally
    unbatched_coros = [_call_agent(p) for p in unbatched]

    all_results = await asyncio.gather(*(rep_coros + unbatched_coros))

    for agent_id, resp in all_results:
        responses_map[agent_id] = resp

    # Distribute representative responses to all group members
    for rep, agent_ids in representative_tasks:
        rep_response = responses_map[rep.agent_id]
        for aid in agent_ids:
            if aid != rep.agent_id:
                responses_map[aid] = rep_response
                batched_from_map[aid] = rep.agent_id
                agent = agent_lookup[aid]
                # Mirror what ParticipantAgent.step() does for the representative:
                # 1. Update stimulus/response state for DDA context on next tick
                agent.last_stimulus = stimuli.get(aid, "")
                agent.last_response = rep_response
                # 2. Append both user and assistant to maintain role alternation
                agent.history.append({"role": "user", "content": f"[Tick {tick}] The environment presents: \"{stimuli.get(aid, '')[:120]}\". How do you respond?"})
                agent.history.append({"role": "assistant", "content": rep_response})
                # 3. Copy parsed action and prompt hash from representative
                agent._last_parsed = rep._last_parsed if hasattr(rep, '_last_parsed') else None
                if hasattr(rep, '_last_prompt_hash'):
                    agent._last_prompt_hash = rep._last_prompt_hash

    # Store batched_from info on agents for event emission
    for p in participants:
        p._batched_from = batched_from_map.get(p.agent_id)

    # Return in original participant order
    return [responses_map[aid] for aid in ordered_ids]


# ── Tick orchestration ────────────────────────────────────────────────────────

async def run_tick(
    tick: int,
    environment: EnvironmentAgent,
    participants: list[ParticipantAgent],
    observers: list[ObserverAgent],
    world_state: WorldState,
    alpha: float = 0.2,
    max_tokens: int = 256,
    score_mode: str = "ema",
    logistic_k: float = 6.0,
    flame_engine=None,
    flame_bridge=None,
    topology_manager: Optional[TopologyManager] = None,
    topology_configs: Optional[list] = None,
    bandit=None,
    archetype_groups: Optional[dict[str, list[str]]] = None,
    batch_config=None,
) -> Optional[dict]:
    """
    Execute one tick with three throughput optimizations:

    [Opt 1] Batch stimulus: always use batch_decide (1 call instead of N sequential)
    [Opt 2] Pipeline A+B: authority generates stimuli while swarm processes responses
            — both GPUs active simultaneously via asyncio.gather
    [Opt 3] Phase C+D overlap: on observer ticks, start observer_a analysis (GPU)
            while embeddings compute (CPU) — different hardware, free overlap
    [Opt 4] Phase F: FLAME GPU 2 population step (GPU 2, runs concurrently with
            post-tick bookkeeping when enabled)

    Returns FLAME snapshot dict if FLAME is active, else None.
    """
    n = len(participants)
    is_observer_tick = (tick % world_state.k == 0)

    # -- Pre-phase: persona summaries (concurrent, swarm GPU) ──────────────
    # Run before Phase A so they don't block inside Phase B's asyncio.gather
    await asyncio.gather(*[
        p._maybe_update_persona_summary(tick) for p in participants
    ])

    # -- Phase A -- batch stimulus generation (1 call instead of N) [Opt 1]
    _multi_topo = (
        topology_configs is not None
        and len(topology_configs) > 1
        and topology_manager is not None
    )

    if _multi_topo:
        # Multi-topology: generate stimuli per topology, combine for each participant
        all_stimuli = {}  # {topo_id: {pid: stimulus}}
        for topo_cfg in topology_configs:
            clusters = (topology_manager.get_clusters(topo_cfg.id)
                        if topo_cfg.type == "clustered" else None)
            all_stimuli[topo_cfg.id] = await environment.batch_decide(
                participants, world_state,
                max_tokens=max(64, max_tokens // 2),
                clusters=clusters,
            )
        # Combine into single stimulus per participant
        stimuli = {}
        for p in participants:
            stimuli[p.agent_id] = combine_stimuli(
                all_stimuli, p.agent_id, topology_configs,
            )
        # Emit with topology metadata
        for p in participants:
            payload = {"content": stimuli[p.agent_id]}
            for topo_cfg in topology_configs:
                payload[f"topology_{topo_cfg.id}"] = all_stimuli[topo_cfg.id].get(p.agent_id, "")
            world_state.stream.emit(tick, "A", STIMULUS, p.agent_id, payload)
    else:
        # Fast path: single topology (current behavior, zero overhead)
        stimuli = await environment.batch_decide(
            participants, world_state, max_tokens=max(64, max_tokens // 2)
        )
        for p in participants:
            world_state.stream.emit(tick, "A", STIMULUS, p.agent_id, {
                "content": stimuli.get(p.agent_id, ""),
            })

    # [Fix 9] Compliance check on batch output
    for p in participants:
        _stim_val = stimuli.get(p.agent_id, "")
        if not isinstance(_stim_val, str):
            _stim_val = str(_stim_val)
        violations = check_compliance(_stim_val, "environment")
        if violations:
            logger.debug(f"[COMPLIANCE] environment tick={tick} violations: {violations}")
            world_state._compliance_log.append({
                "tick": tick, "agent": "environment",
                "role": "environment", "violations": violations,
            })
            world_state.stream.emit(tick, "A", COMPLIANCE, "environment", {
                "role": "environment", "violations": violations,
            })

    # -- Phase B -- concurrent participant responses (swarm GPU) [Opt 2]
    # Bandit: select strategy per agent before LLM call
    _bandit_selections: dict[str, str] = {}
    if bandit is not None:
        for p in participants:
            if p.agent_id in bandit.agents:
                strategy, constraint = bandit.select(p.agent_id)
                _bandit_selections[p.agent_id] = strategy
            else:
                _bandit_selections[p.agent_id] = ""
                constraint = ""

    # Use archetype batching if configured, otherwise normal concurrent calls
    if (batch_config is not None and batch_config.enabled
            and batch_config.mode == "representative" and archetype_groups):
        responses = await _batch_phase_b(
            tick, participants, world_state, stimuli,
            archetype_groups, batch_config.min_group_size,
            max_tokens, _bandit_selections,
        )
    else:
        responses = await asyncio.gather(*[
            p.step(tick, stimuli[p.agent_id], world_state, max_tokens=max_tokens,
                   strategy_constraint=_bandit_selections.get(p.agent_id, ""))
            for p in participants
        ], return_exceptions=True)
        # Replace exceptions with empty fallback so one timeout doesn't kill the tick
        for i, resp in enumerate(responses):
            if isinstance(resp, BaseException):
                logger.warning("[Tick %d] %s raised %s — using empty fallback",
                               tick, participants[i].agent_id, resp)
                responses[i] = ""

    # Emit responses to event stream (with structured data + prompt hash)
    for p, resp in zip(participants, responses):
        payload = {"content": resp}
        if hasattr(p, '_last_parsed') and p._last_parsed:
            payload["structured"] = p._last_parsed
        if hasattr(p, '_last_prompt_hash') and p._last_prompt_hash:
            payload["prompt_hash"] = p._last_prompt_hash
        if hasattr(p, '_batched_from') and p._batched_from:
            payload["batched_from"] = p._batched_from
        world_state.stream.emit(tick, "B", RESPONSE, p.agent_id, payload)

    # -- Phase C+D overlap [Opt 3] ─────────────────────────────────────────
    # Embedding (CPU) and observer_a analysis (authority GPU) use different
    # hardware — run them concurrently on observer ticks.
    dampening = world_state.score_dampening()

    if is_observer_tick:
        # Start embedding (CPU) and observer_a (authority GPU) simultaneously
        logger.debug(f"[Tick {tick}] Observer cycle (C+D overlapped)...")
        obs_a, obs_b = observers[0], observers[1]

        # Update participant references for ReACT interview tool
        if hasattr(obs_a, 'set_participants'):
            obs_a.set_participants(participants)

        async def _phase_c():
            # Use narrative text for embedding when structured output is available
            texts = [
                extract_narrative(
                    getattr(p, '_last_parsed', None), resp
                )
                for p, resp in zip(participants, responses)
            ]
            parsed_actions = [getattr(p, '_last_parsed', None) for p in participants]
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: embed_score_batch(texts, parsed_actions=parsed_actions)
            )

        async def _phase_d1():
            return await obs_a.analyse(tick, world_state, n)

        (signals_and_ses, analysis) = await asyncio.gather(
            _phase_c(), _phase_d1()
        )

        # Emit observer_a analysis to event stream (with structured data)
        obs_payload = {"content": analysis}
        if hasattr(obs_a, '_last_parsed') and obs_a._last_parsed:
            obs_payload["structured"] = obs_a._last_parsed
        world_state.stream.emit(tick, "D", OBSERVATION, "observer_a", obs_payload)

        # Phase D2: observer_b needs observer_a's analysis (must be sequential)
        ivs = await obs_b.intervene(tick, world_state, n, analysis)
        world_state.active_interventions.extend(ivs)

        # Emit interventions to event stream
        for iv in ivs:
            world_state.stream.emit(tick, "D", EV_INTERVENTION, "observer_b", {
                "type": iv.type,
                "description": iv.description,
                "modifier": iv.modifier,
                "activated_at": iv.activated_at,
                "duration": iv.duration,
                "source": iv.source,
            })
    else:
        # Non-observer tick: just embeddings (use narrative text)
        texts = [
            extract_narrative(
                getattr(p, '_last_parsed', None), resp
            )
            for p, resp in zip(participants, responses)
        ]
        parsed_actions = [getattr(p, '_last_parsed', None) for p in participants]
        loop = asyncio.get_running_loop()
        signals_and_ses = await loop.run_in_executor(
            None, lambda: embed_score_batch(texts, parsed_actions=parsed_actions)
        )

    # -- Score update (always after embedding completes) ────────────────────
    for participant, response, (signal, signal_se) in zip(participants, responses, signals_and_ses):
        score_before = participant.behavioral_score
        participant.behavioral_score = update_score(
            score_before, signal, dampening, alpha,
            mode=score_mode, logistic_k=logistic_k,
            susceptibility=participant.susceptibility,
            resilience=participant.resilience,
        )
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
        # Emit score to event stream
        world_state.stream.emit(tick, "C", SCORE, participant.agent_id, {
            "score_before": score_before,
            "score_after": participant.behavioral_score,
            "signal": signal,
            "signal_se": signal_se,
            "dampening": dampening,
        })

    # -- Bandit reward update (after scoring, before FLAME) ─────────────────
    if bandit is not None and _bandit_selections:
        # Build per-agent signal lookup from this tick's log entries
        tick_entries = world_state._log_by_tick.get(tick, [])
        signal_by_agent = {e.participant_id: e.signal for e in tick_entries}
        for p in participants:
            strategy = _bandit_selections.get(p.agent_id)
            if not strategy or p.agent_id not in bandit.agents:
                continue
            reward = signal_by_agent.get(p.agent_id, 0.5)
            bandit.update(p.agent_id, strategy, reward)

        # Decay every 10 ticks to prevent early lock-in
        if tick > 0 and tick % 10 == 0:
            bandit.decay(0.97)

    # -- Belief update (after scoring, before FLAME) ──────────────────────
    for p in participants:
        if hasattr(p, '_beliefs') and p._beliefs is not None and len(p.score_log) >= 2:
            delta = p.score_log[-1] - p.score_log[-2]
            for dim_name in p._beliefs.dimensions:
                p._beliefs.update_continuous(dim_name, p.behavioral_score, weight=abs(delta))

    # -- Phase F -- FLAME population step (GPU 2, optional) [Opt 4] ────────
    flame_snapshot = None
    if flame_engine is not None and flame_bridge is not None:
        scores = [p.behavioral_score for p in participants]
        flame_bridge.push_influencer_scores(flame_engine, scores)

        # Apply current dampening to FLAME environment
        flame_engine.set_environment(dampening=dampening)

        # Run sub-steps on GPU 2 (offloaded to thread to not block event loop)
        await asyncio.get_running_loop().run_in_executor(
            None, flame_engine.step
        )

        flame_snapshot = flame_bridge.pull_population_stats(
            flame_engine, tick, flame_engine.config["sub_steps"]
        )
        logger.debug(
            "[Tick %d] FLAME: pop=%d mean=%.3f std=%.3f",
            tick, flame_snapshot.n_population,
            flame_snapshot.mean_score, flame_snapshot.std_score,
        )
        # Emit FLAME snapshot to event stream
        world_state.stream.emit(tick, "F", FLAME_SNAPSHOT, "flame", {
            "mean_score": flame_snapshot.mean_score,
            "std_score": flame_snapshot.std_score,
            "n_population": flame_snapshot.n_population,
            "histogram": flame_snapshot.histogram,
        })

    world_state.apply_interventions()

    # -- Graph memory feedback loop (distill tick events into shared graph) ─
    if world_state.graph is not None:
        ops = world_state.graph.distill_tick(tick, world_state.stream)
        if ops > 0:
            world_state.stream.emit(tick, "system", GRAPH_UPDATE, "system", {
                "operations": ops,
                "n_nodes": world_state.graph.n_nodes,
                "n_edges": world_state.graph.n_edges,
            })

    return flame_snapshot


