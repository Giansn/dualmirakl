"""Session orchestration — run_simulation entry point."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from simulation.config.legacy_roles import (
    INTERVENTION_THRESHOLD,
    PERSONA_SUMMARY_INTERVAL,
)
from simulation.signal.computation import set_seed, _get_rng, _get_embed
from simulation.signal.preflight import (
    FLAME_ENABLED, load_world_context, detect_missing_context,
)
from simulation.storage.export import export_results
from simulation.knowledge.agent_memory import AgentMemoryStore, DuckDBMemoryBackend
from simulation.knowledge.graph_memory import GraphMemory
from simulation.core.safety import SafetyGate
from simulation.core.topology import TopologyManager
from simulation.storage.tracking import tracker
from orchestrator import close_client

from simulation.core.state import WorldState
from simulation.core.agents_impl import (
    EnvironmentAgent, ParticipantAgent, ObserverAgent,
)
from simulation.core.tick import run_tick

logger = logging.getLogger(__name__)

__all__ = [
    "run_simulation",
]

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
    score_mode: str = "ema",
    logistic_k: float = 6.0,
    on_tick: Optional[callable] = None,
    flame_enabled: Optional[bool] = None,
    flame_config: Optional[dict] = None,
    scenario_config=None,
    continue_from: Optional[str] = None,
) -> tuple[list[ParticipantAgent], WorldState]:
    """
    Run the stratified multi-agent simulation.

    When scenario_config (ScenarioConfig) is provided, params are extracted
    from the config. Individual params serve as fallbacks / overrides for
    backward compatibility.

    FLAME GPU 2 (optional):
        Set flame_enabled=True (or FLAME_ENABLED=1 env var) to activate
        population dynamics on a 3rd GPU. Requires pyflamegpu + NVIDIA GPU.
        dualmirakl runs normally without it.
    """
    # ── Extract params from scenario config if provided ───────────────────
    _scenario_name = None
    if scenario_config is not None:
        _scenario_name = scenario_config.meta.name
        logger.info(f"Running scenario: {_scenario_name}")
        n_ticks = scenario_config.environment.tick_count
        n_participants = scenario_config.participant_count()
        k = int(scenario_config.scoring_param("K", k))
        alpha = scenario_config.scoring_param("alpha", alpha)
        score_mode = scenario_config.scoring.mode
        logistic_k = scenario_config.scoring_param("logistic_k", logistic_k)
        intervention_threshold = scenario_config.scoring_param("threshold", intervention_threshold)
        persona_summary_interval = scenario_config.memory.summary_interval
        # FLAME from config
        if flame_enabled is None:
            flame_enabled = scenario_config.flame.enabled
        if flame_config is None and scenario_config.flame.enabled:
            flame_config = {
                "n_population": scenario_config.flame.population_size,
                "kappa": scenario_config.flame.kappa,
                "influencer_weight": scenario_config.flame.influencer_weight,
                "sub_steps": scenario_config.flame.sub_steps,
            }

    t_start = time.monotonic()

    run_config = {
        "n_ticks": n_ticks, "n_participants": n_participants,
        "k": k, "alpha": alpha, "history_window": history_window,
        "max_tokens": max_tokens, "seed": seed,
        "intervention_threshold": intervention_threshold,
        "persona_summary_interval": persona_summary_interval,
        "score_mode": score_mode, "logistic_k": logistic_k,
    }
    if _scenario_name:
        run_config["scenario"] = _scenario_name

    # Load world context from uploaded documents (if any)
    world_context = load_world_context()
    if world_context:
        logger.info(f"World context loaded ({len(world_context)} chars)")

    # ── Edge case guards ────────────────────────────────────────────────
    if alpha == 0.0:
        logger.warning("alpha=0.0: scores will never move. Set alpha > 0 for meaningful dynamics.")
    if k > n_ticks:
        logger.warning("K=%d > n_ticks=%d: observer will never fire.", k, n_ticks)

    set_seed(seed)
    world_state = WorldState(k=k)

    # Initialize agent memory store (reuses e5-small-v2 embedding model)
    def _embed_single(text: str) -> np.ndarray:
        return _get_embed().encode([text])[0]
    mem_max = 20
    mem_dedup = 0.9
    if scenario_config is not None:
        mem_max = scenario_config.memory.max_entries_per_agent
        mem_dedup = scenario_config.memory.dedup_threshold
    world_state.memory = AgentMemoryStore(
        embed_fn=_embed_single, max_per_agent=mem_max, dedup_threshold=mem_dedup,
    )

    # Initialize graph memory (real-time feedback loop)
    world_state.graph = GraphMemory()

    # ── GraphRAG: seed graph with document-derived knowledge ──────────
    graph_context = ""
    try:
        from simulation.knowledge.graph_rag import query_graph_context
        graph_context = query_graph_context(
            scenario_context=world_context or (scenario_config.meta.description if scenario_config else ""),
            embed_fn=lambda texts: _get_embed().encode(texts),
            top_k=20,
            threshold=0.4,
        )
        if graph_context:
            logger.info("GraphRAG context loaded (%d chars)", len(graph_context))
            # Seed graph_memory with GraphRAG entities
            try:
                from simulation.knowledge.graph_rag import get_graph_entities, get_graph_relations
                from simulation.knowledge.graph_rag import Entity as _GREntity, Relation as _GRRelation
                raw_entities = get_graph_entities()
                raw_relations = get_graph_relations()
                if raw_entities:
                    gr_entities = [
                        _GREntity(id=e["id"], name=e["name"], type=e["type"], properties=e.get("properties", {}))
                        for e in raw_entities
                    ]
                    gr_relations = [
                        _GRRelation(id=r["id"], source=r["source"], target=r["target"],
                                    rel_type=r["type"], context=r.get("context", ""),
                                    weight=r.get("weight", 1.0))
                        for r in raw_relations
                    ]
                    world_state.graph.seed_from_graphrag(gr_entities, gr_relations)
            except Exception as e:
                logger.debug("GraphRAG graph seeding skipped: %s", e)
    except Exception as e:
        logger.debug("GraphRAG context not available: %s", e)

    # ── Memory persistence: DuckDB write-behind backend ───────────────
    _memory_backend = None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _run_id = f"run_{ts}_s{seed}"
    try:
        _memory_backend = DuckDBMemoryBackend(run_id=_run_id)
        logger.info("Memory persistence enabled (run_id=%s)", _run_id)
    except Exception as e:
        logger.debug("Memory persistence not available: %s", e)

    # ── Experiment tracking: register this run ─────────────────────────
    _exp_db = None
    try:
        from simulation.storage.experiment_db import ExperimentDB
        _exp_db = ExperimentDB()
        _exp_db.register_run(
            run_id=_run_id,
            parameters=run_config,
            sim_seed=seed,
        )
    except Exception as e:
        logger.debug("Experiment tracking not available: %s", e)

    # ── Continue from prior run: load memories ────────────────────────
    if continue_from and _memory_backend:
        try:
            prior_memories = _memory_backend.load_from_run(
                prior_run_id=continue_from,
                scenario_context=world_context or "",
                embed_fn=lambda texts: _get_embed().encode(texts),
                top_k=5,
            )
            for agent_id, mems in prior_memories.items():
                for mem in mems:
                    world_state.memory.create(
                        agent_id=agent_id,
                        title=f"[prior] {mem['title']}",
                        content=mem["content"],
                        tags=mem.get("tags", []) + ["prior_run"],
                        tick=0,
                    )
            n_loaded = sum(len(m) for m in prior_memories.values())
            if n_loaded > 0:
                logger.info("Loaded %d memories from prior run '%s'", n_loaded, continue_from)
        except Exception as e:
            logger.warning("Failed to load prior memories: %s", e)

    # Initialize safety gate from config
    if scenario_config is not None and scenario_config.safety.enabled:
        world_state.safety_gate = SafetyGate(
            allowlist=set(scenario_config.safety.action_allowlist),
        )

    # Combine world context with graph context for richer grounding
    _combined_context = world_context or ""
    if graph_context:
        _combined_context = f"{_combined_context}\n\n{graph_context}" if _combined_context else graph_context

    # ── GPU split + pipeline configuration ──────────────────────────────
    _split_enabled = os.getenv("SIM_GPU_SPLIT", "1") == "1"
    _gpu_backends = ["swarm", "authority"]

    # Pipeline mode v3: 4-worker architecture + adaptive GPU balancing.
    # Environment on swarm balances total token load across GPUs.
    # The stimulus worker runs ahead via queue backpressure — generates T(n+1)
    # stimuli while score worker (CPU) processes T(n) embeddings.
    # AdaptiveBalancer reads GPU power via pynvml and shifts participants
    # between GPUs every tick to equalize load at ~195W per GPU.
    _pipeline = os.getenv("SIM_PIPELINE", "1") == "1" and _split_enabled
    _env_backend = "swarm" if _pipeline else None
    environment = EnvironmentAgent(history_window=history_window, world_context=_combined_context,
                                   backend_override=_env_backend)
    if _pipeline:
        logger.info("Pipeline v3: adaptive GPU balancing, stimulus prefetch during Phase C")

    def _pick_backend(idx):
        if not _split_enabled:
            return None  # use default (swarm only)
        return _gpu_backends[idx % len(_gpu_backends)]

    if scenario_config is not None:
        from simulation.config.agents import AgentFactory, sample_agent_params
        try:
            agent_set = AgentFactory.from_config(scenario_config, rng=_get_rng())
            participant_specs = list(agent_set.by_type("participant"))
            participants = []
            for idx, spec in enumerate(participant_specs[:n_participants]):
                params = sample_agent_params(spec.profile, _get_rng())
                p = ParticipantAgent(
                    spec.agent_id,
                    history_window=history_window,
                    susceptibility=params["susceptibility"],
                    resilience=params["resilience"],
                    backend_override=_pick_backend(idx),
                    persona_summary_interval=persona_summary_interval,
                )
                participants.append(p)
                logger.debug(
                    "[%s] archetype=%s susc=%.3f resil=%.3f gpu=%s",
                    spec.agent_id,
                    spec.profile.id if spec.profile else "none",
                    params["susceptibility"], params["resilience"],
                    p.cfg["backend"],
                )
            # Pad if scenario defines fewer participants than requested
            while len(participants) < n_participants:
                i = len(participants)
                participants.append(
                    ParticipantAgent(f"participant_{i}", history_window=history_window,
                                    backend_override=_pick_backend(i),
                                    persona_summary_interval=persona_summary_interval)
                )
        except Exception as e:
            logger.warning("AgentFactory failed (%s), falling back to default creation", e)
            participants = [
                ParticipantAgent(f"participant_{i}", history_window=history_window,
                                backend_override=_pick_backend(i),
                                persona_summary_interval=persona_summary_interval)
                for i in range(n_participants)
            ]
    else:
        participants = [
            ParticipantAgent(f"participant_{i}", history_window=history_window,
                            backend_override=_pick_backend(i),
                            persona_summary_interval=persona_summary_interval)
            for i in range(n_participants)
        ]

    # ── Archetype batching setup ─────────────────────────────────────────
    _archetype_groups: dict[str, list[str]] | None = None
    _batch_config = None
    if scenario_config is not None:
        _batch_config = getattr(scenario_config, "batching", None)
        if _batch_config and _batch_config.enabled:
            try:
                _archetype_groups = {}
                for spec in participant_specs:
                    pid = spec.profile.id if spec.profile else "default"
                    _archetype_groups.setdefault(pid, []).append(spec.agent_id)
                logger.info(
                    "Archetype batching enabled (%s): %d groups, %s",
                    _batch_config.mode,
                    len(_archetype_groups),
                    {k: len(v) for k, v in _archetype_groups.items()},
                )
            except NameError:
                _archetype_groups = None
                logger.debug("Archetype batching: participant_specs not available")

    if _split_enabled:
        _auth_count = sum(1 for p in participants if p.cfg["backend"] == "authority")
        _swarm_count = sum(1 for p in participants if p.cfg["backend"] == "swarm")
        logger.info("GPU split: %d on authority (GPU 0), %d on swarm (GPU 1)", _auth_count, _swarm_count)

    # ── Contextual Bandit (adaptive strategy selection) ─────────────────
    _bandit = None
    _bandit_enabled = False
    if scenario_config is not None:
        _bandit_enabled = getattr(scenario_config, "bandit_enabled", False)
    if _bandit_enabled:
        try:
            from ml.bandit import ContextualBandit
            _bandit = ContextualBandit(seed=seed)
            for p in participants:
                _bandit.register(p.agent_id)
            logger.info("Contextual bandit enabled for %d agents", len(participants))
        except ImportError:
            logger.warning("ml.bandit not available, skipping bandit")

    # ── Belief system (per-agent Bayesian belief tracking, opt-in) ──────
    _beliefs_enabled = False
    if scenario_config is not None:
        _beliefs_enabled = getattr(scenario_config, "beliefs_enabled", False)
    if _beliefs_enabled:
        try:
            from ml.beliefs import AgentBeliefs
            belief_dims = getattr(scenario_config, "belief_dimensions", [])
            for p in participants:
                ab = AgentBeliefs(p.agent_id)
                for dim in belief_dims:
                    ab.add(dim.get("name", "engagement"),
                           dim.get("description", dim.get("name", "engagement")),
                           dim.get("alpha", 2.0), dim.get("beta", 2.0))
                p._beliefs = ab
            logger.info("Belief system enabled: %d dimensions, %d agents",
                        len(belief_dims), len(participants))
        except ImportError:
            logger.warning("ml.beliefs not available, skipping belief system")
            _beliefs_enabled = False

    # ── Persona generation (Enhancement 3): generate from KG if configured ──
    _use_generated_personas = False
    if scenario_config is not None:
        _persona_cfg = getattr(scenario_config, "persona_generation", None)
        if _persona_cfg and getattr(_persona_cfg, "source", "manual") == "graph":
            try:
                from simulation.knowledge.ontology_generator import generate_personas as _gen_personas
                from simulation.storage.db import get_db as _get_storage_db
                _persona_db = None
                try:
                    _persona_db = _get_storage_db()
                except Exception:
                    pass
                archetypes = [
                    {"id": p.id, "label": p.label, "description": p.description,
                     "properties": p.properties}
                    for p in scenario_config.archetypes.profiles
                ]
                distribution = dict(scenario_config.archetypes.distribution)
                personas = await _gen_personas(
                    scenario_name=scenario_config.meta.name,
                    archetypes=archetypes,
                    distribution=distribution,
                    n_personas=n_participants,
                    graph_context=graph_context,
                    db=_persona_db,
                )
                # Override participant system prompts with generated personas
                for i, (participant, persona) in enumerate(zip(participants, personas)):
                    participant._persona_spec = persona
                    participant._system_override = persona.to_system_prompt()
                _use_generated_personas = True
                logger.info("Generated %d personas from KG", len(personas))
            except Exception as e:
                logger.warning("Persona generation failed, using defaults: %s", e)
    # ── Observer setup (ReACT or standard) ──────────────────────────────
    _use_react = False
    _react_cfg = None
    if scenario_config is not None and scenario_config.react.enabled:
        _use_react = True
        _react_cfg = scenario_config.react

    if _use_react:
        from simulation.observe.react_observer import ReactObserver
        obs_a = ReactObserver(
            "observer_a", "observer_a",
            max_steps=_react_cfg.max_steps,
            history_window=history_window,
            max_tokens=max_tokens,
            world_context=world_context,
            enabled_tools=_react_cfg.tools if _react_cfg.tools else None,
        )
        obs_a.set_participants(participants)
    else:
        obs_a = ObserverAgent(
            "observer_a", "observer_a",
            history_window=history_window, max_tokens=max_tokens,
            world_context=world_context,
        )

    observers = [
        obs_a,
        ObserverAgent("observer_b", "observer_b",
                      history_window=history_window, max_tokens=max_tokens,
                      world_context=world_context,
                      intervention_threshold=intervention_threshold),
    ]

    # ── Topology setup (dual-environment, MiroFish-inspired) ────────────
    _topo_configs = None
    _topo_manager = None
    if scenario_config is not None and len(scenario_config.topologies) > 1:
        _topo_configs = scenario_config.topologies
        _topo_manager = TopologyManager()
        pid_list = [p.agent_id for p in participants]
        for topo_cfg in _topo_configs:
            if topo_cfg.type == "clustered":
                _topo_manager.assign_clusters(
                    topo_cfg.id, pid_list,
                    cluster_size=topo_cfg.cluster_size,
                    rng=_get_rng(),
                )

    # ── FLAME boot (optional — auto-configures W&B + Optuna) ─────────────
    use_flame = flame_enabled if flame_enabled is not None else FLAME_ENABLED
    flame_engine, flame_bridge = None, None
    flame_ctx = None
    if use_flame:
        from simulation.flame_setup import flame_boot
        flame_ctx = flame_boot(run_config, flame_config, n_participants)
        flame_engine = flame_ctx.engine
        flame_bridge = flame_ctx.bridge
    else:
        # 2-GPU mode: W&B still available but without FLAME enrichment
        tracker.init_run(run_config)

    # Compact header
    detection = detect_missing_context(scenario_config=scenario_config)
    ctx_status = f"{detection['n_documents']} docs" if detection["has_context"] else "no context"
    if detection["missing"]:
        ctx_status += f" | missing: {', '.join(m['category'] for m in detection['missing'])}"
    gpu_label = "2 GPUs"
    _harmony_mode = _pipeline and not flame_engine
    if flame_engine is not None:
        gpu_label = f"3 GPUs (FLAME: {flame_engine.config['n_population']} pop)"
    elif _harmony_mode:
        gpu_label = "2 GPUs (harmony)"
    print(f"\n\u2500\u2500 sim v3 | {n_ticks} ticks | {n_participants} agents | K={k} | {score_mode} | {gpu_label} \u2500\u2500")
    print(f"  context: {ctx_status}")

    try:
        if _harmony_mode:
            # GPU Harmony: tick-pipelined dual-GPU execution
            from simulation.gpu.harmony import GPUHarmony
            harmony = GPUHarmony(
                environment, participants, observers, world_state,
                alpha=alpha, max_tokens=max_tokens, score_mode=score_mode,
                logistic_k=logistic_k, flame_engine=flame_engine,
                flame_bridge=flame_bridge, topology_manager=_topo_manager,
                topology_configs=_topo_configs, bandit=_bandit,
            )
            # Run pipeline in background, consume tick results from queue
            _harmony_task = asyncio.create_task(harmony.run(n_ticks))

        for tick in range(1, n_ticks + 1):
            ivs_before = len(world_state.active_interventions)

            if _harmony_mode:
                # Harmony hands completed ticks via queue
                _htick, _hparts, _hivs_before = await harmony._tick_done_q.get()
                flame_snapshot = None
            else:
                flame_snapshot = await run_tick(
                    tick, environment, participants, observers, world_state,
                    alpha, max_tokens, score_mode, logistic_k,
                    flame_engine, flame_bridge,
                    topology_manager=_topo_manager,
                    topology_configs=_topo_configs,
                    bandit=_bandit,
                    archetype_groups=_archetype_groups,
                    batch_config=_batch_config,
                )

            # Compact tick line
            bar_filled = int(tick / n_ticks * 12)
            bar = "\u2588" * bar_filled + "\u2591" * (12 - bar_filled)
            scores_str = " ".join(f".{int(p.behavioral_score*100):02d}" for p in participants)

            # Events
            events = []
            is_observer = (tick % k == 0)
            is_persona = (tick % persona_summary_interval == 0 and tick > 0)
            new_ivs = world_state.active_interventions[ivs_before:]
            if is_observer:
                if new_ivs:
                    iv_names = ", ".join(iv.type.split("_")[0] for iv in new_ivs)
                    events.append(f"\u25c8 {iv_names}")
                else:
                    events.append("\u25c8 no intervention")
            if is_persona:
                events.append("\u27f3 persona")
            compliance_this_tick = [c for c in world_state.compliance_report() if c["tick"] == tick]
            if compliance_this_tick:
                events.append(f"! {len(compliance_this_tick)} violations")
            if flame_snapshot is not None:
                events.append(
                    f"\u25a3 pop \u03bc={flame_snapshot.mean_score:.2f} "
                    f"\u03c3={flame_snapshot.std_score:.2f}"
                )

            event_str = "  " + "  ".join(events) if events else ""
            print(f"  T{tick:<3d} {bar}  {scores_str}{event_str}")

            # Progress callback for gateway/UI
            if on_tick:
                pct = int(tick / n_ticks * 100)
                tick_events = []
                if is_observer:
                    if new_ivs:
                        for iv in new_ivs:
                            tick_events.append({"type": "intervention", "detail": iv.type})
                    else:
                        tick_events.append({"type": "observer", "detail": "no intervention needed"})
                if is_persona:
                    tick_events.append({"type": "persona", "detail": "persona summary refresh"})
                if compliance_this_tick:
                    tick_events.append({"type": "compliance", "detail": f"{len(compliance_this_tick)} violations"})
                tick_info = {
                    "tick": tick,
                    "n_ticks": n_ticks,
                    "pct": pct,
                    "scores": [round(p.behavioral_score, 3) for p in participants],
                    "events": tick_events,
                    "participants": participants,
                    "world_state": world_state,
                }
                if flame_snapshot is not None:
                    tick_info["flame"] = {
                        "mean_score": round(flame_snapshot.mean_score, 4),
                        "std_score": round(flame_snapshot.std_score, 4),
                        "n_population": flame_snapshot.n_population,
                        "histogram": flame_snapshot.histogram,
                    }
                on_tick(tick_info)

            # Memory persistence: flush new memories to DuckDB at tick boundary
            if _memory_backend and world_state.memory:
                try:
                    _memory_backend.flush(world_state.memory)
                except Exception as e:
                    logger.debug("Memory flush failed at tick %d: %s", tick, e)

            # Experiment tracking: record per-tick metrics
            if _exp_db:
                try:
                    _tick_metrics = {
                        "mean_score": float(np.mean([p.behavioral_score for p in participants])),
                        "std_score": float(np.std([p.behavioral_score for p in participants])),
                        "min_score": float(min(p.behavioral_score for p in participants)),
                        "max_score": float(max(p.behavioral_score for p in participants)),
                    }
                    _exp_db.record_tick(_run_id, tick, _tick_metrics)
                    for p in participants:
                        _exp_db.record_tick(_run_id, tick, {"score": p.behavioral_score}, agent_id=p.agent_id)
                    _exp_db.flush_ticks()
                except Exception as e:
                    logger.debug("Tick tracking failed at tick %d: %s", tick, e)

            # W&B per-tick logging (no-op if wandb not installed)
            tracker.log_tick(
                tick,
                [p.behavioral_score for p in participants],
                flame_snapshot,
            )

    finally:
        await close_client()

    duration_s = time.monotonic() - t_start

    # Compact summary
    stats = world_state.compute_score_statistics(n_ticks)
    compliance = world_state.compliance_report()
    summary_parts = [f"{duration_s:.1f}s"]
    if stats:
        summary_parts.append(f"mean=.{int(stats['mean']*100):02d}")
        summary_parts.append(f"\u03c3=.{int(stats['std']*100):02d}")
        summary_parts.append(f"{stats['n_above_threshold']}/{stats['n_total']} above 0.7")
    if compliance:
        summary_parts.append(f"{len(compliance)} violations")
    if flame_engine is not None:
        flame_stats = flame_engine.get_population_stats()
        summary_parts.append(
            f"FLAME pop \u03bc={flame_stats['mean_score']:.2f}"
        )

    # Final memory flush to DuckDB
    if _memory_backend and world_state.memory:
        try:
            n_flushed = _memory_backend.flush_all(world_state.memory)
            if n_flushed > 0:
                logger.info("Flushed %d memories to DuckDB", n_flushed)
        except Exception as e:
            logger.warning("Final memory flush failed: %s", e)

    # Export results
    run_config = {
        "n_ticks": n_ticks, "n_participants": n_participants,
        "k": k, "alpha": alpha, "history_window": history_window,
        "max_tokens": max_tokens, "seed": seed,
        "intervention_threshold": intervention_threshold,
        "persona_summary_interval": persona_summary_interval,
        "score_mode": score_mode, "logistic_k": logistic_k,
        "run_id": _run_id,
    }
    if continue_from:
        run_config["continued_from"] = continue_from
    if _use_generated_personas:
        run_config["persona_source"] = "graph"
    if graph_context:
        run_config["graph_context_chars"] = len(graph_context)
    if flame_engine is not None:
        run_config["flame"] = flame_engine.config
    run_dir = export_results(participants, world_state, run_config, duration_s)

    # Experiment tracking: mark run complete
    if _exp_db:
        try:
            _exp_db.complete_run(_run_id, duration_s)
        except Exception as e:
            logger.debug("Run completion tracking failed: %s", e)

    # Export graph memory alongside simulation data
    if world_state.graph is not None:
        graph_export = world_state.graph.export()
        (Path(run_dir) / "graph_memory.json").write_text(
            json.dumps(graph_export, indent=2, default=str)
        )

    # ── Possibility branches report ─────────────────────────────────────
    try:
        from simulation.analysis.possibility_report import compute_possibility_report, render_cli
        _report_config = {
            "alpha": alpha, "kappa": 0.0, "dampening": 1.0,
            "score_mode": score_mode, "logistic_k": logistic_k,
            "susceptibility": float(np.mean([p.susceptibility for p in participants])),
            "resilience": float(np.mean([p.resilience for p in participants])),
        }
        _poss_report = compute_possibility_report(
            score_logs=[p.score_log for p in participants],
            config=_report_config,
            run_id=os.path.basename(run_dir),
        )
        (Path(run_dir) / "possibility_branches.json").write_text(_poss_report.to_json())
        print(render_cli(_poss_report))
    except Exception as e:
        logger.warning("Possibility report failed: %s", e)

    # Export FLAME population data alongside dualmirakl output
    if flame_bridge is not None:
        flame_bridge.export_snapshots(os.path.join(run_dir, "flame_population.json"))
        logger.info("FLAME population data exported to %s", run_dir)

    # W&B summary + artifact (no-op if wandb not installed)
    flame_final = flame_stats if flame_engine is not None else None
    tracker.log_summary(stats, flame_final, duration_s, len(compliance))
    tracker.log_artifact(run_dir, f"run_{seed}")
    tracker.finish()

    # Shutdown FLAME context (engine + all handles)
    if flame_ctx is not None:
        flame_ctx.shutdown()
    elif flame_engine is not None:
        flame_engine.shutdown()

    print(f"\n  \u2500\u2500 done {' | '.join(summary_parts)} \u2500\u2500")
    print(f"  \u2500\u2500 {run_dir} \u2500\u2500")

    return participants, world_state


