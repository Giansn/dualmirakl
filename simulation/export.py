"""Simulation result export.

Extracted from sim_loop.py — writes run artifacts (config, trajectories,
observations, compliance, interventions, event stream, agent memories)
to JSON files in data/{run_id}/.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.environ.get("SIM_OUTPUT_DIR", "data")


def export_results(
    participants,
    world_state,
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

    # Score trajectories + agent parameters
    trajectories = {}
    for p in participants:
        trajectories[p.agent_id] = {
            "initial_score": p.score_log[0] if p.score_log else p.behavioral_score,
            "final_score": p.behavioral_score,
            "susceptibility": round(p.susceptibility, 4),
            "resilience": round(p.resilience, 4),
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

    # Full event stream (unified audit log)
    (run_dir / "event_stream.json").write_text(
        json.dumps(world_state.stream.export(), indent=2)
    )

    # Agent memories (if any were stored during the run)
    if world_state.memory is not None and len(world_state.memory) > 0:
        (run_dir / "agent_memories.json").write_text(
            json.dumps(world_state.memory.export(), indent=2)
        )

    logger.info(f"Results exported to {run_dir}")
    return str(run_dir)
