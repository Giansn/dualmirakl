"""Preflight checks, world context loading, and FLAME initialization.

Extracted from sim_loop.py — owns infrastructure validation (vLLM connectivity,
embedding model, output directory), document context loading/detection,
and FLAME GPU 2 configuration/initialization.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from simulation.signal_computation import _get_embed, DEFAULT_EMBED_PATH

logger = logging.getLogger(__name__)

# ── FLAME GPU 2 (optional, requires 3rd GPU + pyflamegpu) ────────────────────

FLAME_ENABLED = os.environ.get("FLAME_ENABLED", "0") == "1"


def _flame_config_from_env(overrides: Optional[dict] = None) -> dict:
    """Build FLAME config from env vars + optional overrides."""
    _e = os.environ.get
    cfg = {
        "n_population": int(_e("FLAME_N_POPULATION", "10000000")),
        "n_influencers": int(_e("SIM_N_PARTICIPANTS", "4")),
        "space_size": float(_e("FLAME_SPACE_SIZE", "100000")),
        "interaction_radius": float(_e("FLAME_INTERACTION_RADIUS", "500")),
        "alpha": float(_e("SIM_ALPHA", "0.15")),
        "kappa": float(_e("FLAME_KAPPA", "0.1")),
        "dampening": 1.0,
        "influencer_weight": float(_e("FLAME_INFLUENCER_WEIGHT", "5.0")),
        "score_mode": _e("SIM_SCORE_MODE", "ema"),
        "logistic_k": float(_e("SIM_LOGISTIC_K", "6.0")),
        "drift_sigma": float(_e("FLAME_DRIFT_SIGMA", "0.01")),
        "mobility": float(_e("FLAME_MOBILITY", "0.1")),
        "sub_steps": int(_e("FLAME_SUB_STEPS", "10")),
        "gpu_id": int(_e("FLAME_GPU", "2")),
        "seed": int(_e("SIM_SEED", "42")),
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def _try_init_flame(
    config: dict,
    n_participants: int,
) -> tuple:
    """
    Attempt to initialize FLAME engine + bridge. Returns (engine, bridge) or (None, None).
    Fails gracefully if pyflamegpu not installed or no GPU available.
    """
    try:
        from simulation.flame import FlameEngine, FlameBridge
        config["n_influencers"] = n_participants
        engine = FlameEngine(config)
        bridge = FlameBridge(
            n_influencers=n_participants,
            space_size=config.get("space_size", 100.0),
        )
        engine.init()
        bridge.push_influencer_positions(engine)
        logger.info("FLAME GPU 2 initialized: %d population agents on GPU %d",
                     config["n_population"], config["gpu_id"])
        return engine, bridge
    except ImportError:
        logger.warning("pyflamegpu not installed — FLAME disabled")
        return None, None
    except Exception as e:
        logger.warning("FLAME init failed: %s — continuing without FLAME", e)
        return None, None


# ── World context ────────────────────────────────────────────────────────────

CONTEXT_FILE = os.environ.get(
    "SIM_CONTEXT_FILE",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "context", "world_context.json"),
)


def load_world_context() -> Optional[str]:
    """
    Load document context uploaded via the web interface.

    Returns the summary text (max ~3000 chars) from context/world_context.json,
    or None if no documents have been uploaded.

    This text gets injected into environment and observer agent prompts
    so the simulation is grounded in the uploaded data.
    """
    ctx_path = Path(CONTEXT_FILE)
    if not ctx_path.exists():
        return None
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
        summary = ctx.get("summary", "").strip()
        return summary if summary else None
    except Exception:
        return None


# ── World context detection ──────────────────────────────────────────────────

# Expected context categories and why they matter
CONTEXT_REQUIREMENTS: dict[str, dict] = {
    "scenario_description": {
        "keywords": ["scenario", "situation", "setting", "context", "environment",
                     "case study", "use case", "domain"],
        "why": (
            "Without a scenario description, the environment agent generates "
            "generic stimuli with no domain grounding. Agents interact in a "
            "vacuum — results cannot be mapped to any real-world setting."
        ),
    },
    "population_characteristics": {
        "keywords": ["population", "participant", "demographic", "age", "group",
                     "cohort", "sample", "user", "student", "adolescent"],
        "why": (
            "Without population data, all simulated participants are generic. "
            "Heterogeneous agent parameters (susceptibility, resilience) have "
            "no empirical basis — individual differences are random noise "
            "rather than calibrated to a real population."
        ),
    },
    "outcome_criteria": {
        "keywords": ["outcome", "criteria", "threshold", "metric", "measure",
                     "indicator", "score", "prevalence", "rate", "target"],
        "why": (
            "Without outcome criteria, there is no way to validate whether "
            "simulation results are plausible. History Matching targets "
            "default to wide non-constraining ranges — calibration produces "
            "no information."
        ),
    },
    "intervention_rules": {
        "keywords": ["intervention", "rule", "policy", "constraint", "boundary",
                     "limit", "guideline", "protocol", "response"],
        "why": (
            "Without intervention rules, observer agents rely entirely on "
            "generic codebook phrases. Interventions fire based on embedding "
            "similarity rather than domain-appropriate decision criteria."
        ),
    },
    "temporal_structure": {
        "keywords": ["time", "duration", "period", "phase", "stage", "session",
                     "day", "week", "hour", "tick", "timeline"],
        "why": (
            "Without temporal structure, each tick is abstract — there is no "
            "mapping between simulation ticks and real-world time. Observer "
            "frequency (K) and persona summary intervals have no empirical "
            "anchor."
        ),
    },
}


def detect_missing_context(scenario_config=None) -> dict:
    """
    Inspect world context for missing information categories.

    Called when user starts a simulation. Returns what's present, what's
    missing, and why each missing piece matters.

    If scenario_config is provided, uses its context_categories instead of
    the hardcoded CONTEXT_REQUIREMENTS. This enables domain-specific context
    detection per scenario.

    Does NOT block the simulation — the user decides whether to proceed
    or upload more documents first.

    Returns:
        {
            "has_context": bool,
            "n_documents": int,
            "context_length": int,
            "present": [{"category": str, "matched_keywords": [str]}],
            "missing": [{"category": str, "why": str}],
            "warnings": [str],
            "can_proceed": True,  # always True — detection only, not blocking
        }
    """
    # Build requirements from scenario config or fall back to hardcoded
    if scenario_config is not None and hasattr(scenario_config, "context_categories"):
        requirements = {}
        for cat in scenario_config.context_categories:
            requirements[cat.id] = {
                "keywords": cat.id.replace("_", " ").split() + cat.description.lower().split()[:5],
                "why": cat.description,
            }
    else:
        requirements = CONTEXT_REQUIREMENTS

    ctx_path = Path(CONTEXT_FILE)
    result = {
        "has_context": False,
        "n_documents": 0,
        "context_length": 0,
        "present": [],
        "missing": [],
        "warnings": [],
        "can_proceed": True,
    }

    if not ctx_path.exists():
        result["warnings"].append("No documents uploaded. Simulation will run without world context.")
        for cat, spec in requirements.items():
            result["missing"].append({"category": cat, "why": spec["why"]})
        return result

    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    except Exception:
        result["warnings"].append("world_context.json is corrupted. Simulation will run without context.")
        for cat, spec in requirements.items():
            result["missing"].append({"category": cat, "why": spec["why"]})
        return result

    summary = ctx.get("summary", "").lower()
    result["has_context"] = bool(summary.strip())
    result["n_documents"] = ctx.get("n_documents", 0)
    result["context_length"] = len(ctx.get("summary", ""))

    if not summary.strip():
        result["warnings"].append("Documents uploaded but summary is empty.")
        for cat, spec in requirements.items():
            result["missing"].append({"category": cat, "why": spec["why"]})
        return result

    # Scan for each category
    for cat, spec in requirements.items():
        matched = [kw for kw in spec["keywords"] if kw in summary]
        if matched:
            result["present"].append({"category": cat, "matched_keywords": matched})
        else:
            result["missing"].append({"category": cat, "why": spec["why"]})

    if result["missing"]:
        missing_names = [m["category"] for m in result["missing"]]
        result["warnings"].append(
            f"Missing context categories: {', '.join(missing_names)}. "
            f"Results may lack domain grounding in these areas."
        )

    if result["context_length"] < 200:
        result["warnings"].append(
            "Context is very short (<200 chars). Consider uploading more "
            "detailed documents for better agent grounding."
        )

    return result


# ── Preflight check (instance startup) ───────────────────────────────────────

async def preflight_check() -> dict:
    """
    Infrastructure check — called on pod/instance arrival.

    Verifies that the simulation can run:
    - vLLM servers reachable (authority, swarm)
    - Embedding model loadable
    - Required Python modules importable
    - Output directory writable

    Returns:
        {
            "ready": bool,
            "checks": [{"name": str, "status": "ok"|"fail", "detail": str}],
        }
    """
    checks = []

    # Check orchestrator connectivity
    try:
        from orchestrator import health_check
        status = await health_check()
        for name, st in status.items():
            ok = st == "up"
            checks.append({"name": f"vllm_{name}", "status": "ok" if ok else "fail", "detail": st})
    except Exception as e:
        checks.append({"name": "vllm_connectivity", "status": "fail", "detail": str(e)})

    # Check embedding model
    try:
        _get_embed()
        checks.append({"name": "embedding_model", "status": "ok", "detail": DEFAULT_EMBED_PATH})
    except Exception as e:
        checks.append({"name": "embedding_model", "status": "fail", "detail": str(e)})

    # Check output directory writable
    try:
        out_dir = Path(os.environ.get("SIM_OUTPUT_DIR", "data"))
        out_dir.mkdir(parents=True, exist_ok=True)
        test_file = out_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        checks.append({"name": "output_dir", "status": "ok", "detail": str(out_dir)})
    except Exception as e:
        checks.append({"name": "output_dir", "status": "fail", "detail": str(e)})

    # Check context file access
    ctx_exists = Path(CONTEXT_FILE).exists()
    checks.append({
        "name": "world_context",
        "status": "ok" if ctx_exists else "warn",
        "detail": "loaded" if ctx_exists else "no documents uploaded yet",
    })

    # Check FLAME GPU 2 (optional — warn, never fail)
    if FLAME_ENABLED:
        try:
            import pyflamegpu
            checks.append({
                "name": "flame_gpu2",
                "status": "ok",
                "detail": f"pyflamegpu available, GPU {os.environ.get('FLAME_GPU', '2')}",
            })
        except ImportError:
            checks.append({
                "name": "flame_gpu2",
                "status": "warn",
                "detail": "FLAME_ENABLED=1 but pyflamegpu not installed — will run without FLAME",
            })
    else:
        checks.append({
            "name": "flame_gpu2",
            "status": "info",
            "detail": "disabled (set FLAME_ENABLED=1 for 3-GPU mode)",
        })

    ready = all(c["status"] != "fail" for c in checks)
    return {"ready": ready, "checks": checks}
