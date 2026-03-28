"""
Generalized multi-agent simulation loop — stratified tick architecture (v3).

Merges sim_loopv2.1 base with all v3 patches:
  [Fix 2]  Phase D: sequential A-analyse → B-intervene
  [Fix 5]  Intervention threshold from agent_rolesv3
  [Fix 6]  Score NOT leaked to participant prompt
  [Fix 7]  Persona summary injection every PERSONA_SUMMARY_INTERVAL ticks
  [Fix 9]  Compliance checking after each agent turn
  [Fix 12] Batch embedding in Phase C

Architecture layers (Park et al. 2023; Adaptive-VP, ACL 2025):
  Layer 1 — Scenario Engine:     EnvironmentAgent + run_tick() orchestration
  Layer 2 — Persona Module:      ParticipantAgent (six-component prompt encoding)
  Layer 3 — Feedback/Assessment: ObserverAgent (evaluator-generator separation)
  Layer 4 — State Management:    WorldState (log, interventions, windowed summaries)
  Layer 5 — Memory/Context:      history_window (recency) + embed_score (relevance)
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import math
import os
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from simulation.agent_rolesv3 import (
    AGENT_ROLES,
    INTERVENTION_CODEBOOK,
    ENGAGEMENT_ANCHORS,
    INTERVENTION_THRESHOLD,
    PERSONA_SUMMARY_INTERVAL,
    PERSONA_SUMMARY_TEMPLATE,
    EMBED_BATCH_SIZE,
    check_compliance,
)
from simulation.event_stream import (
    EventStream, SimEvent,
    STIMULUS, RESPONSE, SCORE, OBSERVATION, INTERVENTION as EV_INTERVENTION,
    COMPLIANCE, FLAME_SNAPSHOT, PERSONA, CONTEXT, TOOL_USE, GRAPH_UPDATE,
)
from simulation.graph_memory import GraphMemory
from simulation.agent_memory import DuckDBMemoryBackend
from simulation.action_schema import (
    PARTICIPANT_ACTIONS, OBSERVER_A_ACTIONS, OBSERVER_B_ACTIONS,
    schema_to_prompt, parse_action, extract_narrative,
)
from simulation.agent_memory import AgentMemoryStore
from simulation.safety import (
    ObserverMode, SafetyTier, SafetyGate,
    validate_observer_output, ACTION_SAFETY,
)
from orchestrator import agent_turn, close_client
from simulation.tracking import tracker
from simulation.topology import TopologyManager, combine_stimuli

logger = logging.getLogger(__name__)

# ── FLAME GPU 2 (optional, requires 3rd GPU + pyflamegpu) ────────────────────

FLAME_ENABLED = os.environ.get("FLAME_ENABLED", "0") == "1"


def _flame_config_from_env(overrides: Optional[dict] = None) -> dict:
    """Build FLAME config from env vars + optional overrides."""
    _e = os.environ.get
    cfg = {
        "n_population": int(_e("FLAME_N_POPULATION", "10000")),
        "n_influencers": int(_e("SIM_N_PARTICIPANTS", "4")),
        "space_size": 100.0,
        "interaction_radius": float(_e("FLAME_INTERACTION_RADIUS", "10.0")),
        "alpha": float(_e("SIM_ALPHA", "0.15")),
        "kappa": float(_e("FLAME_KAPPA", "0.1")),
        "dampening": 1.0,
        "influencer_weight": float(_e("FLAME_INFLUENCER_WEIGHT", "5.0")),
        "score_mode": _e("SIM_SCORE_MODE", "ema"),
        "logistic_k": float(_e("SIM_LOGISTIC_K", "6.0")),
        "drift_sigma": float(_e("FLAME_DRIFT_SIGMA", "0.01")),
        "move_speed": 0.5,
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

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_EMBED_PATH = os.environ.get(
    "SIM_EMBED_MODEL",
    os.path.join(os.environ.get("HF_HOME", "/per.volume/huggingface"), "hub", "e5-small-v2"),
)
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
CONTEXT_FILE = os.environ.get(
    "SIM_CONTEXT_FILE",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "context", "world_context.json"),
)


# ── World context from uploaded documents ─────────────────────────────────────

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


# ── World context detection ───────────────────────────────────────────────────
#
# Two separate checks:
#   1. detect_missing_context() — called when user starts a simulation.
#      Inspects world_context.json, reports what's present and what's missing,
#      explains WHY each missing piece matters for result validity.
#      Returns warnings but does NOT block the simulation.
#
#   2. preflight_check() — called on instance/pod startup.
#      Checks infrastructure: vLLM servers reachable, models loaded,
#      embedding model available, ports open. Blocks if critical.
# ──────────────────────────────────────────────────────────────────────────────

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


# ── Preflight check (instance startup) ────────────────────────────────────────

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


# ── Reproducibility ───────────────────────────────────────────────────────────

_rng: Optional[np.random.RandomState] = None
_rng_ctx: contextvars.ContextVar[Optional[np.random.RandomState]] = contextvars.ContextVar(
    "_rng_ctx", default=None
)


def set_seed(seed: int = 42) -> None:
    """Set the RNG for the current context (or global fallback)."""
    rng = np.random.RandomState(seed)
    _rng_ctx.set(rng)
    global _rng
    _rng = rng
    logger.info(f"Seed set to {seed}")


def _get_rng() -> np.random.RandomState:
    """Get the RNG for the current context, falling back to global."""
    ctx_rng = _rng_ctx.get(None)
    if ctx_rng is not None:
        return ctx_rng
    global _rng
    if _rng is None:
        set_seed(42)
    return _rng


# ── Embedding helper ──────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

_embed_model: Optional[SentenceTransformer] = None


def _get_embed() -> SentenceTransformer:
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        _embed_model = SentenceTransformer(DEFAULT_EMBED_PATH)
        logger.info(f"Loaded embedding model from: {DEFAULT_EMBED_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load from {DEFAULT_EMBED_PATH}: {e}. Falling back to HuggingFace.")
        _embed_model = SentenceTransformer("intfloat/e5-small-v2")
        logger.info("Loaded embedding model from HuggingFace: intfloat/e5-small-v2")
    return _embed_model


def _cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Resilient agent_turn wrapper ──────────────────────────────────────────────

# Per-backend semaphores for backpressure (respect vLLM max_num_seqs)
_backend_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_semaphore(backend: str, max_concurrent: int = 10) -> asyncio.Semaphore:
    """Get or create a per-backend semaphore for backpressure."""
    if backend not in _backend_semaphores:
        _backend_semaphores[backend] = asyncio.Semaphore(max_concurrent)
    return _backend_semaphores[backend]


async def _resilient_agent_turn(
    agent_id: str,
    backend: str,
    system_prompt: str,
    user_message: str,
    history: list[dict],
    max_tokens: int,
) -> str:
    sem = _get_semaphore(backend)
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await agent_turn(
                    agent_id=agent_id,
                    backend=backend,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    history=history,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                delay = RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[{agent_id}] agent_turn failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(delay)
                else:
                    fallback = f"[{agent_id} fallback: agent_turn failed after {MAX_RETRIES} attempts]"
                    logger.error(f"[{agent_id}] All retries exhausted. Using fallback response.")
                    return fallback


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Intervention:
    type: str
    description: str
    modifier: dict
    activated_at: int
    duration: int = -1
    source: str = ""


@dataclass
class ObsEntry:
    tick: int
    participant_id: str
    score_before: float
    score_after: float
    stimulus: str
    response: str
    signal: float = 0.0
    signal_se: float = 0.0

    def to_str(self) -> str:
        return (
            f"[T{self.tick}] {self.participant_id} "
            f"(score {self.score_before:.2f}\u2192{self.score_after:.2f}, "
            f"signal={self.signal:.2f}\u00b1{self.signal_se:.3f}) "
            f"stimulus: \"{self.stimulus[:70]}\" | response: \"{self.response[:70]}\""
        )


@dataclass
class WorldState:
    k: int = 3
    _log: list[ObsEntry] = field(default_factory=list)
    _log_by_tick: dict = field(default_factory=dict)  # tick → [ObsEntry]
    active_interventions: list[Intervention] = field(default_factory=list)
    _compliance_log: list[dict] = field(default_factory=list)
    stream: EventStream = field(default_factory=EventStream)
    memory: Optional[AgentMemoryStore] = field(default=None)
    safety_gate: SafetyGate = field(default_factory=SafetyGate)
    graph: Optional[GraphMemory] = field(default=None)

    pom_targets: dict = field(default_factory=lambda: {
        "score_distribution": None,
        "temporal_autocorrelation": None,
        "intervention_response": None,
        "convergence_pattern": None,
    })

    def log(self, entry: ObsEntry) -> None:
        self._log.append(entry)
        self._log_by_tick.setdefault(entry.tick, []).append(entry)

    def full_log(self) -> list[ObsEntry]:
        return list(self._log)

    def compliance_report(self) -> list[dict]:
        return list(self._compliance_log)

    def observer_prompt_window(self, tick: int, n_participants: int) -> str:
        # O(K) lookup instead of O(total_entries)
        window = []
        for t in range(max(1, tick - self.k + 1), tick + 1):
            window.extend(self._log_by_tick.get(t, []))
        if not window:
            return "No observations yet."
        if n_participants > 15:
            scores_before = [e.score_before for e in window]
            scores_after  = [e.score_after  for e in window]
            signals       = [e.signal for e in window]
            crossings     = sum(1 for e in window if e.score_before < 0.7 <= e.score_after)
            return (
                f"Ticks {tick - self.k + 1}\u2013{tick} | {n_participants} participants | "
                f"mean score {np.mean(scores_before):.2f}\u2192{np.mean(scores_after):.2f} "
                f"(\u03c3={np.std(scores_after):.3f}) | "
                f"mean signal {np.mean(signals):.2f} (\u03c3={np.std(signals):.3f}) | "
                f"participants crossing 0.7 threshold: {crossings}"
            )
        return "\n".join(e.to_str() for e in window)

    def apply_interventions(self) -> None:
        still_active = []
        for iv in self.active_interventions:
            if iv.duration == -1:
                still_active.append(iv)
            elif iv.duration > 0:
                iv.duration -= 1
                still_active.append(iv)
        self.active_interventions = still_active

    def environment_constraints(self) -> str:
        return " ".join(iv.description for iv in self.active_interventions
                        if iv.type == "environment_constraint")

    def participant_nudges(self) -> str:
        return " ".join(iv.description for iv in self.active_interventions
                        if iv.type == "participant_nudge")

    def score_dampening(self) -> float:
        d = 1.0
        for iv in self.active_interventions:
            if iv.type == "score_modifier":
                d *= iv.modifier.get("dampening", 1.0)
        return d

    def compute_score_statistics(self, tick: int) -> dict:
        entries = self._log_by_tick.get(tick, [])
        if not entries:
            return {}
        scores = [e.score_after for e in entries]
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores) if n > 1 else 0.0
        skew = float(np.mean(((np.array(scores) - mean) / (std + 1e-8)) ** 3)) if n > 2 else 0.0
        above = sum(1 for s in scores if s >= 0.7)

        prev_entries = self._log_by_tick.get(tick - 1, [])
        autocorr = None
        if prev_entries and len(prev_entries) == len(entries):
            prev_scores = [e.score_after for e in prev_entries]
            combined = np.array(scores + prev_scores)
            mu = np.mean(combined)
            num = sum((s - mu) * (p - mu) for s, p in zip(scores, prev_scores))
            den = sum((s - mu) ** 2 for s in combined)
            autocorr = float(num / (den + 1e-8)) if den > 0 else None

        return {
            "mean": float(mean), "std": float(std), "skewness": float(skew),
            "n_above_threshold": above, "n_total": n,
            "autocorrelation_lag1": autocorr,
        }


# ── Score update ──────────────────────────────────────────────────────────────

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


_anchor_high_mask: Optional[np.ndarray] = None


def _compute_signal_from_vec(vec) -> tuple[float, float]:
    """Vectorized cosine similarity against all anchors."""
    global _anchor_high_mask
    _load_anchors()
    if _anchor_high_mask is None:
        _anchor_high_mask = np.array([l == "high" for l in _anchor_labels])
    # Vectorized: cosine sims for all anchors at once
    norms = np.linalg.norm(_anchor_vecs, axis=1) * (np.linalg.norm(vec) + 1e-8)
    sims = _anchor_vecs @ vec / norms
    high_sim = float(np.mean(sims[_anchor_high_mask]))
    low_sim  = float(np.mean(sims[~_anchor_high_mask]))
    signal = float(np.clip((high_sim - low_sim + 1.0) / 2.0, 0.0, 1.0))
    se = float(np.std(sims) / math.sqrt(len(sims)))
    return signal, se


# ── Action-based signal override ─────────────────────────────────────────────
# Base signal for each structured action type.
ACTION_BASE_SIGNALS: dict[str, float] = {
    "disengage": 0.15,
    "respond": 0.45,
    "escalate": 0.75,
}
EMBED_MODIFIER_RANGE = 0.1


def _compute_signal_with_action(
    parsed_action: Optional[dict],
    embed_vec,
) -> tuple[float, float]:
    """
    Compute behavioral signal using structured action as base, with
    embedding similarity as a fine-grained modifier.

    Falls back to pure embedding if no recognized action.
    """
    embed_signal, embed_se = _compute_signal_from_vec(embed_vec)

    if parsed_action is None:
        return embed_signal, embed_se

    action_type = str(parsed_action.get("action", "")).lower().strip()

    if action_type in ACTION_BASE_SIGNALS:
        base = ACTION_BASE_SIGNALS[action_type]
        modifier = (embed_signal - 0.5) * 2.0 * EMBED_MODIFIER_RANGE
        signal = float(np.clip(base + modifier, 0.0, 1.0))
        return signal, embed_se

    intensity = parsed_action.get("intensity")
    if intensity is not None:
        try:
            intensity_f = float(intensity)
            if 0.0 <= intensity_f <= 1.0:
                signal = float(np.clip(0.7 * intensity_f + 0.3 * embed_signal, 0.0, 1.0))
                return signal, embed_se
        except (TypeError, ValueError):
            pass

    return embed_signal, embed_se


def embed_score_batch(
    responses: list[str],
    parsed_actions: Optional[list[Optional[dict]]] = None,
) -> list[tuple[float, float]]:
    """[Fix 12] Batch encode all responses in one model call.
    If parsed_actions provided, uses action type as base signal with embed modifier."""
    _load_anchors()
    model = _get_embed()
    vecs = model.encode(responses, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)

    if parsed_actions is None:
        return [_compute_signal_from_vec(v) for v in vecs]

    return [
        _compute_signal_with_action(
            parsed_actions[i] if i < len(parsed_actions) else None, v
        )
        for i, v in enumerate(vecs)
    ]


def update_score(
    current: float,
    signal: float,
    dampening: float = 1.0,
    alpha: float = 0.2,
    mode: str = "ema",
    logistic_k: float = 6.0,
    susceptibility: float = 1.0,
    resilience: float = 0.0,
) -> float:
    """
    Score update with two modes and heterogeneous agent modifiers.

    Modes:
      ema      — linear EMA: Score += d * α * (signal - score). Analytically
                 tractable, suitable for SA sweeps.
      logistic — sigmoid-transformed signal: captures saturation at extremes.
                 Clinically motivated: deeply engaged users resist both
                 intervention (hard to push below 0.8) and further escalation
                 (ceiling effect). k controls steepness.

    Agent modifiers (heterogeneous agents):
      susceptibility — scales raw signal before update (higher = more affected
                       by stimuli). Default 1.0 = no modification.
      resilience     — additive dampening (higher = more resistant to score
                       change). Applied multiplicatively with intervention
                       dampening: effective_d = dampening * (1 - resilience).
    """
    # Apply susceptibility: modulates how strongly signal affects this agent
    effective_signal = current + susceptibility * (signal - current)

    if mode == "logistic":
        midpoint = 0.5
        effective_signal = 1.0 / (1.0 + np.exp(-logistic_k * (effective_signal - midpoint)))

    # Apply resilience: baseline resistance to score change
    effective_dampening = dampening * (1.0 - resilience)

    delta = alpha * (effective_signal - current)
    return float(np.clip(current + delta * effective_dampening, 0.0, 1.0))


# ── Sensitivity analysis ─────────────────────────────────────────────────────

def morris_screening(
    func,
    bounds: list[tuple[float, float]],
    r: int = 10,
    p: int = 4,
    seed: int = 42,
) -> dict[int, dict]:
    rng = np.random.RandomState(seed)
    k = len(bounds)
    delta = 1.0 / (p - 1) if p > 1 else 0.5
    effects = {i: [] for i in range(k)}

    for _ in range(r):
        x_base = rng.randint(0, p, size=k) / (p - 1)
        x_scaled = np.array([lo + x_base[i] * (hi - lo) for i, (lo, hi) in enumerate(bounds)])
        order = rng.permutation(k)
        x_current = x_scaled.copy()
        y_current = func(x_current)
        for i in order:
            lo, hi = bounds[i]
            step = delta * (hi - lo)
            x_next = x_current.copy()
            x_next[i] += step
            if x_next[i] > hi:
                x_next[i] -= 2 * step
            y_next = func(x_next)
            ee = (y_next - y_current) / (step if step != 0 else 1e-8)
            effects[i].append(ee)
            x_current = x_next
            y_current = y_next

    return {i: {
        "mu_star": float(np.mean(np.abs(effects[i]))),
        "sigma": float(np.std(effects[i])),
        "mu": float(np.mean(effects[i])),
    } for i in range(k)}


def sobol_first_order(
    func,
    bounds: list[tuple[float, float]],
    n_samples: int = 1024,
    seed: int = 42,
) -> dict[int, float]:
    rng = np.random.RandomState(seed)
    k = len(bounds)

    def _sample(n):
        raw = rng.uniform(size=(n, k))
        return np.array([[lo + raw[j, i] * (hi - lo) for i, (lo, hi) in enumerate(bounds)]
                         for j in range(n)])

    A = _sample(n_samples)
    B = _sample(n_samples)
    y_A = np.array([func(A[j]) for j in range(n_samples)])
    y_B = np.array([func(B[j]) for j in range(n_samples)])
    var_y = np.var(np.concatenate([y_A, y_B]))

    if var_y < 1e-12:
        return {i: 0.0 for i in range(k)}

    indices = {}
    for i in range(k):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        y_AB_i = np.array([func(AB_i[j]) for j in range(n_samples)])
        s_i = float(np.mean(y_B * (y_AB_i - y_A)) / var_y)
        indices[i] = max(0.0, min(1.0, s_i))
    return indices


# ── Intervention extraction ───────────────────────────────────────────────────

_codebook_vecs: Optional[np.ndarray] = None
_codebook_keys: list[str] = []
_codebook_phrases: list[str] = []


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


def extract_interventions(observer_id: str, response: str, tick: int, precomputed_vec=None) -> list[Intervention]:
    _load_codebook()
    vec = precomputed_vec if precomputed_vec is not None else _get_embed().encode([response])[0]
    triggered = {}
    # Vectorized cosine similarity against codebook
    norms = np.linalg.norm(_codebook_vecs, axis=1) * (np.linalg.norm(vec) + 1e-8)
    sims = _codebook_vecs @ vec / norms
    for idx, (key, phrase) in enumerate(zip(_codebook_keys, _codebook_phrases)):
        s = float(sims[idx])
        if s >= INTERVENTION_THRESHOLD:
            if key not in triggered or s > triggered[key][0]:
                triggered[key] = (s, phrase)

    ivs = []
    for key, (sim, phrase) in triggered.items():
        iv_type, description, modifier = _make_intervention(key)
        ivs.append(Intervention(
            type=iv_type, description=description, modifier=modifier,
            activated_at=tick, source=observer_id,
        ))
        logger.info(f"  [{observer_id}] INTERVENTION: {key} (sim={sim:.2f})")
    return ivs


def _make_intervention(key: str) -> tuple[str, str, dict]:
    defaults = {
        "pause_prompt":        ("participant_nudge",
                                "A pause prompt appeared: 'Take a moment to reflect.'", {}),
        "boundary_warning":    ("environment_constraint",
                                "Flag content or interactions approaching scenario boundaries.", {}),
        "pacing_adjustment":   ("participant_nudge",
                                "Consider adjusting the pace of the current interaction.", {}),
        "dynamics_dampening":  ("score_modifier",
                                "Behavioral dynamics dampened by observer recommendation.",
                                {"dampening": 0.6}),
    }
    return defaults.get(key, ("environment_constraint", key, {}))


def _format_stats(stats: dict) -> str:
    if not stats:
        return ""
    s = (
        f"\nPopulation statistics: mean={stats['mean']:.3f}, "
        f"\u03c3={stats['std']:.3f}, skew={stats['skewness']:.2f}, "
        f"above 0.7: {stats['n_above_threshold']}/{stats['n_total']}"
    )
    if stats.get("autocorrelation_lag1") is not None:
        s += f", lag-1 autocorr={stats['autocorrelation_lag1']:.3f}"
    return s


# ── Agent classes ─────────────────────────────────────────────────────────────

class EnvironmentAgent:
    """
    Generates stimuli per participant (Phase A). Singleton.
    DDA: score >= 0.8 -> stabilising, >= 0.5 -> variation, < 0.5 -> baseline.
    """

    def __init__(self, history_window: int = 4, world_context: Optional[str] = None,
                 backend_override: Optional[str] = None):
        self.cfg = {**AGENT_ROLES["environment"]}
        if backend_override:
            self.cfg["backend"] = backend_override
        self.history: list[dict] = []
        self.history_window = history_window
        self.world_context = world_context

    def _system_prompt(self) -> str:
        base = self.cfg["system"]
        if self.world_context:
            return f"[World Context]\n{self.world_context}\n\n{base}"
        return base

    async def decide(self, participant: "ParticipantAgent", world_state: WorldState, max_tokens: int = 128) -> str:
        constraints = world_state.environment_constraints()
        constraint_note = f" ACTIVE CONSTRAINTS: {constraints}" if constraints else ""

        if participant.behavioral_score >= 0.8:
            dda_note = " [HIGH SCORE \u2014 test boundary maintenance, present stabilising stimuli]"
        elif participant.behavioral_score >= 0.5:
            dda_note = " [MODERATE SCORE \u2014 introduce gradual variation]"
        else:
            dda_note = ""

        prompt = (
            f"Participant {participant.agent_id}: behavioral_score={participant.behavioral_score:.2f}."
            f"{constraint_note}{dda_note} "
            f"Last step you presented: \"{participant.last_stimulus[:80]}\". "
            f"They responded: \"{participant.last_response[:80]}\". "
            f"Decide what stimulus to present next. Be specific."
        )
        response = await _resilient_agent_turn(
            agent_id="environment",
            backend=self.cfg["backend"],
            system_prompt=self._system_prompt(),
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=max_tokens,
        )
        self.history.append({"role": "assistant", "content": response})
        return response

    async def batch_decide(
        self, participants: list["ParticipantAgent"], world_state: WorldState,
        max_tokens: int = 256,
    ) -> dict[str, str]:
        constraints = world_state.environment_constraints()
        constraint_note = f"\nACTIVE CONSTRAINTS: {constraints}" if constraints else ""

        participant_summaries = []
        for p in participants:
            if p.behavioral_score >= 0.8:
                dda = "HIGH"
            elif p.behavioral_score >= 0.5:
                dda = "MODERATE"
            else:
                dda = "BASELINE"
            participant_summaries.append(
                f"  {p.agent_id}: score={p.behavioral_score:.2f} ({dda}) "
                f"last_response=\"{p.last_response[:60]}\""
            )

        prompt = (
            f"Generate a stimulus for each of the following {len(participants)} participants. "
            f"Respond with a JSON object mapping participant_id \u2192 stimulus string.{constraint_note}\n\n"
            + "\n".join(participant_summaries)
        )

        # Cap batch output: ~80 tokens per participant stimulus is typical
        batch_max = min(max_tokens * len(participants), max(256, 80 * len(participants)))

        response = await _resilient_agent_turn(
            agent_id="environment",
            backend=self.cfg["backend"],
            system_prompt=self._system_prompt(),
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=batch_max,
        )
        self.history.append({"role": "assistant", "content": response})

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            stimuli = json.loads(cleaned)
            # Ensure all values are strings (LLM may return nested objects)
            stimuli = {k: str(v) if not isinstance(v, str) else v for k, v in stimuli.items()}
            missing = [p.agent_id for p in participants if p.agent_id not in stimuli]
            if missing:
                raise ValueError(f"Missing participants in batch response: {missing}")
            return stimuli
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"batch_decide parse failed ({e}), falling back to sequential")
            stimuli = {}
            for p in participants:
                stimuli[p.agent_id] = await self.decide(p, world_state, max_tokens=max(64, max_tokens // 2))
            return stimuli

    async def batch_decide_clustered(
        self,
        participants: list["ParticipantAgent"],
        world_state: WorldState,
        clusters: list,
        max_tokens: int = 256,
    ) -> dict[str, str]:
        """
        Generate cluster-aware stimuli. Participants in the same cluster
        share context about their neighbors' recent activity.
        """
        constraints = world_state.environment_constraints()
        constraint_note = f"\nACTIVE CONSTRAINTS: {constraints}" if constraints else ""

        cluster_summaries = []
        for cluster in clusters:
            members_info = []
            for pid in cluster.members:
                p = next((x for x in participants if x.agent_id == pid), None)
                if p:
                    members_info.append(
                        f"    {pid}: score={p.behavioral_score:.2f}, "
                        f"last=\"{p.last_response[:50]}\""
                    )
            cluster_summaries.append(
                f"  Cluster {cluster.id} ({len(cluster.members)} members):\n"
                + "\n".join(members_info)
            )

        prompt = (
            f"Generate a community-style stimulus for each participant. "
            f"Members of the same cluster share a common context — reference "
            f"what their neighbors are doing. Respond with a JSON object "
            f"mapping participant_id → stimulus string.{constraint_note}\n\n"
            + "\n".join(cluster_summaries)
        )

        batch_max = min(max_tokens * len(participants), max(256, 80 * len(participants)))

        response = await _resilient_agent_turn(
            agent_id="environment_clustered",
            backend=self.cfg["backend"],
            system_prompt=self._system_prompt(),
            user_message=prompt,
            history=[],
            max_tokens=batch_max,
        )

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            stimuli = json.loads(cleaned)
            missing = [p.agent_id for p in participants if p.agent_id not in stimuli]
            if missing:
                raise ValueError(f"Missing participants in clustered response: {missing}")
            return stimuli
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"batch_decide_clustered parse failed ({e}), falling back to broadcast")
            return await self.batch_decide(participants, world_state, max_tokens)


class ParticipantAgent:
    """
    Reacts to environment stimuli (Phase B). N instances, concurrent.

    Heterogeneous agent parameters (sampled at init):
      susceptibility ∈ [0,1] — how strongly platform stimuli affect engagement signal.
        High susceptibility = signal has more pull on score.
        Sampled from Beta(2, 3) → mode ≈ 0.33, most agents moderately susceptible.
      resilience ∈ [0,1] — baseline resistance to score change.
        High resilience = dampened score updates (resistant to both escalation
        and intervention). Sampled from Beta(2, 5) → mode ≈ 0.2, most agents
        have low-to-moderate resilience.

    These capture individual differences: susceptibility modulates how
    strongly stimuli affect the agent, resilience modulates capacity for
    self-regulation.
    """

    def __init__(
        self,
        agent_id: str,
        history_window: int = 4,
        susceptibility: Optional[float] = None,
        resilience: Optional[float] = None,
        backend_override: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.cfg = {**AGENT_ROLES["participant"]}
        if backend_override:
            self.cfg["backend"] = backend_override
        self.history: list[dict] = []
        self.history_window = history_window
        rng = _get_rng()
        self.behavioral_score: float = float(rng.uniform(0.1, 0.5))
        self.susceptibility: float = susceptibility if susceptibility is not None else float(rng.beta(2.0, 3.0))
        self.resilience: float = resilience if resilience is not None else float(rng.beta(2.0, 5.0))
        if self.susceptibility < 1e-6:
            logger.warning("[%s] susceptibility=%.6f near zero — agent ignores signals", agent_id, self.susceptibility)
        if self.resilience > 1.0 - 1e-6:
            logger.warning("[%s] resilience=%.6f near 1.0 — agent score frozen", agent_id, self.resilience)
        self.score_log: list[float] = []
        self.last_stimulus: str = "nothing yet"
        self.last_response: str = ""
        self.persona_summary: str = ""

    async def _maybe_update_persona_summary(self, tick: int) -> None:
        """[Fix 7] Generate persona summary every PERSONA_SUMMARY_INTERVAL ticks."""
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
        self.persona_summary = await _resilient_agent_turn(
            agent_id=f"{self.agent_id}_summariser",
            backend=self.cfg["backend"],
            system_prompt="You create concise character summaries from conversation history.",
            user_message=summary_prompt,
            history=self.history[-20:],
            max_tokens=120,
        )

    def _build_system_prompt(self, strategy_constraint: str = "") -> str:
        """Prepend persona traits + summary + bandit strategy if available."""
        base = self.cfg["system"]

        # Inject heterogeneous personality traits
        if self.susceptibility > 0.6:
            trait = "You are easily drawn in by engaging content and tend to lose track of time."
        elif self.susceptibility < 0.3:
            trait = "You are generally measured and deliberate in how you engage."
        else:
            trait = ""

        if self.resilience > 0.5:
            trait += " You are good at setting boundaries and stepping back when needed."
        elif self.resilience < 0.2:
            trait += " You sometimes find it hard to disengage once you are involved."

        parts = []
        if self.persona_summary:
            parts.append(PERSONA_SUMMARY_TEMPLATE.format(
                interval=PERSONA_SUMMARY_INTERVAL,
                summary=self.persona_summary,
            ))
        if trait.strip():
            parts.append(trait.strip())
        if strategy_constraint:
            parts.append(f"Current behavioral strategy: {strategy_constraint}")
        parts.append(base)
        return "\n\n".join(parts)

    async def step(self, tick: int, stimulus: str, world_state: WorldState,
                   max_tokens: int = 256, strategy_constraint: str = "") -> str:
        # Persona summaries handled at tick level (concurrent pre-phase)
        nudge = world_state.participant_nudges()
        nudge_note = f" {nudge}" if nudge else ""

        # Retrieve relevant memories and inject into prompt
        memory_context = ""
        if world_state.memory is not None:
            memory_context = world_state.memory.format_for_prompt(
                self.agent_id, stimulus, tick, top_k=3,
            )

        # [Fix 6] Score NOT included in participant prompt
        prompt = (
            f"{memory_context}"
            f"[Tick {tick}]{nudge_note} "
            f"The environment presents: \"{stimulus[:120]}\". "
            f"How do you respond? What do you do?"
        )

        # Inject structured output schema + bandit strategy into system prompt
        system = self._build_system_prompt(strategy_constraint=strategy_constraint)
        system += schema_to_prompt(PARTICIPANT_ACTIONS, "participant")

        # Prompt versioning: hash the full prompt for drift detection
        try:
            from simulation.response_cache import compute_prompt_hash
            self._last_prompt_hash = compute_prompt_hash(system + "\n" + prompt)
        except Exception:
            self._last_prompt_hash = None

        response = await _resilient_agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=system,
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=max_tokens,
        )

        # Parse structured output; fall back to raw text if parsing fails
        self._last_parsed = parse_action(response, PARTICIPANT_ACTIONS)
        if self._last_parsed:
            logger.debug(f"[{self.agent_id}] structured: action={self._last_parsed.get('action')}")

        # Store memory if agent included one in structured output
        if world_state.memory is not None:
            world_state.memory.process_agent_memory_output(
                self.agent_id, self._last_parsed, tick,
            )

        # [Fix 9] Compliance check (on narrative or raw text)
        check_text = extract_narrative(self._last_parsed, response)
        violations = check_compliance(check_text, "participant")
        if violations:
            logger.debug(f"[COMPLIANCE] {self.agent_id} tick={tick} violations: {violations}")
            world_state._compliance_log.append({
                "tick": tick, "agent": self.agent_id,
                "role": "participant", "violations": violations,
            })

        self.last_stimulus = stimulus
        self.last_response = response
        self.history.append({"role": "assistant", "content": response})
        return response


class ObserverAgent:
    """
    Analyst agent — fires every K ticks (Phase D).
    [Fix 2] Split into analyse() (A) and intervene() (B).
    """

    def __init__(self, agent_id: str, role: str, history_window: int = 4,
                 max_tokens: int = 256, world_context: Optional[str] = None):
        self.agent_id = agent_id
        self.cfg = AGENT_ROLES[role]
        self.history: list[dict] = []
        self.history_window = history_window
        self.max_tokens = max_tokens
        self.analyses: list[str] = []
        self.world_context = world_context

    def _system_prompt(self) -> str:
        base = self.cfg["system"]
        if self.world_context:
            return f"[World Context]\n{self.world_context}\n\n{base}"
        return base

    async def analyse(self, tick: int, world_state: WorldState, n_participants: int) -> str:
        """
        Observer A: analysis only. No codebook extraction.
        Mode: ANALYSE (externally controlled by orchestrator).
        Mandatory reasoning gate: structured output must include 'reasoning' field.
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
            f"Describe what you see \u2014 do NOT recommend any interventions."
        )

        # Inject structured output schema
        system = self._system_prompt()
        system += schema_to_prompt(OBSERVER_A_ACTIONS, "observer_a")

        response = await _resilient_agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=system,
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=self.max_tokens,
        )
        logger.debug(f"[{self.agent_id} ANALYSIS] {response[:120]}...")

        # Parse structured output
        self._last_parsed = parse_action(response, OBSERVER_A_ACTIONS)
        if self._last_parsed:
            logger.debug(
                f"[{self.agent_id}] structured: clustering={self._last_parsed.get('clustering')}, "
                f"concern={self._last_parsed.get('concern_level')}"
            )

        # Mode enforcement: ANALYSE mode compliance gate
        mode_violations = validate_observer_output(
            response, ObserverMode.ANALYSE, self._last_parsed,
        )
        if mode_violations:
            logger.debug(f"[MODE] {self.agent_id} ANALYSE violations: {mode_violations}")
            world_state._compliance_log.append({
                "tick": tick, "agent": self.agent_id,
                "role": "observer_a", "violations": mode_violations,
            })
            world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                "mode": ObserverMode.ANALYSE.value,
                "violations": mode_violations,
            })

        # [Fix 9] observer_a must NOT contain intervention keywords
        violations = check_compliance(response, "observer_a")
        if violations:
            logger.debug(f"[COMPLIANCE] {self.agent_id} used intervention keywords: {violations}")
            world_state._compliance_log.append({
                "tick": tick, "agent": self.agent_id,
                "role": "observer_a", "violations": violations,
            })

        self.history.append({"role": "assistant", "content": response})
        self.analyses.append(response)
        return response

    async def intervene(
        self,
        tick: int,
        world_state: WorldState,
        n_participants: int,
        analysis: str,
    ) -> list[Intervention]:
        """[Fix 2] Observer B: proposes interventions based on observer_a's analysis."""
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

        # Inject structured output schema
        system = self._system_prompt()
        system += schema_to_prompt(OBSERVER_B_ACTIONS, "observer_b")

        response = await _resilient_agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=system,
            user_message=prompt,
            history=self.history[-self.history_window:],
            max_tokens=self.max_tokens,
        )
        logger.debug(f"[{self.agent_id} INTERVENTION] {response[:120]}...")
        self.history.append({"role": "assistant", "content": response})
        self.analyses.append(response)

        # Try structured extraction first; fall back to cosine codebook matching
        parsed = parse_action(response, OBSERVER_B_ACTIONS)
        if parsed and parsed.get("action") == "intervene":
            iv_key = parsed.get("intervention_type")
            if iv_key:
                # Safety gate: check if this intervention is allowed
                decision = world_state.safety_gate.evaluate_intervention(iv_key)
                if not decision["allowed"]:
                    logger.warning(
                        f"  [{self.agent_id}] BLOCKED by safety gate: {decision['reason']}"
                    )
                    world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                        "safety_tier": decision["tier"].value,
                        "action": decision["action"],
                        "status": "blocked",
                        "reason": decision["reason"],
                    })
                    return []

                iv_type, description, modifier = _make_intervention(iv_key)
                logger.info(
                    f"  [{self.agent_id}] STRUCTURED INTERVENTION: {iv_key} "
                    f"(safety={decision['tier'].value})"
                )

                # Log reviewed actions to event stream
                if decision["tier"] == SafetyTier.REVIEW:
                    world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                        "safety_tier": "review",
                        "action": decision["action"],
                        "status": "executed",
                    })

                return [Intervention(
                    type=iv_type, description=description, modifier=modifier,
                    activated_at=tick, source=self.agent_id,
                )]
        if parsed and parsed.get("action") == "no_intervention":
            logger.info(f"  [{self.agent_id}] structured: no intervention needed")
            return []

        # Fallback: cosine codebook matching with safety gate
        logger.debug(f"[{self.agent_id}] structured parse failed, falling back to cosine codebook")
        raw_ivs = extract_interventions(self.agent_id, response, tick)
        # Apply safety gate to cosine-extracted interventions
        approved_ivs = []
        for iv in raw_ivs:
            # Map intervention type back to codebook key for safety check
            codebook_key = next(
                (k for k, v in {
                    "pause_prompt": "participant_nudge",
                    "boundary_warning": "environment_constraint",
                    "pacing_adjustment": "participant_nudge",
                    "dynamics_dampening": "score_modifier",
                }.items() if v == iv.type),
                iv.type,
            )
            decision = world_state.safety_gate.evaluate_intervention(codebook_key)
            if decision["allowed"]:
                approved_ivs.append(iv)
                if decision["tier"] == SafetyTier.REVIEW:
                    world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                        "safety_tier": "review",
                        "action": decision["action"],
                        "status": "executed",
                    })
            else:
                logger.warning(
                    f"  [{self.agent_id}] BLOCKED by safety gate: {decision['reason']}"
                )
                world_state.stream.emit(tick, "D", COMPLIANCE, self.agent_id, {
                    "safety_tier": decision["tier"].value,
                    "action": decision["action"],
                    "status": "blocked",
                })
        return approved_ivs


# ── Archetype batching (Phase B) ──────────────────────────────────────────────

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
                # Update agent history with the shared response
                agent = agent_lookup[aid]
                agent.history.append({"role": "assistant", "content": rep_response})
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
            if topo_cfg.type == "clustered":
                clusters = topology_manager.get_clusters(topo_cfg.id)
                all_stimuli[topo_cfg.id] = await environment.batch_decide_clustered(
                    participants, world_state, clusters,
                    max_tokens=max(64, max_tokens // 2),
                )
            else:
                all_stimuli[topo_cfg.id] = await environment.batch_decide(
                    participants, world_state,
                    max_tokens=max(64, max_tokens // 2),
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
        ])

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
            return embed_score_batch(texts, parsed_actions=parsed_actions)

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
        signals_and_ses = embed_score_batch(texts, parsed_actions=parsed_actions)

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
        for p in participants:
            strategy = _bandit_selections.get(p.agent_id)
            if not strategy or p.agent_id not in bandit.agents:
                continue
            # Reward: how much did the score move in the desired direction?
            # Use the signal directly as reward — higher engagement signal = better
            last_entry = world_state._log[-1] if world_state._log else None
            reward = 0.5
            if last_entry and last_entry.participant_id == p.agent_id:
                reward = last_entry.signal
            bandit.update(p.agent_id, strategy, reward)

        # Decay every 10 ticks to prevent early lock-in
        if tick > 0 and tick % 10 == 0:
            bandit.decay(0.97)

    # -- Phase F -- FLAME population step (GPU 2, optional) [Opt 4] ────────
    flame_snapshot = None
    if flame_engine is not None and flame_bridge is not None:
        scores = [p.behavioral_score for p in participants]
        flame_bridge.push_influencer_scores(flame_engine, scores)

        # Apply current dampening to FLAME environment
        flame_engine.set_environment(dampening=dampening)

        # Run sub-steps on GPU 2 (offloaded to thread to not block event loop)
        await asyncio.get_event_loop().run_in_executor(
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

    # Temporarily override module-level threshold for this run
    import simulation.agent_rolesv3 as _cfg
    _original_threshold = _cfg.INTERVENTION_THRESHOLD
    _original_interval = _cfg.PERSONA_SUMMARY_INTERVAL
    _cfg.INTERVENTION_THRESHOLD = intervention_threshold
    _cfg.PERSONA_SUMMARY_INTERVAL = persona_summary_interval

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
        from simulation.graph_rag import query_graph_context
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
                from simulation.graph_rag import get_graph_entities, get_graph_relations
                from simulation.graph_rag import Entity as _GREntity, Relation as _GRRelation
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
        from simulation.experiment_db import ExperimentDB
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

    # Pipeline mode v2: environment on authority (GPU 0) — stimulus generation
    # overlaps with Phase C (CPU embedding). Stimulus worker runs ahead, filling
    # the gap when both GPUs would otherwise be idle during CPU-only scoring.
    _pipeline = os.getenv("SIM_PIPELINE", "1") == "1" and _split_enabled
    _env_backend = None  # authority (default) — stimulus prefetch during Phase C
    environment = EnvironmentAgent(history_window=history_window, world_context=_combined_context,
                                   backend_override=_env_backend)
    if _pipeline:
        logger.info("Pipeline v2: environment+observers on authority, stimulus prefetch during Phase C")

    def _pick_backend(idx):
        if not _split_enabled:
            return None  # use default (swarm only)
        return _gpu_backends[idx % len(_gpu_backends)]

    if scenario_config is not None:
        from simulation.agents import AgentFactory, sample_agent_params
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
                                    backend_override=_pick_backend(i))
                )
        except Exception as e:
            logger.warning("AgentFactory failed (%s), falling back to default creation", e)
            participants = [
                ParticipantAgent(f"participant_{i}", history_window=history_window,
                                backend_override=_pick_backend(i))
                for i in range(n_participants)
            ]
    else:
        participants = [
            ParticipantAgent(f"participant_{i}", history_window=history_window,
                            backend_override=_pick_backend(i))
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

    # ── Persona generation (Enhancement 3): generate from KG if configured ──
    _use_generated_personas = False
    if scenario_config is not None:
        _persona_cfg = getattr(scenario_config, "persona_generation", None)
        if _persona_cfg and getattr(_persona_cfg, "source", "manual") == "graph":
            try:
                from simulation.ontology_generator import generate_personas as _gen_personas
                from simulation.storage import get_db as _get_storage_db
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
        from simulation.react_observer import ReactObserver
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
                      world_context=world_context),
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
            from simulation.gpu_harmony import GPUHarmony
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
        _cfg.INTERVENTION_THRESHOLD = _original_threshold
        _cfg.PERSONA_SUMMARY_INTERVAL = _original_interval

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
        from simulation.possibility_report import compute_possibility_report, render_cli
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


# ── Data export ───────────────────────────────────────────────────────────────

OUTPUT_DIR = os.environ.get("SIM_OUTPUT_DIR", "data")


def export_results(
    participants: list[ParticipantAgent],
    world_state: WorldState,
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


# ── Sensitivity analysis runner ───────────────────────────────────────────────

def run_sensitivity_analysis(
    mode: str = "morris",
    n_ticks: int = 6,
    n_participants: int = 4,
    r: int = 10,
    n_samples: int = 256,
) -> dict:
    bounds = [
        (0.1, 0.4),    # alpha
        (1.0, 6.0),    # K (cast to int)
        (0.60, 0.85),  # intervention threshold
        (0.3, 1.0),    # dampening coefficient
        (0.2, 1.0),    # susceptibility
        (0.0, 0.6),    # resilience
        (3.0, 10.0),   # logistic_k (steepness)
    ]
    param_names = ["alpha", "K", "threshold", "dampening",
                   "susceptibility", "resilience", "logistic_k"]

    def objective(x: np.ndarray) -> float:
        alpha, k_float, threshold, dampening, susceptibility, resilience, logistic_k = x
        rng = np.random.RandomState(42)
        scores = []
        score = 0.3
        for t in range(n_ticks):
            signal = 0.5 + 0.3 * math.sin(t * 0.5) + rng.normal(0, 0.1)
            signal = max(0.0, min(1.0, signal))
            d = dampening if (t % max(1, int(k_float)) == 0) else 1.0
            score = update_score(score, signal, d, alpha,
                                 mode="logistic", logistic_k=logistic_k,
                                 susceptibility=susceptibility,
                                 resilience=resilience)
            scores.append(score)
        return float(np.mean(scores))

    if mode == "morris":
        results = morris_screening(objective, bounds, r=r)
        print("\n\u2500\u2500 Morris Screening Results \u2500\u2500")
        for i, name in enumerate(param_names):
            r_i = results[i]
            print(f"  {name:<15} \u03bc*={r_i['mu_star']:.4f}  \u03c3={r_i['sigma']:.4f}  \u03bc={r_i['mu']:.4f}")
        return {"mode": "morris", "param_names": param_names, "results": results}

    elif mode == "sobol":
        results = sobol_first_order(objective, bounds, n_samples=n_samples)
        print("\n\u2500\u2500 Sobol First-Order Indices \u2500\u2500")
        for i, name in enumerate(param_names):
            print(f"  {name:<15} S_i={results[i]:.4f}")
        return {"mode": "sobol", "param_names": param_names, "results": results}

    else:
        raise ValueError(f"Unknown SA mode: {mode}. Use 'morris' or 'sobol'.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _prompt(label: str, default, cast, valid_range: str = "") -> any:
    hint = f"  ({valid_range})" if valid_range else ""
    raw = input(f"  {label:<22} [default: {default}]{hint}: ").strip()
    return cast(raw) if raw else default


if __name__ == "__main__":
    import sys as _sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Parse --scenario flag from argv before interactive prompts
    _scenario_path = None
    _argv = list(_sys.argv[1:])
    if "--scenario" in _argv:
        idx = _argv.index("--scenario")
        if idx + 1 < len(_argv):
            _scenario_path = _argv[idx + 1]
            _argv = _argv[:idx] + _argv[idx + 2:]

    print("\n-- multi-agent simulation (v3) --")
    print("Modes: [s]imulation (default) | [m]orris screening | [b]sobol analysis | [o]ptuna optimization\n")

    mode = input("  Mode [s/m/b/o]: ").strip().lower() or "s"

    if mode in ("m", "morris"):
        run_sensitivity_analysis(mode="morris")

    elif mode in ("b", "sobol"):
        run_sensitivity_analysis(mode="sobol")

    elif mode in ("o", "optuna"):
        from simulation.optimize import run_optimization
        _e = lambda k, d: os.environ.get(k, d)
        opt_mode = _prompt("Optuna mode", "fast", str, "fast|full")
        n_trials = _prompt("Number of trials", 100, int, "10-1000")
        include_flame = _prompt("Include FLAME params", _e("FLAME_ENABLED", "0"), str, "0|1") == "1"
        target_mean = _prompt("Target mean score", 0.5, float, "0.0-1.0")
        run_optimization(
            mode=opt_mode, n_trials=n_trials,
            include_flame=include_flame, target_mean=target_mean,
        )

    else:
        # Load scenario config if --scenario was provided
        _scenario_cfg = None
        if _scenario_path:
            from simulation.scenario import ScenarioConfig
            _scenario_cfg = ScenarioConfig.load(_scenario_path)
            _scenario_cfg.validate_scenario(strict=True)
            print(f"  scenario: {_scenario_cfg.meta.name} ({_scenario_path})")

        # Defaults from env vars (set in .env), overridable via interactive prompt
        _e = lambda k, d: os.environ.get(k, d)
        n_ticks        = _prompt("Ticks to simulate",      int(_e("SIM_N_TICKS", "12")),       int,   "1-168")
        n_participants = _prompt("Number of participants",  int(_e("SIM_N_PARTICIPANTS", "4")), int,   "1-50")
        k              = _prompt("Observer frequency K",    int(_e("SIM_OBSERVER_K", "4")),     int,   "1-n_ticks")
        alpha          = _prompt("EMA alpha",               float(_e("SIM_ALPHA", "0.15")),     float, "0.1-0.4")
        history_window = _prompt("History window (turns)",  int(_e("SIM_HISTORY_WINDOW", "4")), int,   "1-12")
        max_tokens     = _prompt("Context size (tokens)",   int(_e("SIM_MAX_TOKENS", "192")),   int,   "64-8192")
        seed           = _prompt("Random seed",             int(_e("SIM_SEED", "42")),          int,   "any int")
        score_mode     = _prompt("Score mode",              _e("SIM_SCORE_MODE", "ema"),        str,   "ema|logistic")
        logistic_k     = _prompt("Logistic steepness k",    float(_e("SIM_LOGISTIC_K", "6.0")), float, "3.0-10.0")

        # FLAME GPU 2 (optional 3rd GPU)
        flame_default = _e("FLAME_ENABLED", "0")
        flame_input = _prompt("FLAME GPU 2 (3rd GPU)",  flame_default,  str,   "0=off, 1=on")
        use_flame = flame_input == "1"

        async def main():
            try:
                await run_simulation(
                    n_ticks=n_ticks, n_participants=n_participants,
                    k=k, alpha=alpha, history_window=history_window,
                    max_tokens=max_tokens, seed=seed,
                    score_mode=score_mode, logistic_k=logistic_k,
                    flame_enabled=use_flame,
                    scenario_config=_scenario_cfg,
                )
            finally:
                await close_client()

        asyncio.run(main())
