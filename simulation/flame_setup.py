"""
FLAME boot sequence — configures FLAME engine, W&B tracking, and Optuna when
a 3rd GPU is available.

Called once from run_simulation() when FLAME is enabled.
All integrations are optional — missing packages are skipped gracefully.

Boot sequence:
    1. Init FLAME engine + bridge (GPU 2)
    2. If wandb installed: init W&B run with FLAME-enriched config
    3. If optuna installed: create/load Optuna study with FLAME param space
    4. Return FlameContext with all handles (engine, bridge, tracker, study)

When FLAME is NOT enabled, none of this runs. dualmirakl operates as before.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FlameContext:
    """
    Container for all FLAME-related state during a simulation run.
    Returned by flame_boot(), consumed by run_simulation().
    """
    engine: object = None         # FlameEngine or None
    bridge: object = None         # FlameBridge or None
    wandb_active: bool = False    # W&B tracking initialized
    optuna_study: object = None   # optuna.Study or None
    optuna_trial: object = None   # current optuna trial (if running inside optimize)

    @property
    def active(self) -> bool:
        """True if FLAME engine is running."""
        return self.engine is not None

    def shutdown(self):
        """Clean shutdown of all FLAME-related resources."""
        if self.engine is not None and hasattr(self.engine, "shutdown"):
            self.engine.shutdown()
        self.engine = None
        self.bridge = None
        self.optuna_study = None
        self.optuna_trial = None


def flame_boot(
    sim_config: dict,
    flame_config: Optional[dict] = None,
    n_participants: int = 4,
) -> FlameContext:
    """
    Boot FLAME and auto-configure W&B + Optuna.

    Called from run_simulation() when flame_enabled=True.
    Returns FlameContext with all initialized handles.

    Args:
        sim_config: Full simulation config dict (alpha, k, score_mode, etc.)
        flame_config: Optional FLAME overrides (kappa, n_population, etc.)
        n_participants: Number of LLM participants (= influencer count)

    Returns:
        FlameContext — .active is True if FLAME engine started successfully
    """
    ctx = FlameContext()
    _e = os.environ.get

    # ── Step 1: Init FLAME engine ────────────────────────────────────────
    from simulation.preflight import _flame_config_from_env, _try_init_flame

    fcfg = _flame_config_from_env(flame_config)
    fcfg.update({
        "alpha": sim_config.get("alpha", 0.15),
        "score_mode": sim_config.get("score_mode", "ema"),
        "logistic_k": sim_config.get("logistic_k", 6.0),
        "seed": sim_config.get("seed", 42),
    })

    engine, bridge = _try_init_flame(fcfg, n_participants)
    ctx.engine = engine
    ctx.bridge = bridge

    if not ctx.active:
        logger.info("FLAME boot: engine failed — returning inactive context")
        return ctx

    logger.info("FLAME boot: engine active — configuring tracking + optimization")

    # ── Step 2: Auto-configure W&B ───────────────────────────────────────
    wandb_project = _e("WANDB_PROJECT", "")
    if wandb_project:
        ctx.wandb_active = _setup_wandb(sim_config, fcfg, wandb_project)
    else:
        # Even without explicit project, try default "dualmirakl" when FLAME boots
        ctx.wandb_active = _setup_wandb(sim_config, fcfg, "dualmirakl")

    # ── Step 3: Auto-configure Optuna study ──────────────────────────────
    ctx.optuna_study = _setup_optuna_study(fcfg)

    return ctx


def _setup_wandb(
    sim_config: dict,
    flame_config: dict,
    project: str,
) -> bool:
    """
    Initialize W&B with FLAME-enriched config.
    Returns True if successful.
    """
    from simulation.tracking import tracker

    if not tracker.available:
        logger.debug("FLAME boot: wandb not installed — skipping tracking")
        return False

    _e = os.environ.get

    # Merge sim + flame config for full parameter visibility
    full_config = {**sim_config}
    full_config["flame"] = flame_config

    tags = ["flame", "3gpu", sim_config.get("score_mode", "ema")]

    # Add population scale tag
    n_pop = flame_config.get("n_population", 10000)
    if n_pop >= 100000:
        tags.append("large-pop")
    elif n_pop >= 10000:
        tags.append("medium-pop")
    else:
        tags.append("small-pop")

    entity = _e("WANDB_ENTITY", "") or None

    ok = tracker.init_run(
        config=full_config,
        project=project,
        tags=tags,
    )

    if ok:
        logger.info("FLAME boot: W&B tracking active (project=%s)", project)
        # Define custom W&B metrics for FLAME population
        try:
            import wandb
            wandb.define_metric("flame_pop_mean", summary="last")
            wandb.define_metric("flame_pop_std", summary="last")
            wandb.define_metric("flame_pop_count", summary="last")
            wandb.define_metric("score_mean", summary="last,min,max")
        except Exception:
            pass
    return ok


def _setup_optuna_study(flame_config: dict):
    """
    Create or load an Optuna study configured for FLAME parameter space.
    Returns optuna.Study or None.
    """
    try:
        import optuna
    except ImportError:
        logger.debug("FLAME boot: optuna not installed — skipping optimization setup")
        return None

    _e = os.environ.get
    storage = _e("OPTUNA_STORAGE", "") or None
    prefix = _e("OPTUNA_STUDY_PREFIX", "dualmirakl")

    # Study name includes FLAME config signature for reproducibility
    n_pop = flame_config.get("n_population", 10000)
    kappa = flame_config.get("kappa", 0.1)
    study_name = f"{prefix}_flame_pop{n_pop}_k{kappa:.1f}"

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
        )
        logger.info(
            "FLAME boot: Optuna study ready (name=%s, storage=%s, trials=%d)",
            study_name, storage or "in-memory", len(study.trials),
        )
        return study
    except Exception as e:
        logger.warning("FLAME boot: Optuna setup failed: %s", e)
        return None


def flame_status() -> dict:
    """
    Return current FLAME + tracking + optimization status.
    For preflight and status endpoints.
    """
    _e = os.environ.get
    status = {
        "flame_enabled": _e("FLAME_ENABLED", "0") == "1",
        "flame_gpu": int(_e("FLAME_GPU", "2")),
        "flame_n_population": int(_e("FLAME_N_POPULATION", "10000")),
    }

    # Check pyflamegpu
    try:
        import pyflamegpu
        status["pyflamegpu"] = "installed"
    except ImportError:
        status["pyflamegpu"] = "not installed"

    # Check wandb
    try:
        import wandb
        status["wandb"] = "installed"
        status["wandb_project"] = _e("WANDB_PROJECT", "") or "dualmirakl (default)"
    except ImportError:
        status["wandb"] = "not installed"

    # Check optuna
    try:
        import optuna
        status["optuna"] = "installed"
        status["optuna_storage"] = _e("OPTUNA_STORAGE", "") or "in-memory"
    except ImportError:
        status["optuna"] = "not installed"

    return status
