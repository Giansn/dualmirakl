"""
Weights & Biases experiment tracking for dualmirakl simulations.

Optional — if wandb is not installed, all functions are no-ops.
dualmirakl runs identically with or without wandb.

Usage:
    from simulation.tracking import tracker

    # At simulation start:
    tracker.init_run(run_config)

    # Each tick:
    tracker.log_tick(tick, scores, flame_snapshot)

    # At simulation end:
    tracker.finish(summary_stats)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


class SimTracker:
    """
    Thin wrapper around wandb. All methods are safe to call
    even if wandb is not installed — they become no-ops.
    """

    def __init__(self):
        self._run = None
        self._enabled = False

    @property
    def available(self) -> bool:
        return _HAS_WANDB

    @property
    def active(self) -> bool:
        return self._enabled and self._run is not None

    def init_run(
        self,
        config: dict,
        project: str = "dualmirakl",
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> bool:
        """
        Initialize a wandb run. Returns True if successful.

        Args:
            config: Full simulation config dict (sim params + flame params)
            project: wandb project name
            name: Run name (auto-generated if None)
            tags: Optional tags (e.g., ["flame", "3gpu", "sweep"])
        """
        if not _HAS_WANDB:
            logger.debug("wandb not installed — tracking disabled")
            return False

        try:
            # Build tags
            run_tags = tags or []
            if config.get("flame"):
                run_tags.append("flame")
                run_tags.append("3gpu")
            else:
                run_tags.append("2gpu")
            run_tags.append(config.get("score_mode", "ema"))

            # Flatten flame config for wandb (nested dicts don't display well)
            flat_config = {k: v for k, v in config.items() if k != "flame"}
            flame_cfg = config.get("flame", {})
            for k, v in flame_cfg.items():
                flat_config[f"flame_{k}"] = v

            self._run = wandb.init(
                project=project,
                name=name,
                config=flat_config,
                tags=run_tags,
                reinit=True,
            )
            self._enabled = True
            logger.info("wandb run initialized: %s", self._run.url)
            return True
        except Exception as e:
            logger.warning("wandb init failed: %s — continuing without tracking", e)
            self._enabled = False
            return False

    def log_tick(
        self,
        tick: int,
        scores: list[float],
        flame_snapshot=None,
        events: Optional[list[dict]] = None,
    ):
        """Log per-tick metrics."""
        if not self.active:
            return

        metrics = {
            "tick": tick,
            "score_mean": sum(scores) / len(scores) if scores else 0,
            "score_std": _std(scores),
            "score_min": min(scores) if scores else 0,
            "score_max": max(scores) if scores else 0,
        }

        # Per-agent scores
        for i, s in enumerate(scores):
            metrics[f"agent_{i}_score"] = s

        # FLAME population metrics
        if flame_snapshot is not None:
            metrics["flame_pop_mean"] = flame_snapshot.mean_score
            metrics["flame_pop_std"] = flame_snapshot.std_score
            metrics["flame_pop_min"] = flame_snapshot.min_score
            metrics["flame_pop_max"] = flame_snapshot.max_score
            metrics["flame_pop_count"] = flame_snapshot.n_population

        # Events as step metadata
        if events:
            for ev in events:
                metrics[f"event_{ev.get('type', 'unknown')}"] = 1

        try:
            self._run.log(metrics, step=tick)
        except Exception as e:
            logger.debug("wandb log failed at tick %d: %s", tick, e)

    def log_summary(
        self,
        stats: Optional[dict] = None,
        flame_final: Optional[dict] = None,
        duration_s: float = 0,
        n_violations: int = 0,
    ):
        """Log final summary metrics."""
        if not self.active:
            return

        summary = {"duration_s": duration_s, "n_violations": n_violations}
        if stats:
            summary["final_mean"] = stats.get("mean", 0)
            summary["final_std"] = stats.get("std", 0)
            summary["n_above_threshold"] = stats.get("n_above_threshold", 0)
        if flame_final:
            summary["flame_final_mean"] = flame_final.get("mean_score", 0)
            summary["flame_final_std"] = flame_final.get("std_score", 0)
            summary["flame_final_count"] = flame_final.get("count", 0)

        try:
            for k, v in summary.items():
                self._run.summary[k] = v
        except Exception as e:
            logger.debug("wandb summary failed: %s", e)

    def log_artifact(self, path: str, name: str, artifact_type: str = "results"):
        """Log a file or directory as a wandb artifact."""
        if not self.active:
            return
        try:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_dir(path) if _is_dir(path) else artifact.add_file(path)
            self._run.log_artifact(artifact)
        except Exception as e:
            logger.debug("wandb artifact failed: %s", e)

    def finish(self):
        """Finalize the wandb run."""
        if not self.active:
            return
        try:
            self._run.finish()
        except Exception:
            pass
        self._run = None
        self._enabled = False


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def _is_dir(path: str) -> bool:
    from pathlib import Path
    return Path(path).is_dir()


# Module-level singleton — import and use directly
tracker = SimTracker()
