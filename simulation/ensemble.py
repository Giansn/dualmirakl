"""
Ensemble orchestration for dualmirakl.

Runs multiple simulations with sequential seeds, stops early when the
coefficient of variation of a convergence metric drops below threshold.
Aggregates results into percentile bands and writes ensemble summaries
to DuckDB.

Usage:
    from simulation.ensemble import run_ensemble

    result = await run_ensemble(n_runs=10, cv_threshold=0.05, base_seed=42)
    print(result.convergence)  # {'achieved': True, 'at_run_n': 7, 'final_cv': 0.03}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Result of an ensemble of simulation runs."""

    experiment_id: str
    runs: list[dict] = field(default_factory=list)
    ensemble_summary: dict = field(default_factory=dict)
    convergence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "experiment_id": self.experiment_id,
            "n_runs_completed": len([r for r in self.runs if r.get("status") == "completed"]),
            "n_runs_failed": len([r for r in self.runs if r.get("status") == "failed"]),
            "runs": self.runs,
            "ensemble_summary": self.ensemble_summary,
            "convergence": self.convergence,
        }


def _check_convergence(values: list[float], cv_threshold: float) -> dict:
    """Compute CV and check convergence."""
    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv = std / mean if mean != 0 else float("inf")
    return {
        "cv": round(cv, 6),
        "converged": cv < cv_threshold,
        "n_runs": len(values),
        "mean": round(mean, 6),
        "std": round(std, 6),
    }


def _compute_percentile_bands(
    all_score_logs: list[list[list[float]]],
    n_ticks: int,
) -> dict:
    """
    Compute per-step percentile bands across runs.

    all_score_logs: list (per run) of list (per agent) of list (per tick) of float.
    Returns: {step: {mean, std, p5, p25, median, p75, p95, n_runs}}
    """
    bands = {}
    for step in range(n_ticks):
        step_values = []
        for run_logs in all_score_logs:
            # Mean score across agents at this step for this run
            agent_scores = []
            for agent_log in run_logs:
                if step < len(agent_log):
                    agent_scores.append(agent_log[step])
            if agent_scores:
                step_values.append(float(np.mean(agent_scores)))
        if step_values:
            arr = np.array(step_values)
            bands[step + 1] = {
                "n_runs": len(step_values),
                "mean": round(float(np.mean(arr)), 6),
                "std": round(float(np.std(arr)), 6),
                "p5": round(float(np.percentile(arr, 5)), 6),
                "p25": round(float(np.percentile(arr, 25)), 6),
                "median": round(float(np.percentile(arr, 50)), 6),
                "p75": round(float(np.percentile(arr, 75)), 6),
                "p95": round(float(np.percentile(arr, 95)), 6),
            }
    return bands


async def run_ensemble(
    n_runs: int = 10,
    scenario_config=None,
    convergence_metric: str = "mean_score",
    cv_threshold: float = 0.05,
    max_runs: int | None = None,
    base_seed: int = 42,
    on_run: Callable | None = None,
    experiment_name: str | None = None,
    # Pass-through sim params for non-scenario usage:
    n_ticks: int = 12,
    n_participants: int = 4,
    **sim_kwargs,
) -> EnsembleResult:
    """
    Run multiple simulations with sequential seeds, stopping early
    when the coefficient of variation of convergence_metric drops
    below cv_threshold.

    Args:
        n_runs: Target number of runs.
        scenario_config: ScenarioConfig (optional, overrides sim params).
        convergence_metric: Metric to check for convergence.
        cv_threshold: CV threshold for early stopping.
        max_runs: Hard upper bound (None = n_runs).
        base_seed: First seed; subsequent runs use base_seed+1, +2, ...
        on_run: Progress callback called after each run.
        experiment_name: Name for the experiment record.
        n_ticks: Ticks per run (if no scenario_config).
        n_participants: Agents per run (if no scenario_config).
        **sim_kwargs: Additional kwargs passed to run_simulation.

    Returns:
        EnsembleResult with all run data, percentile bands, and convergence info.
    """
    from simulation.sim_loop import run_simulation

    effective_max = max_runs or n_runs

    # Create experiment record
    exp_db = None
    experiment_id = f"ensemble_{int(time.time())}"
    try:
        from simulation.experiment_db import ExperimentDB, get_git_hash
        exp_db = ExperimentDB()
        config_dict = {
            "n_runs": n_runs, "max_runs": effective_max,
            "convergence_metric": convergence_metric,
            "cv_threshold": cv_threshold, "base_seed": base_seed,
            "n_ticks": n_ticks, "n_participants": n_participants,
        }
        if scenario_config is not None:
            config_dict["scenario"] = scenario_config.meta.name
        experiment_id = exp_db.create_experiment(
            name=experiment_name or "ensemble",
            description=f"{n_runs} runs, cv<{cv_threshold}",
            config=config_dict,
        )
    except Exception as e:
        logger.debug("Experiment DB not available: %s", e)

    result = EnsembleResult(experiment_id=experiment_id)
    completed_runs: list[dict] = []
    all_score_logs: list[list[list[float]]] = []
    metric_values: list[float] = []

    # Extract ticks from scenario if provided
    _n_ticks = n_ticks
    if scenario_config is not None:
        _n_ticks = scenario_config.environment.tick_count

    convergence_achieved = False

    for i in range(effective_max):
        seed = base_seed + i
        run_start = time.monotonic()
        run_info: dict = {"seed": seed, "index": i, "status": "running"}

        logger.info("Ensemble run %d/%d (seed=%d)", i + 1, effective_max, seed)

        try:
            participants, world_state = await run_simulation(
                n_ticks=_n_ticks,
                n_participants=n_participants,
                seed=seed,
                scenario_config=scenario_config,
                **sim_kwargs,
            )

            wall_time = time.monotonic() - run_start
            final_scores = [p.behavioral_score for p in participants]
            score_logs = [p.score_log for p in participants]

            run_info.update({
                "status": "completed",
                "wall_time": round(wall_time, 2),
                "run_id": f"run_*_s{seed}",  # approximate; actual run_id set inside run_simulation
                "final_scores": [round(s, 4) for s in final_scores],
                "mean_final_score": round(float(np.mean(final_scores)), 4),
                "std_final_score": round(float(np.std(final_scores)), 4),
            })

            completed_runs.append(run_info)
            all_score_logs.append(score_logs)

            # Extract convergence metric
            if convergence_metric == "mean_score":
                metric_values.append(float(np.mean(final_scores)))
            elif convergence_metric == "std_score":
                metric_values.append(float(np.std(final_scores)))
            elif convergence_metric == "max_score":
                metric_values.append(float(np.max(final_scores)))
            else:
                metric_values.append(float(np.mean(final_scores)))

            # Check convergence after >= 3 successful runs
            if len(metric_values) >= 3:
                conv = _check_convergence(metric_values, cv_threshold)
                result.convergence = {
                    "achieved": conv["converged"],
                    "at_run_n": i + 1 if conv["converged"] else None,
                    "final_cv": conv["cv"],
                    "metric": convergence_metric,
                    "threshold": cv_threshold,
                }
                if conv["converged"]:
                    convergence_achieved = True
                    logger.info(
                        "Convergence achieved at run %d (CV=%.4f < %.4f)",
                        i + 1, conv["cv"], cv_threshold,
                    )

        except Exception as e:
            wall_time = time.monotonic() - run_start
            run_info.update({
                "status": "failed",
                "wall_time": round(wall_time, 2),
                "error": str(e),
            })
            logger.warning("Ensemble run %d failed: %s", i + 1, e)

        result.runs.append(run_info)

        # Progress callback
        if on_run:
            try:
                on_run({
                    "completed_runs": len(completed_runs),
                    "total_runs": effective_max,
                    "pct": int((i + 1) / effective_max * 100),
                    "convergence": result.convergence,
                    "latest_run": run_info,
                })
            except Exception:
                pass

        if convergence_achieved:
            break

    # Aggregate: percentile bands per step
    if all_score_logs:
        result.ensemble_summary = {
            "percentile_bands": _compute_percentile_bands(all_score_logs, _n_ticks),
            "n_completed": len(completed_runs),
            "n_failed": len(result.runs) - len(completed_runs),
            "metric_values": [round(v, 6) for v in metric_values],
        }

        # Write ensemble summaries to DuckDB
        if exp_db:
            try:
                bands = result.ensemble_summary["percentile_bands"]
                for step, stats in bands.items():
                    exp_db.write_ensemble_summary(
                        experiment_id, int(step), "mean_score", stats,
                    )
            except Exception as e:
                logger.debug("Ensemble summary write failed: %s", e)

    # Set convergence if never checked (< 3 runs)
    if not result.convergence:
        result.convergence = {
            "achieved": False,
            "at_run_n": None,
            "final_cv": None,
            "metric": convergence_metric,
            "threshold": cv_threshold,
            "reason": f"Only {len(metric_values)} successful runs (need >= 3)",
        }

    logger.info(
        "Ensemble complete: %d/%d runs, convergence=%s",
        len(completed_runs), len(result.runs), convergence_achieved,
    )

    return result
