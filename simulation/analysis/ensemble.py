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
    all_score_logs: list[list[list[float]]] = field(default_factory=list)  # per run, per agent, per tick

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
    from simulation.core.runner import run_simulation

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

    # Store raw trajectory data for scenario tree construction
    result.all_score_logs = all_score_logs

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


# ── Nested Monte Carlo ensemble ──────────────────────────────────────────────

@dataclass
class NestedEnsembleResult:
    """Result of a nested ensemble (outer epistemic × inner aleatory+LLM)."""

    experiment_id: str
    param_sets: list[dict] = field(default_factory=list)
    inner_results: list[list[dict]] = field(default_factory=list)  # per param set
    variance_decomposition: dict = field(default_factory=dict)
    ensemble_summary: dict = field(default_factory=dict)
    all_score_logs: list[list[list[float]]] = field(default_factory=list)
    total_runs: int = 0
    total_completed: int = 0

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "n_param_sets": len(self.param_sets),
            "param_sets": self.param_sets,
            "total_runs": self.total_runs,
            "total_completed": self.total_completed,
            "variance_decomposition": self.variance_decomposition,
            "ensemble_summary": self.ensemble_summary,
        }


async def run_nested_ensemble(
    parameter_grid: list[dict],
    n_inner: int = 5,
    scenario_config=None,
    base_seed: int = 42,
    experiment_name: str | None = None,
    on_run: Callable | None = None,
    n_ticks: int = 12,
    n_participants: int = 4,
    **sim_kwargs,
) -> NestedEnsembleResult:
    """
    Nested Monte Carlo ensemble for variance decomposition.

    Outer loop iterates over parameter_grid (epistemic uncertainty).
    Inner loop runs n_inner simulations per param set with different seeds
    (aleatory + LLM uncertainty).

    Variance decomposition:
        var_epistemic = weighted variance of per-param-set mean scores
        var_within    = weighted mean of per-param-set score variances
        var_total     = var_epistemic + var_within

    Args:
        parameter_grid: List of dicts with ScenarioConfig.replicate() kwargs.
            Example: [{"scoring_alpha": 0.1}, {"scoring_alpha": 0.2}]
        n_inner: Number of inner runs per parameter set.
        scenario_config: Base ScenarioConfig to replicate from.
        base_seed: Starting seed. Run i,j gets seed = base_seed + i*n_inner + j.
        experiment_name: Name for the experiment record.
        on_run: Progress callback.
        **sim_kwargs: Pass-through to run_simulation().

    Returns:
        NestedEnsembleResult with variance decomposition.
    """
    from simulation.core.runner import run_simulation

    # Extract ticks from scenario if provided
    _n_ticks = n_ticks
    if scenario_config is not None:
        _n_ticks = scenario_config.environment.tick_count

    # Create experiment
    exp_db = None
    experiment_id = f"nested_{int(time.time())}"
    try:
        from simulation.experiment_db import ExperimentDB
        exp_db = ExperimentDB()
        experiment_id = exp_db.create_experiment(
            name=experiment_name or "nested_ensemble",
            description=f"{len(parameter_grid)} param sets × {n_inner} inner runs",
            config={
                "parameter_grid": parameter_grid,
                "n_inner": n_inner,
                "base_seed": base_seed,
            },
        )
    except Exception as e:
        logger.debug("Experiment DB not available: %s", e)

    result = NestedEnsembleResult(
        experiment_id=experiment_id,
        param_sets=parameter_grid,
    )

    group_means: list[float] = []
    group_variances: list[float] = []
    group_sizes: list[int] = []
    all_score_logs: list[list[list[float]]] = []
    run_count = 0

    for outer_idx, param_set in enumerate(parameter_grid):
        # Create scenario variant for this param set
        variant = scenario_config
        if scenario_config is not None and param_set:
            try:
                variant = scenario_config.replicate(**param_set)
            except Exception as e:
                logger.warning("Failed to replicate config for param set %d: %s", outer_idx, e)
                continue

        inner_runs: list[dict] = []
        inner_metric_values: list[float] = []

        for inner_idx in range(n_inner):
            seed = base_seed + outer_idx * n_inner + inner_idx
            run_count += 1
            run_info: dict = {
                "param_set_id": outer_idx,
                "param_set": param_set,
                "inner_idx": inner_idx,
                "seed": seed,
                "status": "running",
            }

            logger.info(
                "Nested run [%d/%d][%d/%d] (seed=%d)",
                outer_idx + 1, len(parameter_grid),
                inner_idx + 1, n_inner, seed,
            )

            try:
                participants, world_state = await run_simulation(
                    n_ticks=_n_ticks,
                    n_participants=n_participants,
                    seed=seed,
                    scenario_config=variant,
                    **sim_kwargs,
                )

                final_scores = [p.behavioral_score for p in participants]
                score_logs = [p.score_log for p in participants]
                mean_score = float(np.mean(final_scores))

                run_info.update({
                    "status": "completed",
                    "mean_final_score": round(mean_score, 6),
                })
                inner_metric_values.append(mean_score)
                all_score_logs.append(score_logs)
                result.total_completed += 1

            except Exception as e:
                run_info["status"] = "failed"
                run_info["error"] = str(e)
                logger.warning("Nested run failed: %s", e)

            inner_runs.append(run_info)

            if on_run:
                try:
                    on_run({
                        "param_set_id": outer_idx,
                        "inner_idx": inner_idx,
                        "completed_runs": result.total_completed,
                        "total_runs": len(parameter_grid) * n_inner,
                        "pct": int(run_count / (len(parameter_grid) * n_inner) * 100),
                    })
                except Exception:
                    pass

        result.inner_results.append(inner_runs)

        # Compute per-group statistics
        if len(inner_metric_values) >= 2:
            arr = np.array(inner_metric_values)
            group_means.append(float(np.mean(arr)))
            group_variances.append(float(np.var(arr)))
            group_sizes.append(len(inner_metric_values))
        elif len(inner_metric_values) == 1:
            group_means.append(inner_metric_values[0])
            group_variances.append(0.0)
            group_sizes.append(1)

    result.total_runs = run_count

    result.all_score_logs = all_score_logs

    # Variance decomposition
    if len(group_means) >= 2:
        from stats.validation import decompose_variance
        result.variance_decomposition = decompose_variance(
            group_means, group_variances, group_sizes,
        )
    elif len(group_means) == 1:
        result.variance_decomposition = {
            "var_epistemic": 0.0,
            "var_within": group_variances[0] if group_variances else 0.0,
            "var_total": group_variances[0] if group_variances else 0.0,
            "pct_epistemic": 0.0,
            "pct_within": 1.0,
        }

    # Percentile bands across ALL runs
    if all_score_logs:
        result.ensemble_summary = {
            "percentile_bands": _compute_percentile_bands(all_score_logs, _n_ticks),
            "n_param_sets": len(parameter_grid),
            "n_inner": n_inner,
            "n_completed": result.total_completed,
        }

        # Write to DuckDB with variance decomposition
        if exp_db and result.variance_decomposition:
            try:
                bands = result.ensemble_summary["percentile_bands"]
                vd = result.variance_decomposition
                for step, band_stats in bands.items():
                    band_stats_with_var = {
                        **band_stats,
                        "var_epistemic": vd.get("var_epistemic"),
                        "var_aleatory": vd.get("var_within"),  # aleatory+LLM combined
                        "var_llm": None,  # full LLM isolation deferred
                    }
                    exp_db.write_ensemble_summary(
                        experiment_id, int(step), "mean_score", band_stats_with_var,
                    )
            except Exception as e:
                logger.debug("Nested ensemble summary write failed: %s", e)

    logger.info(
        "Nested ensemble complete: %d param sets × %d inner = %d runs (%d completed)",
        len(parameter_grid), n_inner, run_count, result.total_completed,
    )

    return result


# ── Evolutionary ensemble ────────────────────────────────────────────────────

async def run_evolutionary_ensemble(
    n_generations: int = 5,
    n_runs_per_gen: int = 5,
    mu: int = 10,
    lambda_: int = 20,
    fitness_fn=None,
    scenario_config=None,
    base_seed: int = 42,
    **sim_kwargs,
) -> dict:
    """Run N generations of simulation + evolution.

    Each generation runs an ensemble, assigns fitness to genomes based on
    agent scores, then evolves the population for the next generation.

    Args:
        n_generations: Number of evolutionary generations.
        n_runs_per_gen: Ensemble runs per generation.
        mu: Parents retained per generation.
        lambda_: Offspring per generation.
        fitness_fn: Callable(agent_id, score_log) -> float. Defaults to mean score.
        scenario_config: ScenarioConfig (optional).
        base_seed: Starting seed.
        **sim_kwargs: Passed to run_ensemble.

    Returns:
        Dict with generation_stats list and final_population.
    """
    try:
        from ml.evolution import EvolutionEngine, AgentGenome
    except ImportError:
        raise ImportError("ml.evolution not available. pip install -r requirements-ml.txt")

    if fitness_fn is None:
        fitness_fn = lambda agent_id, score_log: float(np.mean(score_log)) if score_log else 0.0

    evo = EvolutionEngine(mu=mu, lambda_=lambda_, seed=base_seed)

    # Initialize population from scenario archetypes or defaults
    n_agents = sim_kwargs.get("n_participants", 4)
    if not evo.population:
        evo.population = [
            AgentGenome(
                agent_id=f"participant_{i}",
                personality={
                    "openness": float(np.random.default_rng(base_seed + i).random()),
                    "conscientiousness": float(np.random.default_rng(base_seed + i + 1).random()),
                    "extraversion": float(np.random.default_rng(base_seed + i + 2).random()),
                    "agreeableness": float(np.random.default_rng(base_seed + i + 3).random()),
                    "neuroticism": float(np.random.default_rng(base_seed + i + 4).random()),
                },
                initial_stance=0.0,
                influence_weight=0.5,
                strategy_bias={},
            )
            for i in range(max(mu, n_agents))
        ]

    generation_stats = []

    for gen in range(n_generations):
        logger.info("Evolution generation %d/%d (pop=%d)",
                     gen + 1, n_generations, len(evo.population))

        # Map genome parameters to simulation kwargs
        gen_seed = base_seed + gen * n_runs_per_gen
        result = await run_ensemble(
            n_runs=n_runs_per_gen,
            base_seed=gen_seed,
            scenario_config=scenario_config,
            **sim_kwargs,
        )

        # Assign fitness from ensemble results
        if result.all_score_logs:
            # Average across runs for each agent position
            for i, genome in enumerate(evo.population[:n_agents]):
                agent_scores = []
                for run_logs in result.all_score_logs:
                    if i < len(run_logs):
                        agent_scores.extend(run_logs[i])
                genome.fitness = fitness_fn(genome.agent_id, agent_scores)

        evo.evolve()
        stats = evo.stats()
        stats["seed"] = gen_seed
        generation_stats.append(stats)
        logger.info("  Gen %d: mean_fitness=%.4f, max=%.4f",
                     gen + 1, stats["mean_fitness"], stats["max_fitness"])

    return {
        "n_generations": n_generations,
        "generation_stats": generation_stats,
        "final_population": [
            {"id": g.agent_id, "fitness": g.fitness,
             "personality": g.personality}
            for g in evo.population
        ],
    }
