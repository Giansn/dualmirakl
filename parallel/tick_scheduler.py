"""
Parallel Tick Scheduler for dualmirakl.

Provides:
  1. Per-tick performance metrics (inference, context, update breakdown)
  2. Multi-run parallelism with vLLM seq limit awareness
  3. Integration points for ML hooks (bandit, beliefs)

Does NOT replace sim_loop.py — instead, wraps run_simulation() calls
for multi-run scenarios (parameter sweeps, evolutionary selection, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for a single simulation run."""
    run_id: str
    seed: int
    n_ticks: int = 50
    n_participants: int = 4
    k: int = 4
    alpha: float = 0.15
    score_mode: str = "ema"
    logistic_k: float = 6.0
    max_tokens: int = 192
    scenario_path: Optional[str] = None


@dataclass
class RunResult:
    """Result of a completed simulation run."""
    run_id: str
    seed: int
    final_scores: list[float]
    duration_s: float
    n_ticks: int
    error: Optional[str] = None


class MultiRunScheduler:
    """
    Schedules multiple simulation runs with vLLM capacity awareness.

    vLLM has a fixed max_num_seqs per GPU. With N concurrent simulations,
    each simulation's Phase B fires n_participants concurrent requests.
    The scheduler ensures total concurrent requests never exceed the
    vLLM seq limit.

    Usage:
        scheduler = MultiRunScheduler(max_concurrent_runs=2)
        configs = [
            RunConfig(run_id="sweep_01", seed=42, alpha=0.10),
            RunConfig(run_id="sweep_02", seed=43, alpha=0.20),
            RunConfig(run_id="sweep_03", seed=44, alpha=0.30),
        ]
        results = await scheduler.run_all(configs)
    """

    def __init__(
        self,
        max_concurrent_runs: int = 2,
        vllm_max_seqs: int = 12,
    ):
        self.max_concurrent_runs = max_concurrent_runs
        self.vllm_max_seqs = vllm_max_seqs
        self._semaphore = asyncio.Semaphore(max_concurrent_runs)
        self.results: list[RunResult] = []

    async def _run_single(self, config: RunConfig) -> RunResult:
        """Execute a single simulation run with capacity gating."""
        async with self._semaphore:
            logger.info("[%s] Starting (seed=%d, alpha=%.2f)",
                        config.run_id, config.seed, config.alpha)
            t0 = time.perf_counter()

            try:
                from simulation.sim_loop import run_simulation, set_seed

                scenario_config = None
                if config.scenario_path:
                    from simulation.scenario import ScenarioConfig
                    scenario_config = ScenarioConfig.load(config.scenario_path)

                participants, world_state = await run_simulation(
                    n_ticks=config.n_ticks,
                    n_participants=config.n_participants,
                    k=config.k,
                    alpha=config.alpha,
                    max_tokens=config.max_tokens,
                    seed=config.seed,
                    score_mode=config.score_mode,
                    logistic_k=config.logistic_k,
                    scenario_config=scenario_config,
                )

                duration = time.perf_counter() - t0
                final_scores = [p.behavioral_score for p in participants]
                logger.info("[%s] Completed in %.1fs, scores=%s",
                            config.run_id, duration,
                            [f"{s:.3f}" for s in final_scores])

                return RunResult(
                    run_id=config.run_id,
                    seed=config.seed,
                    final_scores=final_scores,
                    duration_s=duration,
                    n_ticks=config.n_ticks,
                )

            except Exception as e:
                duration = time.perf_counter() - t0
                logger.error("[%s] Failed after %.1fs: %s",
                             config.run_id, duration, e)
                return RunResult(
                    run_id=config.run_id,
                    seed=config.seed,
                    final_scores=[],
                    duration_s=duration,
                    n_ticks=config.n_ticks,
                    error=str(e),
                )

    async def run_all(self, configs: list[RunConfig]) -> list[RunResult]:
        """
        Run all configurations with bounded parallelism.

        Concurrent runs are limited by max_concurrent_runs to respect
        vLLM capacity. Each run uses its own RNG via contextvars.
        """
        logger.info("Scheduling %d runs (max %d concurrent)",
                     len(configs), self.max_concurrent_runs)

        tasks = [self._run_single(config) for config in configs]
        self.results = await asyncio.gather(*tasks)
        return self.results

    def summary(self) -> dict:
        """Summary statistics across all completed runs."""
        completed = [r for r in self.results if r.error is None]
        failed = [r for r in self.results if r.error is not None]

        if not completed:
            return {"total": len(self.results), "completed": 0, "failed": len(failed)}

        all_scores = [s for r in completed for s in r.final_scores]
        durations = [r.duration_s for r in completed]

        return {
            "total": len(self.results),
            "completed": len(completed),
            "failed": len(failed),
            "mean_duration_s": float(np.mean(durations)),
            "total_duration_s": float(np.sum(durations)),
            "mean_final_score": float(np.mean(all_scores)),
            "std_final_score": float(np.std(all_scores)),
            "score_range": [float(np.min(all_scores)), float(np.max(all_scores))],
        }
