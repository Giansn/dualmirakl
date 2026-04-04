"""
Domain-agnostic output pipeline for dualmirakl simulations.

Chains all analysis components into a single async run():
  ensemble -> statistics -> dynamics -> scenario tree ->
  possibility branches -> validation -> report

Works for any simulation domain (social, market, technical, network)
without code changes -- only the scenario YAML changes.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class PipelineStageConfig:
    """Controls which pipeline stages to execute."""
    ensemble: bool = True
    statistics: bool = True
    dynamics: bool = True
    scenario_tree: bool = True
    possibility_branches: bool = True
    validation: bool = True
    report: bool = True

    # Stage-specific tunables
    dynamics_min_ticks: int = 10
    tree_max_depth: int = 3
    tree_max_branches: int = 4
    tree_target_scenarios: Optional[int] = None  # Dupacova reduction target
    report_format: str = "html"  # "html" | "json" | "both"


# ── Results ───────────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    """Outcome of a single pipeline stage."""
    name: str
    status: str  # "completed" | "skipped" | "failed"
    duration_s: float = 0.0
    skip_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete output of the output pipeline."""
    experiment_id: str
    scenario_name: str
    timestamp: str
    total_duration_s: float = 0.0

    ensemble: Optional[Any] = None
    statistics: Optional[dict] = None
    dynamics: Optional[dict] = None
    scenario_tree: Optional[Any] = None
    scenario_tree_reduced: Optional[Any] = None
    possibility_report: Optional[Any] = None
    validation: Optional[dict] = None
    report_html: Optional[str] = None

    stages: list[StageResult] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return all(s.status in ("completed", "skipped") for s in self.stages)

    @property
    def failed_stages(self) -> list[str]:
        return [s.name for s in self.stages if s.status == "failed"]

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "scenario_name": self.scenario_name,
            "timestamp": self.timestamp,
            "total_duration_s": round(self.total_duration_s, 2),
            "succeeded": self.succeeded,
            "stages": [
                {
                    "name": s.name,
                    "status": s.status,
                    "duration_s": round(s.duration_s, 2),
                    "skip_reason": s.skip_reason,
                    "error": s.error,
                }
                for s in self.stages
            ],
        }
        if self.ensemble is not None:
            d["ensemble"] = self.ensemble.to_dict()
        if self.statistics is not None:
            d["statistics"] = self.statistics
        if self.dynamics is not None:
            d["dynamics"] = self.dynamics
        if self.scenario_tree is not None:
            from simulation.scenario_tree import tree_to_dict
            d["scenario_tree"] = tree_to_dict(self.scenario_tree)
        if self.possibility_report is not None:
            d["possibility_report"] = self.possibility_report.to_dict()
        if self.validation is not None:
            d["validation"] = self.validation
        return d


# ── Pipeline ──────────────────────────────────────────────────────────────────

class OutputPipeline:
    """
    Domain-agnostic output pipeline.

    Usage:
        pipeline = OutputPipeline(scenario_config)
        result = await pipeline.run(n_runs=10)
        result.report_html   # HTML report
        result.validation    # scoring against outcome criteria
    """

    def __init__(
        self,
        scenario_config,
        stage_config: Optional[PipelineStageConfig] = None,
        output_dir: Optional[str] = None,
        on_stage: Optional[callable] = None,
    ):
        self.config = scenario_config
        self.stage_config = stage_config or PipelineStageConfig()
        self.output_dir = Path(output_dir) if output_dir else None
        self.on_stage = on_stage

    def _log_stage(self, name: str, status: str):
        logger.info("Pipeline [%s] %s", name, status)
        if self.on_stage:
            try:
                self.on_stage(name, status)
            except Exception:
                pass

    async def run(
        self,
        n_runs: Optional[int] = None,
        base_seed: int = 42,
        **sim_kwargs,
    ) -> PipelineResult:
        t_start = time.monotonic()
        experiment_id = f"pipeline_{int(time.time())}"

        if self.output_dir is None:
            self.output_dir = Path("data") / experiment_id / "pipeline"

        result = PipelineResult(
            experiment_id=experiment_id,
            scenario_name=self.config.meta.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        ens_cfg = self.config.ensemble
        _n_runs = n_runs or (ens_cfg.n_runs if ens_cfg.enabled else 10)

        # ── Stage 1: Ensemble ────────────────────────────────────────
        all_score_logs: list[list[list[float]]] = []

        if self.stage_config.ensemble:
            stage = await self._stage_ensemble(_n_runs, base_seed, result, **sim_kwargs)
            result.stages.append(stage)
            if stage.status == "completed" and result.ensemble is not None:
                all_score_logs = result.ensemble.all_score_logs
        else:
            result.stages.append(StageResult(
                name="ensemble", status="skipped", skip_reason="disabled",
            ))

        # ── Stage 2: Statistics ──────────────────────────────────────
        if self.stage_config.statistics and all_score_logs:
            result.stages.append(self._stage_statistics(all_score_logs, result))
        else:
            result.stages.append(StageResult(
                name="statistics", status="skipped",
                skip_reason="disabled or no data",
            ))

        # ── Stage 3: Dynamics ────────────────────────────────────────
        if self.stage_config.dynamics and all_score_logs:
            result.stages.append(self._stage_dynamics(all_score_logs, result))
        else:
            result.stages.append(StageResult(
                name="dynamics", status="skipped",
                skip_reason="disabled or no data",
            ))

        # ── Stage 4: Scenario Tree ───────────────────────────────────
        if self.stage_config.scenario_tree and len(all_score_logs) >= 3:
            result.stages.append(self._stage_scenario_tree(all_score_logs, result))
        else:
            result.stages.append(StageResult(
                name="scenario_tree", status="skipped",
                skip_reason="disabled or <3 runs",
            ))

        # ── Stage 5: Possibility Branches ────────────────────────────
        if self.stage_config.possibility_branches and all_score_logs:
            result.stages.append(self._stage_possibility(all_score_logs, result))
        else:
            result.stages.append(StageResult(
                name="possibility_branches", status="skipped",
                skip_reason="disabled or no data",
            ))

        # ── Stage 6: Validation ──────────────────────────────────────
        oc = self.config.outcome_criteria
        if (self.stage_config.validation
                and oc is not None
                and oc.targets
                and result.ensemble is not None):
            result.stages.append(self._stage_validation(all_score_logs, result))
        else:
            result.stages.append(StageResult(
                name="validation", status="skipped",
                skip_reason="disabled or no outcome_criteria",
            ))

        # ── Stage 7: Report ──────────────────────────────────────────
        if self.stage_config.report:
            result.stages.append(self._stage_report(result))
        else:
            result.stages.append(StageResult(
                name="report", status="skipped", skip_reason="disabled",
            ))

        result.total_duration_s = time.monotonic() - t_start
        self._save_artifacts(result)
        return result

    # ── Stage implementations ─────────────────────────────────────────────

    async def _stage_ensemble(self, n_runs, base_seed, result, **sim_kwargs):
        t0 = time.monotonic()
        self._log_stage("ensemble", f"starting ({n_runs} runs)")
        try:
            from simulation.ensemble import run_ensemble

            ens_cfg = self.config.ensemble
            ensemble_result = await run_ensemble(
                n_runs=n_runs,
                scenario_config=self.config,
                convergence_metric=(
                    ens_cfg.convergence_metric if ens_cfg.enabled else "mean_score"
                ),
                cv_threshold=ens_cfg.cv_threshold if ens_cfg.enabled else 0.05,
                max_runs=ens_cfg.max_runs if ens_cfg.enabled else None,
                base_seed=base_seed,
                experiment_name=f"pipeline_{self.config.meta.name}",
                **sim_kwargs,
            )
            result.ensemble = ensemble_result
            self._log_stage("ensemble", "completed")
            return StageResult(
                name="ensemble", status="completed",
                duration_s=time.monotonic() - t0,
            )
        except Exception as e:
            logger.exception("Ensemble stage failed")
            return StageResult(
                name="ensemble", status="failed",
                duration_s=time.monotonic() - t0, error=str(e),
            )

    def _stage_statistics(self, all_score_logs, result):
        t0 = time.monotonic()
        self._log_stage("statistics", "starting")
        try:
            from stats.core import stance_drift, polarization, opinion_clusters
            from stats.validation import convergence_check, bootstrap_ci

            combined: dict[str, Any] = {}

            # Per-agent stats from first run
            if all_score_logs:
                first_run = all_score_logs[0]
                n_ticks = min(len(log) for log in first_run) if first_run else 0
                if n_ticks >= 2 and len(first_run) >= 2:
                    stances = np.array(
                        [log[:n_ticks] for log in first_run]
                    ).T  # (T, N)
                    combined["stance_drift"] = stance_drift(stances)
                    combined["polarization"] = polarization(stances)
                    if stances.shape[1] >= 3:
                        combined["opinion_clusters"] = opinion_clusters(stances)

            # Cross-run convergence
            if len(all_score_logs) >= 3:
                run_means = []
                for run_logs in all_score_logs:
                    finals = [log[-1] for log in run_logs if log]
                    if finals:
                        run_means.append(float(np.mean(finals)))
                if len(run_means) >= 3:
                    combined["cross_run_convergence"] = convergence_check(
                        np.array(run_means),
                        window=min(5, len(run_means) // 2 + 1),
                    )
                    combined["cross_run_bootstrap_ci"] = bootstrap_ci(
                        np.array(run_means),
                    )

            result.statistics = combined
            self._log_stage("statistics", "completed")
            return StageResult(
                name="statistics", status="completed",
                duration_s=time.monotonic() - t0,
            )
        except Exception as e:
            logger.exception("Statistics stage failed")
            return StageResult(
                name="statistics", status="failed",
                duration_s=time.monotonic() - t0, error=str(e),
            )

    def _stage_dynamics(self, all_score_logs, result):
        t0 = time.monotonic()
        self._log_stage("dynamics", "starting")
        try:
            from simulation.dynamics import analyze_simulation

            first_run = all_score_logs[0] if all_score_logs else []
            n_ticks = min(len(log) for log in first_run) if first_run else 0

            if n_ticks < self.stage_config.dynamics_min_ticks:
                return StageResult(
                    name="dynamics", status="skipped",
                    duration_s=time.monotonic() - t0,
                    skip_reason=f"{n_ticks} ticks < min {self.stage_config.dynamics_min_ticks}",
                )

            run_config = {
                "alpha": self.config.scoring_param("alpha", 0.15),
                "kappa": self.config.scoring_param("coupling_kappa", 0.0),
                "dampening": self.config.scoring_param("dampening", 1.0),
                "score_mode": self.config.scoring.mode,
                "logistic_k": self.config.scoring_param("logistic_k", 6.0),
                "susceptibility": 0.4,
                "resilience": 0.29,
            }

            analysis = analyze_simulation(
                score_logs=first_run,
                run_config=run_config,
            )
            result.dynamics = analysis
            self._log_stage("dynamics", "completed")
            return StageResult(
                name="dynamics", status="completed",
                duration_s=time.monotonic() - t0,
            )
        except Exception as e:
            logger.exception("Dynamics stage failed")
            return StageResult(
                name="dynamics", status="failed",
                duration_s=time.monotonic() - t0, error=str(e),
            )

    def _stage_scenario_tree(self, all_score_logs, result):
        t0 = time.monotonic()
        self._log_stage("scenario_tree", "starting")
        try:
            from simulation.scenario_tree import build_scenario_tree, reduce_tree

            tree = build_scenario_tree(
                all_score_logs,
                max_depth=self.stage_config.tree_max_depth,
                max_branches=self.stage_config.tree_max_branches,
            )
            result.scenario_tree = tree

            if self.stage_config.tree_target_scenarios is not None:
                result.scenario_tree_reduced = reduce_tree(
                    tree, self.stage_config.tree_target_scenarios,
                )

            self._log_stage("scenario_tree", "completed")
            return StageResult(
                name="scenario_tree", status="completed",
                duration_s=time.monotonic() - t0,
            )
        except Exception as e:
            logger.exception("Scenario tree stage failed")
            return StageResult(
                name="scenario_tree", status="failed",
                duration_s=time.monotonic() - t0, error=str(e),
            )

    def _stage_possibility(self, all_score_logs, result):
        t0 = time.monotonic()
        self._log_stage("possibility_branches", "starting")
        try:
            from simulation.possibility_report import compute_possibility_report

            first_run = all_score_logs[0] if all_score_logs else []

            config = {
                "alpha": self.config.scoring_param("alpha", 0.15),
                "kappa": self.config.scoring_param("coupling_kappa", 0.0),
                "dampening": self.config.scoring_param("dampening", 1.0),
                "score_mode": self.config.scoring.mode,
                "logistic_k": self.config.scoring_param("logistic_k", 6.0),
                "susceptibility": 0.4,
                "resilience": 0.29,
            }

            report = compute_possibility_report(
                score_logs=first_run,
                config=config,
                run_id=result.experiment_id,
                multi_run_logs=(
                    all_score_logs if len(all_score_logs) >= 3 else None
                ),
            )
            result.possibility_report = report
            self._log_stage(
                "possibility_branches",
                f"completed ({report.n_branches} branches)",
            )
            return StageResult(
                name="possibility_branches", status="completed",
                duration_s=time.monotonic() - t0,
            )
        except Exception as e:
            logger.exception("Possibility branches stage failed")
            return StageResult(
                name="possibility_branches", status="failed",
                duration_s=time.monotonic() - t0, error=str(e),
            )

    def _stage_validation(self, all_score_logs, result):
        t0 = time.monotonic()
        self._log_stage("validation", "starting")
        try:
            from stats.scoring import score_ensemble

            criteria = self.config.outcome_criteria

            # Build forecast distributions from ensemble runs
            forecasts: dict[str, np.ndarray] = {}
            observed: dict[str, float] = {}

            for target in criteria.targets:
                target_values = []
                for run_logs in all_score_logs:
                    if not run_logs:
                        continue
                    metric_val = self._extract_metric(target.name, run_logs)
                    if metric_val is not None:
                        target_values.append(metric_val)

                if target_values:
                    forecasts[target.name] = np.array(target_values)
                if target.observed is not None:
                    observed[target.name] = target.observed

            # Scoring rules (CRPS, Wasserstein, etc.)
            scoring_results = score_ensemble(forecasts, observed) if observed else {}

            # Range checks
            range_checks: dict[str, dict] = {}
            for target in criteria.targets:
                if target.name in forecasts:
                    vals = forecasts[target.name]
                    median = float(np.median(vals))
                    in_range = target.low <= median <= target.high
                    range_checks[target.name] = {
                        "median_forecast": round(median, 6),
                        "plausible_range": [target.low, target.high],
                        "in_range": in_range,
                        "pct_in_range": round(
                            float(np.mean(
                                (vals >= target.low) & (vals <= target.high)
                            )),
                            4,
                        ),
                    }

            converged = result.ensemble.convergence.get("achieved", False)
            validation = {
                "scoring": scoring_results,
                "range_checks": range_checks,
                "all_in_range": all(
                    rc["in_range"] for rc in range_checks.values()
                ),
                "convergence_achieved": converged,
                "passed": (
                    all(rc["in_range"] for rc in range_checks.values())
                    and (not criteria.convergence_required or converged)
                ),
            }

            result.validation = validation
            self._log_stage("validation", f"completed (passed={validation['passed']})")
            return StageResult(
                name="validation", status="completed",
                duration_s=time.monotonic() - t0,
            )
        except Exception as e:
            logger.exception("Validation stage failed")
            return StageResult(
                name="validation", status="failed",
                duration_s=time.monotonic() - t0, error=str(e),
            )

    def _stage_report(self, result):
        t0 = time.monotonic()
        self._log_stage("report", "starting")
        try:
            from simulation.report import generate_report
            from simulation.scenario_tree import tree_to_dict

            ensemble_dict = (
                result.ensemble.to_dict() if result.ensemble else None
            )
            tree_dict = (
                tree_to_dict(result.scenario_tree)
                if result.scenario_tree else None
            )

            observed_data = None
            if self.config.outcome_criteria:
                obs = {
                    t.name: t.observed
                    for t in self.config.outcome_criteria.targets
                    if t.observed is not None
                }
                if obs:
                    observed_data = obs

            html = generate_report(
                ensemble_result=ensemble_dict,
                dynamics_analysis=result.dynamics,
                scenario_tree=tree_dict,
                observed_data=observed_data,
                title=f"dualmirakl Pipeline — {self.config.meta.name}",
            )
            result.report_html = html
            self._log_stage("report", "completed")
            return StageResult(
                name="report", status="completed",
                duration_s=time.monotonic() - t0,
            )
        except Exception as e:
            logger.exception("Report stage failed")
            return StageResult(
                name="report", status="failed",
                duration_s=time.monotonic() - t0, error=str(e),
            )

    # ── Metric extraction ─────────────────────────────────────────────────

    @staticmethod
    def _extract_metric(
        name: str, run_logs: list[list[float]]
    ) -> Optional[float]:
        """Extract a named metric from a single run's score logs.

        Domain-agnostic: computes everything from raw score trajectories.
        """
        if not run_logs:
            return None
        finals = [log[-1] for log in run_logs if log]
        if not finals:
            return None

        if name == "mean_score":
            return float(np.mean(finals))
        elif name == "score_std":
            return float(np.std(finals))
        elif name == "max_score":
            return float(np.max(finals))
        elif name == "min_score":
            return float(np.min(finals))
        elif name == "score_range":
            return float(np.max(finals) - np.min(finals))
        elif name == "polarization":
            from stats.core import polarization
            n_ticks = min(len(log) for log in run_logs)
            if n_ticks >= 2 and len(run_logs) >= 2:
                stances = np.array([log[:n_ticks] for log in run_logs]).T
                pol = polarization(stances)
                return pol.get("final_polarization", 0.0)
            return 0.0
        elif name == "convergence_ratio":
            from stats.validation import convergence_check
            mean_log = np.mean(run_logs, axis=0).tolist()
            if len(mean_log) >= 4:
                conv = convergence_check(np.array(mean_log))
                return 1.0 if conv.get("converged", False) else 0.0
            return 0.0
        elif name.startswith("fraction_above_"):
            # e.g. "fraction_above_0.7" -> threshold=0.7
            try:
                threshold = float(name.split("_")[-1])
                return float(np.mean([f >= threshold for f in finals]))
            except (ValueError, IndexError):
                pass
        # Unknown metric -> fallback
        logger.warning("Unknown metric '%s', defaulting to mean_score", name)
        return float(np.mean(finals))

    # ── Artifact saving ───────────────────────────────────────────────────

    def _save_artifacts(self, result: PipelineResult):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            (self.output_dir / "pipeline_result.json").write_text(
                json.dumps(result.to_dict(), indent=2, default=str)
            )

            if result.report_html:
                (self.output_dir / "report.html").write_text(result.report_html)

            if result.possibility_report:
                (self.output_dir / "possibility_report.json").write_text(
                    result.possibility_report.to_json()
                )

            if result.scenario_tree:
                from simulation.scenario_tree import tree_to_dict
                (self.output_dir / "scenario_tree.json").write_text(
                    json.dumps(tree_to_dict(result.scenario_tree), indent=2)
                )

            if result.validation:
                (self.output_dir / "validation.json").write_text(
                    json.dumps(result.validation, indent=2, default=str)
                )

            if result.dynamics:
                (self.output_dir / "dynamics_analysis.json").write_text(
                    json.dumps(result.dynamics, indent=2, default=str)
                )

            if result.statistics:
                (self.output_dir / "statistics.json").write_text(
                    json.dumps(result.statistics, indent=2, default=str)
                )

            logger.info("Pipeline artifacts saved to %s", self.output_dir)
        except Exception as e:
            logger.warning("Failed to save artifacts: %s", e)
