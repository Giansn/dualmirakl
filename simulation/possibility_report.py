"""
Structured simulation output: possibility branches sorted by probability.

Composes attractor basins, bifurcation analysis, sensitivity indices,
and convergence checks from dynamics.py + stats/ into a ranked report
of divergent outcome trajectories.

Each branch represents a distinct attractor basin with:
  - Probability estimate (basin_size blended with empirical agent distribution)
  - Human-readable narrative (template-based, deterministic, auditable)
  - Key metrics (polarization, convergence, Lyapunov regime, emergence)
  - Parameter levers (which params push toward/away from this outcome)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ParameterLever:
    """A parameter that pushes the system toward or away from a branch."""
    name: str
    direction: str              # "toward" or "away"
    sensitivity: float          # Sobol S1 or Morris mu_star (0-1)
    current_value: float
    threshold: Optional[float] = None  # bifurcation point, if detected


@dataclass
class BranchMetrics:
    """Key metrics characterizing a possibility branch."""
    mean_final_score: float
    score_std: float
    n_agents_in_basin: int
    polarization: float
    convergence: bool
    lyapunov_regime: str        # "stable" | "marginal" | "chaotic"
    emergence_index: float


@dataclass
class PossibilityBranch:
    """A single divergent outcome trajectory."""
    branch_id: int
    label: str
    probability: float          # 0.0 - 1.0
    narrative: str
    metrics: BranchMetrics
    attractor_center: float
    basin_range: tuple[float, float]
    levers: list[ParameterLever] = field(default_factory=list)
    confidence_interval: Optional[tuple[float, float]] = None


@dataclass
class ReportMetadata:
    """Provenance: what computation produced this report."""
    n_ticks: int
    n_agents: int
    scoring_mode: str
    attractor_method: str = "map_attractor_basins"
    sensitivity_method: str = "none"
    multi_run: bool = False
    n_runs: int = 1
    lyapunov_method: str = "timeseries"
    computation_time_s: float = 0.0


@dataclass
class PossibilityReport:
    """Complete structured output: ranked possibility branches."""
    run_id: str
    timestamp: str
    n_branches: int
    branches: list[PossibilityBranch]
    metadata: ReportMetadata
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ── Computation Pipeline ─────────────────────────────────────────────────────

def _classify_agents_to_basins(
    score_logs: list[list[float]],
    attractors: list,
) -> dict[int, list[int]]:
    """Map each agent to the basin their final score falls into."""
    assignments: dict[int, list[int]] = {}
    for agent_idx, log in enumerate(score_logs):
        if not log:
            continue
        final = log[-1]
        best = min(range(len(attractors)),
                   key=lambda i: abs(final - attractors[i].center))
        assignments.setdefault(best, []).append(agent_idx)
    return assignments


def _generate_label(center: float, basin_size: float, is_bimodal: bool,
                    n_basins: int = 1) -> str:
    """Generate a short label for a branch."""
    if is_bimodal and n_basins > 1:
        return "Polarized Split"
    if center < 0.3:
        return "Low-Score Equilibrium"
    elif center < 0.5:
        return "Mid-Low Consensus"
    elif center < 0.7:
        return "Mid-High Consensus"
    else:
        return "High-Score Lock-in"


def _generate_narrative(
    center: float,
    basin_range: tuple[float, float],
    metrics: BranchMetrics,
    levers: list[ParameterLever],
) -> str:
    """Deterministic template-based narrative. No LLM involved."""
    parts = []

    if center < 0.3:
        parts.append(f"Agents settle into a low-score equilibrium near {center:.2f}.")
    elif center > 0.7:
        parts.append(f"Agents converge toward a high-score state near {center:.2f}.")
    else:
        parts.append(f"Agents stabilize around the mid-range ({center:.2f}).")

    width = basin_range[1] - basin_range[0]
    if width > 0.6:
        parts.append("This outcome is robust — a wide range of starting conditions lead here.")
    elif width < 0.2:
        parts.append("This outcome is fragile — it requires specific initial conditions.")

    if metrics.lyapunov_regime == "chaotic":
        parts.append("Dynamics within this basin are chaotic: exact trajectories are unpredictable.")
    elif not metrics.convergence:
        parts.append("Agents have not fully converged; scores are still drifting.")

    if metrics.polarization > 0.04:
        parts.append(f"Significant spread within this outcome (polarization={metrics.polarization:.3f}), suggesting subgroups.")

    if levers:
        top = levers[0]
        verb = "increasing" if top.direction == "toward" else "decreasing"
        parts.append(f"Most influential parameter: '{top.name}' (S1={top.sensitivity:.3f}); {verb} it pushes {top.direction} this outcome.")

    return " ".join(parts)


def compute_possibility_report(
    score_logs: list[list[float]],
    config: dict,
    run_id: str = "unknown",
    multi_run_logs: Optional[list[list[list[float]]]] = None,
) -> PossibilityReport:
    """
    Main entry point. Call after simulation completes.

    Args:
        score_logs: per-agent score trajectories [agent][tick]
        config: run parameters (alpha, kappa, dampening, susceptibility, resilience, etc.)
        run_id: identifier for this run
        multi_run_logs: optional list of score_logs from repeated runs (for bootstrap CI)
    """
    t0 = time.monotonic()
    warnings: list[str] = []
    n_agents = len(score_logs)
    n_ticks = max((len(log) for log in score_logs), default=0)

    alpha = config.get("alpha", 0.15)
    kappa = config.get("kappa", 0.0)
    dampening = config.get("dampening", 1.0)
    score_mode = config.get("score_mode", "ema")
    logistic_k = config.get("logistic_k", 6.0)

    # Mean susceptibility/resilience for basin mapping
    susceptibility = config.get("susceptibility", 0.4)
    resilience = config.get("resilience", 0.29)

    # ── Step 1: Attractor Basin Mapping ──────────────────────────────────
    from simulation.dynamics import map_attractor_basins, Attractor

    basins = map_attractor_basins(
        alpha=alpha, dampening=dampening,
        susceptibility=susceptibility, resilience=resilience,
        kappa=kappa, score_mode=score_mode, logistic_k=logistic_k,
        n_grid=100, n_ticks=80,
    )
    attractors: list[Attractor] = basins.get("attractors", [])

    if not attractors:
        # Single attractor fallback — compute from data directly
        all_final = [log[-1] for log in score_logs if log]
        attractors = [Attractor(
            center=float(np.mean(all_final)),
            basin_low=0.0, basin_high=1.0,
            basin_size=1.0, n_converged=len(all_final),
            stability=float(np.std(all_final)),
        )]
        warnings.append("No distinct attractor basins detected; reporting single outcome.")

    # ── Step 2: Classify agents to basins ────────────────────────────────
    assignments = _classify_agents_to_basins(score_logs, attractors)

    # Blend theoretical basin_size with empirical agent fraction
    converged = n_ticks >= 30
    w_theory = 0.4 if converged else 0.7
    w_empirical = 1.0 - w_theory

    # ── Step 3: Lyapunov + Convergence per branch ────────────────────────
    from simulation.dynamics import estimate_system_lyapunov, compute_emergence

    lyapunov_result = estimate_system_lyapunov(score_logs)
    regime = lyapunov_result.get("regime", "marginal")

    if lyapunov_result.get("sample_size_warning"):
        warnings.append(lyapunov_result["sample_size_warning"])

    emergence = compute_emergence(score_logs)

    # ── Step 4: Polarization per branch ──────────────────────────────────
    from stats.core import polarization as compute_polarization

    # ── Step 5: Build branches ───────────────────────────────────────────
    branches: list[PossibilityBranch] = []

    for basin_idx, attractor in enumerate(attractors):
        agent_indices = assignments.get(basin_idx, [])
        n_in_basin = len(agent_indices)
        empirical_frac = n_in_basin / n_agents if n_agents > 0 else 0.0
        raw_prob = w_theory * attractor.basin_size + w_empirical * empirical_frac

        # Chaotic regime: redistribute probability mass
        if regime == "chaotic":
            raw_prob *= 0.7  # keep 70%, redistribute 30% later

        # Per-branch metrics
        if agent_indices and n_ticks > 1:
            branch_logs = [score_logs[i] for i in agent_indices]
            branch_finals = [log[-1] for log in branch_logs if log]
            min_len = min(len(log) for log in branch_logs)
            stances = np.array([log[:min_len] for log in branch_logs]).T  # (T, N_branch)

            if stances.shape[1] >= 2 and stances.shape[0] >= 2:
                pol_result = compute_polarization(stances)
                pol_val = pol_result["final_polarization"]
                is_bimodal = pol_result["is_bimodal"]
            else:
                pol_val = 0.0
                is_bimodal = False

            # Convergence: check if tail variance is low
            from stats.validation import convergence_check
            mean_log = np.mean([log for log in branch_logs], axis=0)
            conv = convergence_check(mean_log, window=min(20, len(mean_log) // 3 + 1))
            converged_branch = conv.get("converged", False)
        else:
            branch_finals = [attractor.center]
            pol_val = 0.0
            is_bimodal = False
            converged_branch = False

        label = _generate_label(attractor.center, attractor.basin_size, is_bimodal,
                               n_basins=len(attractors))
        metrics = BranchMetrics(
            mean_final_score=float(np.mean(branch_finals)) if branch_finals else attractor.center,
            score_std=float(np.std(branch_finals)) if len(branch_finals) > 1 else 0.0,
            n_agents_in_basin=n_in_basin,
            polarization=pol_val,
            convergence=converged_branch,
            lyapunov_regime=regime,
            emergence_index=emergence.get("mutual_information", 0.0),
        )

        narrative = _generate_narrative(
            attractor.center,
            (attractor.basin_low, attractor.basin_high),
            metrics, [],
        )

        branches.append(PossibilityBranch(
            branch_id=basin_idx,
            label=label,
            probability=raw_prob,
            narrative=narrative,
            metrics=metrics,
            attractor_center=attractor.center,
            basin_range=(attractor.basin_low, attractor.basin_high),
        ))

    # ── Step 6: Bootstrap CI from multi-run data ─────────────────────────
    n_runs = 1
    if multi_run_logs and len(multi_run_logs) >= 3:
        from stats.validation import bootstrap_ci
        n_runs = len(multi_run_logs)
        per_branch_fracs = []
        for run_logs in multi_run_logs:
            run_assignments = _classify_agents_to_basins(run_logs, attractors)
            fracs = [len(run_assignments.get(i, [])) / len(run_logs)
                     for i in range(len(attractors))]
            per_branch_fracs.append(fracs)

        for branch_idx, branch in enumerate(branches):
            values = np.array([run[branch_idx] for run in per_branch_fracs
                               if branch_idx < len(run)])
            if len(values) >= 3:
                ci = bootstrap_ci(values, n_bootstrap=5000, alpha=0.1)
                branch.confidence_interval = (ci["ci_lower"], ci["ci_upper"])
    else:
        if len(branches) > 1:
            warnings.append("Single-run data: no bootstrap CI. Run multiple seeds for confidence intervals.")

    # ── Step 7: Normalize probabilities ──────────────────────────────────
    total = sum(b.probability for b in branches)
    if total > 0:
        for b in branches:
            b.probability = b.probability / total

    # Sort by probability descending
    branches.sort(key=lambda b: b.probability, reverse=True)

    computation_time = time.monotonic() - t0

    return PossibilityReport(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        n_branches=len(branches),
        branches=branches,
        metadata=ReportMetadata(
            n_ticks=n_ticks,
            n_agents=n_agents,
            scoring_mode=score_mode,
            multi_run=n_runs > 1,
            n_runs=n_runs,
            computation_time_s=round(computation_time, 2),
        ),
        warnings=warnings,
    )


# ── CLI Rendering ────────────────────────────────────────────────────────────

def render_cli(report: PossibilityReport) -> str:
    """Render the possibility report as a CLI-friendly string."""
    lines = []
    lines.append("")
    lines.append(f"  POSSIBILITY BRANCHES  ({report.n_branches} outcomes)")
    lines.append(f"  run: {report.run_id}")
    lines.append("  " + "\u2500" * 68)

    for b in report.branches:
        pct = b.probability * 100
        bar_len = 50
        filled = int(pct / 100 * bar_len)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        ci_str = ""
        if b.confidence_interval:
            ci_str = f"  CI: [{b.confidence_interval[0]*100:.0f}%-{b.confidence_interval[1]*100:.0f}%]"

        lines.append("")
        lines.append(f"  [{b.branch_id}] {b.label}   {pct:.1f}%{ci_str}")
        lines.append(f"      |{bar}|")
        lines.append(f"      attractor: {b.attractor_center:.3f}   "
                     f"basin: [{b.basin_range[0]:.2f}, {b.basin_range[1]:.2f}]   "
                     f"regime: {b.metrics.lyapunov_regime}")

        # Word-wrap narrative to ~70 chars
        words = b.narrative.split()
        line = "      "
        for w in words:
            if len(line) + len(w) + 1 > 74:
                lines.append(line)
                line = "      " + w
            else:
                line += (" " if line.strip() else "") + w
        if line.strip():
            lines.append(line)

        if b.levers:
            top = b.levers[0]
            lines.append(f"      lever: {top.name}={top.current_value} "
                         f"(S1={top.sensitivity:.3f}, {top.direction})")

    if report.warnings:
        lines.append("")
        lines.append("  WARNINGS:")
        for w in report.warnings:
            lines.append(f"    \u2022 {w}")

    lines.append("")
    run_type = "multi-run" if report.metadata.multi_run else "single-run"
    lines.append(f"  computed in {report.metadata.computation_time_s:.1f}s "
                 f"| {report.metadata.attractor_method} | {run_type}")
    lines.append("  " + "\u2500" * 68)
    return "\n".join(lines)
