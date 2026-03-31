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
class KeyFinding:
    """A single actionable finding from the simulation."""
    icon: str               # emoji/symbol for UI
    text: str               # plain-language finding
    severity: str = "info"  # "info" | "warning" | "critical"


@dataclass
class PossibilityReport:
    """Complete structured output: ranked possibility branches."""
    run_id: str
    timestamp: str
    n_branches: int
    branches: list[PossibilityBranch]
    metadata: ReportMetadata
    warnings: list[str] = field(default_factory=list)
    summary: str = ""                                        # executive summary
    key_findings: list[KeyFinding] = field(default_factory=list)
    risk_assessment: dict = field(default_factory=dict)      # threshold-based risks

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


@dataclass
class DomainVocabulary:
    """Domain-specific labels for score ranges and outcomes."""
    domain: str = "generic"
    subject: str = "agents"           # "adolescents", "traders", "nodes"
    score_noun: str = "engagement"    # "compulsive usage", "risk exposure"
    low_label: str = "Self-regulated"
    mid_label: str = "Moderate engagement"
    high_label: str = "Compulsive patterns"
    polarized_label: str = "Polarized: resilient vs. at-risk"
    low_desc: str = "maintain healthy self-regulation"
    mid_desc: str = "show moderate, variable engagement"
    high_desc: str = "develop compulsive patterns"
    threshold_label: str = "concerning"  # label for score > 0.7


_DOMAIN_VOCABS = {
    "social_media_adolescent": DomainVocabulary(
        domain="social_media_adolescent",
        subject="adolescents",
        score_noun="usage intensity",
        low_label="Healthy self-regulation",
        mid_label="Moderate usage with occasional overuse",
        high_label="Compulsive usage patterns",
        polarized_label="Split: self-regulated vs. compulsive users",
        low_desc="maintain healthy boundaries with social media",
        mid_desc="show variable engagement with intermittent overuse episodes",
        high_desc="develop compulsive usage patterns requiring intervention",
        threshold_label="compulsive",
    ),
    "misinformation": DomainVocabulary(
        domain="misinformation",
        subject="participants",
        score_noun="belief polarization",
        low_label="Grounded consensus",
        mid_label="Moderate susceptibility",
        high_label="Deep misinformation adoption",
        polarized_label="Echo chamber bifurcation",
        low_desc="maintain evidence-based beliefs",
        mid_desc="show susceptibility to misleading content but self-correct",
        high_desc="adopt and amplify misinformation narratives",
        threshold_label="radicalized",
    ),
    "market": DomainVocabulary(
        domain="market",
        subject="traders",
        score_noun="herd behavior intensity",
        low_label="Independent trading",
        mid_label="Moderate herding",
        high_label="Full herd behavior",
        polarized_label="Market bifurcation: contrarians vs. herd",
        low_desc="trade independently based on fundamentals",
        mid_desc="show intermittent herding with some independent decisions",
        high_desc="follow the herd, amplifying market volatility",
        threshold_label="herding",
    ),
}

_DOMAIN_KEYWORDS = {
    "social_media_adolescent": ["adolescent", "social media", "screen time", "compulsive", "scrolling", "media usage", "teenager", "digital wellbeing"],
    "misinformation": ["misinformation", "belief", "echo chamber", "fake news", "polarization", "disinformation"],
    "market": ["trader", "market", "herd", "stock", "financial", "portfolio"],
}


def infer_domain_vocabulary(world_context: str = "") -> DomainVocabulary:
    """Match world context keywords to a built-in domain vocabulary."""
    if not world_context:
        return DomainVocabulary()
    ctx_lower = world_context.lower()
    best_domain, best_score = "generic", 0
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in ctx_lower)
        if score > best_score:
            best_domain, best_score = domain, score
    if best_score >= 2:
        logger.info("Domain vocabulary: %s (matched %d keywords)", best_domain, best_score)
        return _DOMAIN_VOCABS[best_domain]
    return DomainVocabulary()


def _generate_label(center: float, basin_size: float, is_bimodal: bool,
                    n_basins: int = 1, vocab: DomainVocabulary = None) -> str:
    """Generate a domain-specific label for a branch."""
    v = vocab or DomainVocabulary()
    if is_bimodal and n_basins > 1:
        return v.polarized_label
    if center < 0.3:
        return v.low_label
    elif center < 0.7:
        return v.mid_label
    else:
        return v.high_label


def _generate_narrative(
    center: float,
    basin_range: tuple[float, float],
    metrics: BranchMetrics,
    levers: list[ParameterLever],
    probability: float = 1.0,
    n_agents_in_basin: int = 0,
    n_agents_total: int = 0,
    vocab: DomainVocabulary = None,
) -> str:
    """Deterministic domain-aware narrative. No LLM involved."""
    v = vocab or DomainVocabulary()
    parts = []
    pct = f"{probability*100:.0f}%"
    agent_frac = f"{n_agents_in_basin} of {n_agents_total} {v.subject}" if n_agents_total > 0 else v.subject

    if center < 0.3:
        parts.append(f"{pct} probability: {agent_frac} {v.low_desc} (score near {center:.2f}).")
    elif center > 0.7:
        parts.append(f"{pct} probability: {agent_frac} {v.high_desc} (score near {center:.2f}).")
    else:
        parts.append(f"{pct} probability: {agent_frac} {v.mid_desc} (score near {center:.2f}).")

    width = basin_range[1] - basin_range[0]
    if width > 0.6:
        parts.append("This outcome is robust across a wide range of starting conditions.")
    elif width < 0.2:
        parts.append("This outcome is sensitive to initial conditions.")

    if metrics.lyapunov_regime == "chaotic":
        parts.append("Individual trajectories are unpredictable (chaotic regime).")
    elif not metrics.convergence:
        parts.append(f"{v.score_noun.capitalize()} levels are still drifting — longer runs may shift this outcome.")

    if metrics.polarization > 0.04:
        parts.append(f"Subgroups are forming within this outcome (polarization={metrics.polarization:.3f}).")

    if levers:
        top = levers[0]
        parts.append(f"Key lever: '{top.name}' (sensitivity={top.sensitivity:.3f}).")

    return " ".join(parts)


def _analyze_trajectories(
    score_logs: list[list[float]],
    interventions: list[dict],
    n_ticks: int,
    vocab: DomainVocabulary,
    config: dict,
) -> tuple[str, list[KeyFinding], dict]:
    """Produce executive summary, key findings, and risk assessment from raw data."""
    n_agents = len(score_logs)
    v = vocab
    findings: list[KeyFinding] = []

    # ── Per-agent analysis ──
    finals = [log[-1] for log in score_logs if log]
    initials = [log[0] for log in score_logs if log]
    group_mean = float(np.mean(finals)) if finals else 0.0
    group_init = float(np.mean(initials)) if initials else 0.0
    group_delta = group_mean - group_init

    # Peak scores per agent
    peaks = [max(log) for log in score_logs if log]
    peak_agents = [i for i, log in enumerate(score_logs) if log and max(log) >= 0.7]
    sustained_high = []  # agents with score > 0.7 for 3+ consecutive ticks
    for i, log in enumerate(score_logs):
        run = 0
        for s in log:
            if s >= 0.7:
                run += 1
                if run >= 3:
                    sustained_high.append(i)
                    break
            else:
                run = 0

    # Trend: linear slope per agent (simple regression)
    slopes = []
    for log in score_logs:
        if len(log) >= 4:
            x = np.arange(len(log))
            slope = float(np.polyfit(x, log, 1)[0])
            slopes.append(slope)
        else:
            slopes.append(0.0)

    rising = [i for i, s in enumerate(slopes) if s > 0.002]
    falling = [i for i, s in enumerate(slopes) if s < -0.002]
    stable = [i for i in range(n_agents) if i not in rising and i not in falling]

    # Projected ticks to reach 0.7 for rising agents not yet there
    projections = {}
    for i in rising:
        if finals[i] < 0.7 and slopes[i] > 0:
            ticks_to_07 = (0.7 - finals[i]) / slopes[i]
            projections[i] = int(ticks_to_07)

    # Susceptibility categories
    susceptibilities = []
    for log in score_logs:
        if len(log) >= 2:
            max_jump = max(abs(log[j+1] - log[j]) for j in range(len(log)-1))
            susceptibilities.append(max_jump)
        else:
            susceptibilities.append(0.0)

    # ── Executive summary ──
    summary_parts = []
    duration_desc = f"{n_ticks}-tick simulation of {n_agents} {v.subject}"
    summary_parts.append(f"In this {duration_desc}:")

    if sustained_high:
        n_sh = len(sustained_high)
        summary_parts.append(
            f"{n_sh} of {n_agents} {v.subject} reached {v.threshold_label} levels "
            f"(score >{chr(160)}0.7 sustained for 3+ ticks)."
        )
    elif peak_agents:
        n_pa = len(peak_agents)
        summary_parts.append(
            f"{n_pa} of {n_agents} {v.subject} briefly crossed the {v.threshold_label} threshold "
            f"but did not sustain it."
        )
    else:
        summary_parts.append(
            f"No {v.subject} reached the {v.threshold_label} threshold (0.7)."
        )

    if group_delta > 0.05:
        summary_parts.append(
            f"Group {v.score_noun} drifted upward from {group_init:.2f} to {group_mean:.2f} "
            f"(+{group_delta:.2f}), suggesting gradual escalation."
        )
    elif group_delta < -0.05:
        summary_parts.append(
            f"Group {v.score_noun} declined from {group_init:.2f} to {group_mean:.2f} "
            f"({group_delta:.2f}), indicating de-escalation."
        )
    else:
        summary_parts.append(
            f"Group {v.score_noun} remained stable around {group_mean:.2f}."
        )

    if projections:
        nearest = min(projections.items(), key=lambda x: x[1])
        agent_id, ticks_left = nearest
        summary_parts.append(
            f"At current rates, P{agent_id} is projected to reach {v.threshold_label} levels "
            f"in ~{ticks_left} additional ticks."
        )

    summary = " ".join(summary_parts)

    # ── Key findings ──
    # 1. Overall trend
    if len(rising) > len(falling):
        findings.append(KeyFinding(
            icon="\u2197",
            text=f"{len(rising)} of {n_agents} {v.subject} show upward {v.score_noun} trends "
                 f"(avg slope +{np.mean([slopes[i] for i in rising]):.4f}/tick).",
            severity="warning" if len(rising) > n_agents // 2 else "info",
        ))
    elif len(falling) > len(rising):
        findings.append(KeyFinding(
            icon="\u2198",
            text=f"{len(falling)} of {n_agents} {v.subject} show declining {v.score_noun}.",
            severity="info",
        ))

    # 2. Convergence / divergence
    score_spread = float(np.std(finals)) if len(finals) > 1 else 0.0
    if score_spread < 0.05:
        findings.append(KeyFinding(
            icon="\u2261",
            text=f"Strong group convergence (std={score_spread:.3f}): {v.subject} are "
                 f"settling on similar behavior patterns.",
            severity="info",
        ))
    elif score_spread > 0.15:
        findings.append(KeyFinding(
            icon="\u2194",
            text=f"Group is diverging (std={score_spread:.3f}): subgroups forming "
                 f"with distinct {v.score_noun} levels.",
            severity="warning",
        ))

    # 3. Threshold crossings
    if sustained_high:
        findings.append(KeyFinding(
            icon="\u26a0",
            text=f"P{', P'.join(str(i) for i in sustained_high)} sustained {v.threshold_label} "
                 f"levels for 3+ ticks — this meets the concern threshold.",
            severity="critical",
        ))
    elif peak_agents:
        findings.append(KeyFinding(
            icon="\u26a1",
            text=f"P{', P'.join(str(i) for i in peak_agents)} spiked above 0.7 "
                 f"but recovered — volatile, not yet {v.threshold_label}.",
            severity="warning",
        ))

    # 4. Intervention effectiveness
    if interventions:
        n_iv = len(interventions)
        # Check if scores dropped after interventions
        effective = 0
        for iv in interventions:
            tick = iv.get("activated_at", 0)
            if tick > 0 and tick < n_ticks:
                post_scores = [log[min(tick + 2, len(log)-1)] for log in score_logs if len(log) > tick]
                pre_scores = [log[tick - 1] for log in score_logs if len(log) > tick]
                if post_scores and pre_scores and np.mean(post_scores) < np.mean(pre_scores):
                    effective += 1
        if effective > 0:
            findings.append(KeyFinding(
                icon="\u2714",
                text=f"{effective} of {n_iv} interventions showed measurable effect "
                     f"(group score decreased after trigger).",
                severity="info",
            ))
        else:
            findings.append(KeyFinding(
                icon="\u2718",
                text=f"{n_iv} intervention(s) triggered but none produced measurable score "
                     f"reduction — consider earlier or stronger interventions.",
                severity="warning",
            ))
    else:
        if group_mean > 0.5:
            findings.append(KeyFinding(
                icon="\u2718",
                text=f"No interventions triggered despite group mean of {group_mean:.2f} — "
                     f"intervention thresholds may be set too high.",
                severity="warning",
            ))

    # 5. Projections
    if projections:
        for agent_id, ticks_left in sorted(projections.items(), key=lambda x: x[1]):
            findings.append(KeyFinding(
                icon="\u23f1",
                text=f"P{agent_id} projected to reach {v.threshold_label} threshold in "
                     f"~{ticks_left} ticks at current trajectory.",
                severity="warning" if ticks_left < n_ticks else "info",
            ))

    # ── Risk assessment ──
    risk = {
        "threshold_crossed": len(sustained_high) > 0,
        "n_at_risk": len(peak_agents),
        "n_sustained_high": len(sustained_high),
        "group_mean": round(group_mean, 3),
        "group_trend": "rising" if group_delta > 0.03 else "falling" if group_delta < -0.03 else "stable",
        "group_delta": round(group_delta, 3),
        "score_spread": round(score_spread, 3),
        "projections": {f"P{k}": v for k, v in projections.items()},
        "intervention_count": len(interventions),
        "risk_level": "critical" if sustained_high else "elevated" if peak_agents or group_mean > 0.6 else "moderate" if group_mean > 0.4 else "low",
    }

    return summary, findings, risk


def compute_possibility_report(
    score_logs: list[list[float]],
    config: dict,
    run_id: str = "unknown",
    multi_run_logs: Optional[list[list[list[float]]]] = None,
    world_context: str = "",
    interventions: Optional[list[dict]] = None,
) -> PossibilityReport:
    """
    Main entry point. Call after simulation completes.

    Args:
        score_logs: per-agent score trajectories [agent][tick]
        config: run parameters (alpha, kappa, dampening, susceptibility, resilience, etc.)
        run_id: identifier for this run
        multi_run_logs: optional list of score_logs from repeated runs (for bootstrap CI)
        world_context: uploaded domain context text (for domain-specific labels)
        interventions: list of intervention dicts from the run
    """
    t0 = time.monotonic()
    warnings: list[str] = []
    n_agents = len(score_logs)
    n_ticks = max((len(log) for log in score_logs), default=0)
    vocab = infer_domain_vocabulary(world_context)

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

    lyapunov_threshold = config.get("lyapunov_threshold", 0.05)
    lyapunov_result = estimate_system_lyapunov(score_logs)
    lyapunov_max = lyapunov_result.get("max_lyapunov", 0.0)
    # Override regime using configurable threshold
    if lyapunov_max > lyapunov_threshold:
        regime = "chaotic"
    elif lyapunov_max < -lyapunov_threshold:
        regime = "stable"
    else:
        regime = "marginal"

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
                               n_basins=len(attractors), vocab=vocab)
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
            probability=raw_prob,
            n_agents_in_basin=n_in_basin,
            n_agents_total=n_agents,
            vocab=vocab,
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

    # ── Step 8: Executive summary + key findings ─────────────────────────
    _interventions = interventions or []
    summary, key_findings, risk_assessment = _analyze_trajectories(
        score_logs, _interventions, n_ticks, vocab, config,
    )

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
        summary=summary,
        key_findings=key_findings,
        risk_assessment=risk_assessment,
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
