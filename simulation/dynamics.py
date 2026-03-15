"""
dynamics.py — Coupled nonlinear dynamics, phase analysis, and chaos detection.

Three modules:

(A) Coupled ODE score dynamics — mean-field inter-agent coupling that creates
    emergent group behavior (synchronization, bifurcation, chaos).

(B) Phase portrait / bifurcation analysis — extract phase-space coordinates
    from score trajectories, sweep parameters to find qualitative transitions.

(C) Lyapunov exponent estimation — quantify sensitivity to initial conditions.
    Positive λ = chaotic, zero = periodic/quasiperiodic, negative = stable.

All functions work on score trajectories (lists of floats). No vLLM needed.
Can be used post-hoc on simulation output or inline during run_tick().
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# (A) COUPLED ODE SCORE DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Standard (uncoupled) update:
#   S_i(t+1) = S_i(t) + d·α·(signal_i - S_i)
#
# Coupled update adds a mean-field interaction term:
#   S_i(t+1) = S_i(t) + d·α·(signal_i - S_i) + κ·g(S̄ - S_i)
#
# where:
#   S̄  = population mean score (or local neighborhood mean)
#   κ  = coupling strength ∈ [0, 1]
#       κ=0: independent agents (current behavior)
#       κ>0: agents pulled toward group mean (conformity/peer pressure)
#       κ<0: agents pushed away from group mean (polarization)
#   g  = coupling function (linear, sigmoid, or threshold)
#
# This is analogous to:
#   - Kuramoto model (coupled oscillators, synchronization)
#   - Ising model (spin alignment, phase transitions)
#   - Vicsek model (flocking, collective motion)
#
# The coupling creates feedback loops that can produce:
#   - Synchronization: all agents converge to similar scores
#   - Bifurcation: population splits into distinct clusters
#   - Chaos: sensitive dependence on κ and initial conditions
# ═══════════════════════════════════════════════════════════════════════════════


def coupled_score_update(
    current: float,
    signal: float,
    population_scores: list[float],
    dampening: float = 1.0,
    alpha: float = 0.15,
    kappa: float = 0.0,
    coupling_mode: str = "linear",
    susceptibility: float = 1.0,
    resilience: float = 0.0,
    logistic_k: float = 6.0,
    score_mode: str = "ema",
) -> float:
    """
    Score update with inter-agent coupling.

    Extends update_score() with a coupling term that models peer influence:
    agents are pulled toward (or pushed away from) the population mean.

    Args:
        current: agent's current score
        signal: embedding-based behavioral signal
        population_scores: list of ALL agents' current scores (including self)
        dampening: intervention dampening factor d
        alpha: EMA learning rate
        kappa: coupling strength κ
            0.0 = independent (no peer influence)
            >0  = conformity (pulled toward group mean)
            <0  = polarization (pushed away from group mean)
        coupling_mode: "linear", "sigmoid", or "threshold"
        susceptibility: agent-specific signal scaling
        resilience: agent-specific change resistance
        logistic_k: steepness for logistic score mode
        score_mode: "ema" or "logistic"

    Returns:
        Updated score ∈ [0, 1]
    """
    # Individual update (same as sim_loop.update_score)
    effective_signal = current + susceptibility * (signal - current)

    if score_mode == "logistic":
        midpoint = 0.5
        effective_signal = 1.0 / (1.0 + np.exp(-logistic_k * (effective_signal - midpoint)))

    effective_dampening = dampening * (1.0 - resilience)
    individual_delta = alpha * (effective_signal - current)

    # Coupling term: g(S̄ - S_i)
    if kappa != 0.0 and len(population_scores) > 1:
        pop_mean = np.mean(population_scores)
        diff = pop_mean - current

        if coupling_mode == "linear":
            coupling_delta = kappa * diff
        elif coupling_mode == "sigmoid":
            # Saturating coupling: strong effect near mean, weak at extremes
            coupling_delta = kappa * np.tanh(3.0 * diff)
        elif coupling_mode == "threshold":
            # Only couple if difference exceeds threshold (0.1)
            coupling_delta = kappa * diff if abs(diff) > 0.1 else 0.0
        else:
            coupling_delta = 0.0
    else:
        coupling_delta = 0.0

    new_score = current + individual_delta * effective_dampening + coupling_delta
    return float(np.clip(new_score, 0.0, 1.0))


def compute_coupling_matrix(
    scores: list[float],
    mode: str = "mean_field",
) -> np.ndarray:
    """
    Compute the N×N coupling influence matrix.

    mean_field: uniform coupling (all agents influence each other equally)
    distance:   closer scores → stronger coupling (social similarity)
    """
    n = len(scores)
    s = np.array(scores)

    if mode == "mean_field":
        # Uniform: each agent influenced equally by all others
        W = np.ones((n, n)) / (n - 1) if n > 1 else np.ones((1, 1))
        np.fill_diagonal(W, 0)
    elif mode == "distance":
        # Distance-based: closer scores → stronger coupling
        diffs = np.abs(s[:, None] - s[None, :])
        similarity = np.exp(-5.0 * diffs)  # Gaussian kernel
        np.fill_diagonal(similarity, 0)
        row_sums = similarity.sum(axis=1, keepdims=True)
        W = similarity / (row_sums + 1e-8)
    else:
        W = np.zeros((n, n))

    return W


def coupled_batch_update(
    scores: list[float],
    signals: list[float],
    dampening: float = 1.0,
    alpha: float = 0.15,
    kappa: float = 0.0,
    coupling_mode: str = "linear",
    susceptibilities: Optional[list[float]] = None,
    resiliences: Optional[list[float]] = None,
    score_mode: str = "ema",
    logistic_k: float = 6.0,
) -> list[float]:
    """
    Batch coupled update for all agents simultaneously.

    Computes the mean-field coupling for each agent and updates all scores
    in one pass. Used in Phase C as a drop-in replacement for individual
    update_score() calls.

    Returns list of new scores.
    """
    n = len(scores)
    if susceptibilities is None:
        susceptibilities = [1.0] * n
    if resiliences is None:
        resiliences = [0.0] * n

    new_scores = []
    for i in range(n):
        new_scores.append(coupled_score_update(
            current=scores[i],
            signal=signals[i],
            population_scores=scores,
            dampening=dampening,
            alpha=alpha,
            kappa=kappa,
            coupling_mode=coupling_mode,
            susceptibility=susceptibilities[i],
            resilience=resiliences[i],
            score_mode=score_mode,
            logistic_k=logistic_k,
        ))
    return new_scores


# ═══════════════════════════════════════════════════════════════════════════════
# (B) PHASE PORTRAIT & BIFURCATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Phase space coordinates from a score trajectory S(t):
#   x = S(t)           — score (position)
#   y = ΔS(t)          — rate of change (velocity)
#   z = Δ²S(t)         — acceleration
#
# These reveal:
#   Fixed points:  trajectories that converge to (S*, 0, 0)
#   Limit cycles:  closed orbits in (S, ΔS) plane
#   Strange attractors: fractal structure → chaos
#
# Bifurcation diagram:
#   Sweep a parameter (e.g., α or κ), run the model to steady state,
#   record the final score distribution. Qualitative changes in the
#   distribution (splitting, merging, onset of oscillation) are bifurcations.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PhasePoint:
    """Single point in 3D phase space."""
    score: float
    velocity: float      # ΔS/Δt
    acceleration: float  # Δ²S/Δt²


def extract_phase_trajectory(score_log: list[float]) -> list[PhasePoint]:
    """
    Convert a score trajectory into phase-space coordinates.

    Returns list of (score, velocity, acceleration) tuples.
    First two points have no acceleration (need 3 points for Δ²).
    """
    if len(score_log) < 2:
        return [PhasePoint(s, 0.0, 0.0) for s in score_log]

    points = []
    for i in range(len(score_log)):
        s = score_log[i]
        # Velocity: forward difference (or backward at end)
        if i < len(score_log) - 1:
            v = score_log[i + 1] - score_log[i]
        else:
            v = score_log[i] - score_log[i - 1]
        # Acceleration: central difference (or one-sided at boundaries)
        if 0 < i < len(score_log) - 1:
            a = score_log[i + 1] - 2 * score_log[i] + score_log[i - 1]
        else:
            a = 0.0
        points.append(PhasePoint(s, v, a))

    return points


def find_fixed_points(
    score_logs: list[list[float]],
    tail_fraction: float = 0.25,
    velocity_threshold: float = 0.005,
) -> list[dict]:
    """
    Detect fixed points (attractors) from score trajectories.

    A fixed point is detected when an agent's score velocity drops
    below threshold for the final fraction of the trajectory.

    Returns list of {agent_idx, score, mean_velocity, is_fixed_point}.
    """
    results = []
    for idx, log in enumerate(score_logs):
        if len(log) < 4:
            continue
        tail_start = max(1, int(len(log) * (1 - tail_fraction)))
        tail = log[tail_start:]
        velocities = [abs(tail[i] - tail[i - 1]) for i in range(1, len(tail))]
        mean_v = np.mean(velocities) if velocities else 0.0
        results.append({
            "agent_idx": idx,
            "score": float(log[-1]),
            "mean_velocity": float(mean_v),
            "is_fixed_point": bool(mean_v < velocity_threshold),
        })
    return results


def bifurcation_sweep(
    param_name: str,
    param_values: list[float],
    base_params: dict,
    n_agents: int = 8,
    n_ticks: int = 50,
    transient: int = 20,
    seed: int = 42,
) -> dict:
    """
    Sweep a single parameter and record the steady-state score distribution.

    Uses the synthetic objective (no vLLM). For each parameter value, runs
    n_agents through n_ticks of coupled updates, discards the first `transient`
    ticks, and records the final score distribution.

    Args:
        param_name: key in base_params to sweep (e.g., "alpha", "kappa")
        param_values: list of values to sweep
        base_params: default parameter dict with keys:
            alpha, kappa, dampening, susceptibility, resilience,
            logistic_k, score_mode, coupling_mode
        n_agents: number of synthetic agents
        n_ticks: total ticks to simulate
        transient: ticks to discard before recording (let dynamics settle)
        seed: RNG seed

    Returns:
        {param_name, param_values, distributions: [[final_scores] per value],
         means, stds, n_clusters}
    """
    rng = np.random.RandomState(seed)
    distributions = []
    means = []
    stds = []
    n_clusters_list = []

    for pval in param_values:
        params = {**base_params, param_name: pval}

        # Initialize agents
        scores = [float(rng.uniform(0.1, 0.5)) for _ in range(n_agents)]
        susceptibilities = [float(rng.beta(2, 3)) for _ in range(n_agents)]
        resiliences = [float(rng.beta(2, 5)) for _ in range(n_agents)]

        # Run simulation
        for t in range(n_ticks):
            # Synthetic signals (sinusoidal + noise + per-agent phase offset)
            signals = [
                float(np.clip(
                    0.5 + 0.3 * math.sin(t * 0.5 + i * 0.7) + rng.normal(0, 0.1),
                    0.0, 1.0
                ))
                for i in range(n_agents)
            ]

            d = params.get("dampening", 1.0)
            scores = coupled_batch_update(
                scores=scores, signals=signals,
                dampening=d,
                alpha=params.get("alpha", 0.15),
                kappa=params.get("kappa", 0.0),
                coupling_mode=params.get("coupling_mode", "linear"),
                susceptibilities=susceptibilities,
                resiliences=resiliences,
                score_mode=params.get("score_mode", "ema"),
                logistic_k=params.get("logistic_k", 6.0),
            )

        # Record post-transient distribution
        distributions.append([round(s, 4) for s in scores])
        means.append(float(np.mean(scores)))
        stds.append(float(np.std(scores)))

        # Cluster detection (simple: count modes via histogram)
        hist, edges = np.histogram(scores, bins=5, range=(0, 1))
        n_clusters_list.append(int(np.sum(hist > 0)))

    return {
        "param_name": param_name,
        "param_values": [round(v, 4) for v in param_values],
        "distributions": distributions,
        "means": means,
        "stds": stds,
        "n_clusters": n_clusters_list,
    }


def detect_bifurcation_points(sweep_result: dict, std_jump_threshold: float = 0.03) -> list[dict]:
    """
    Detect bifurcation points from a sweep result.

    A bifurcation is flagged where:
    - std jumps significantly (population splits)
    - number of clusters changes
    - mean shifts abruptly
    """
    bifurcations = []
    vals = sweep_result["param_values"]
    stds = sweep_result["stds"]
    means = sweep_result["means"]
    clusters = sweep_result["n_clusters"]

    for i in range(1, len(vals)):
        std_jump = abs(stds[i] - stds[i - 1])
        mean_jump = abs(means[i] - means[i - 1])
        cluster_change = clusters[i] != clusters[i - 1]

        if std_jump > std_jump_threshold or cluster_change:
            bifurcations.append({
                "param_value": vals[i],
                "param_value_prev": vals[i - 1],
                "std_jump": round(std_jump, 4),
                "mean_jump": round(mean_jump, 4),
                "cluster_change": cluster_change,
                "clusters_before": clusters[i - 1],
                "clusters_after": clusters[i],
            })

    return bifurcations


# ═══════════════════════════════════════════════════════════════════════════════
# (C) LYAPUNOV EXPONENT ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# The maximal Lyapunov exponent λ measures the average exponential rate
# of divergence of nearby trajectories:
#
#   |δS(t)| ≈ |δS(0)| · e^{λt}
#
#   λ > 0: chaotic (exponential divergence)
#   λ = 0: marginal / periodic
#   λ < 0: stable (convergent)
#
# Method: Twin trajectory approach
#   1. Run the simulation with initial condition S₀
#   2. Run again with perturbed S₀ + ε (ε = 1e-8)
#   3. Track log(|S(t) - S'(t)|) / t over time
#   4. λ = lim_{t→∞} (1/t) Σ log(|S(t) - S'(t)| / |S(t-1) - S'(t-1)|)
#
# For coupled systems, estimate per-agent and population-level λ.
# ═══════════════════════════════════════════════════════════════════════════════


def lyapunov_exponent_twin(
    score_log: list[float],
    score_log_perturbed: list[float],
    dt: float = 1.0,
) -> float:
    """
    Estimate maximal Lyapunov exponent from twin trajectories.

    Args:
        score_log: original trajectory S(t)
        score_log_perturbed: perturbed trajectory S'(t)
        dt: time step (1.0 for discrete ticks)

    Returns:
        Estimated λ (positive = chaotic, negative = stable)
    """
    n = min(len(score_log), len(score_log_perturbed))
    if n < 2:
        return 0.0

    log_ratios = []
    for t in range(1, n):
        d_t = abs(score_log[t] - score_log_perturbed[t])
        d_prev = abs(score_log[t - 1] - score_log_perturbed[t - 1])

        if d_prev < 1e-15:
            continue  # skip if trajectories are identical
        if d_t < 1e-15:
            log_ratios.append(-30.0)  # effectively -∞, strong convergence
            continue

        log_ratios.append(math.log(d_t / d_prev) / dt)

    return float(np.mean(log_ratios)) if log_ratios else 0.0


def lyapunov_from_timeseries(
    score_log: list[float],
    embedding_dim: int = 3,
    tau: int = 1,
) -> float:
    """
    Estimate Lyapunov exponent from a single time series using
    Rosenstein's method (1993).

    Reconstructs the phase space via time-delay embedding, finds nearest
    neighbors, and tracks their divergence rate.

    Args:
        score_log: score trajectory (at least 20 points recommended)
        embedding_dim: dimension of reconstructed phase space (default 3)
        tau: time delay for embedding (default 1 tick)

    Returns:
        Estimated maximal Lyapunov exponent
    """
    N = len(score_log)
    M = N - (embedding_dim - 1) * tau  # number of embedded vectors

    if M < 10:
        return 0.0

    # Time-delay embedding: X_i = [S(i), S(i+τ), S(i+2τ), ...]
    X = np.array([
        [score_log[i + j * tau] for j in range(embedding_dim)]
        for i in range(M)
    ])

    # Find nearest neighbor for each point (exclude temporal neighbors)
    min_separation = embedding_dim * tau + 1
    divergence = []

    for i in range(M):
        min_dist = float("inf")
        nn_idx = -1
        for j in range(M):
            if abs(i - j) < min_separation:
                continue
            dist = np.linalg.norm(X[i] - X[j])
            if 0 < dist < min_dist:
                min_dist = dist
                nn_idx = j

        if nn_idx < 0 or min_dist < 1e-15:
            continue

        # Track divergence over time
        max_steps = min(M - i, M - nn_idx) - 1
        for k in range(1, min(max_steps, 10)):
            d_k = abs(score_log[i + k] - score_log[nn_idx + k])
            if d_k > 1e-15:
                divergence.append((k, math.log(d_k / min_dist)))

    if not divergence:
        return 0.0

    # Linear regression of log(divergence) vs time step → slope = λ
    steps = np.array([d[0] for d in divergence], dtype=float)
    log_divs = np.array([d[1] for d in divergence])

    # Least squares: λ = (Σ t·ln(d) - n·t̄·ln(d)̄) / (Σ t² - n·t̄²)
    n = len(steps)
    t_mean = np.mean(steps)
    d_mean = np.mean(log_divs)
    numerator = np.sum(steps * log_divs) - n * t_mean * d_mean
    denominator = np.sum(steps ** 2) - n * t_mean ** 2

    if abs(denominator) < 1e-15:
        return 0.0

    return float(numerator / denominator)


def estimate_system_lyapunov(
    score_logs: list[list[float]],
    method: str = "timeseries",
    perturbed_logs: Optional[list[list[float]]] = None,
) -> dict:
    """
    Estimate Lyapunov exponents for the full agent population.

    Returns:
        {per_agent: [λ_0, λ_1, ...], population_mean: float,
         max_lyapunov: float, regime: "chaotic"|"stable"|"marginal"}
    """
    per_agent = []

    if method == "twin" and perturbed_logs is not None:
        for log, plog in zip(score_logs, perturbed_logs):
            per_agent.append(lyapunov_exponent_twin(log, plog))
    else:
        for log in score_logs:
            per_agent.append(lyapunov_from_timeseries(log))

    pop_mean = float(np.mean(per_agent)) if per_agent else 0.0
    max_lya = float(max(per_agent)) if per_agent else 0.0

    if max_lya > 0.05:
        regime = "chaotic"
    elif max_lya < -0.05:
        regime = "stable"
    else:
        regime = "marginal"

    return {
        "per_agent": [round(l, 4) for l in per_agent],
        "population_mean": round(pop_mean, 4),
        "max_lyapunov": round(max_lya, 4),
        "regime": regime,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# (D) SOBOL SECOND-ORDER INTERACTION INDICES
# ═══════════════════════════════════════════════════════════════════════════════
#
# First-order S_i measures individual parameter contributions.
# Second-order S_ij measures the INTERACTION effect — variance explained
# by the joint effect of parameters i and j that is NOT captured by
# their individual effects.
#
#   S_ij = Var(E[Y|X_i, X_j]) / Var(Y) - S_i - S_j
#
#   S_ij >> 0: synergistic interaction — combined effect exceeds sum of parts
#   S_ij ≈ 0: additive — parameters act independently
#   S_ij < 0:  antagonistic — combined effect less than sum
#
# This quantifies whether the system produces outcomes that neither
# parameter predicts alone (superadditivity).
# ═══════════════════════════════════════════════════════════════════════════════


def sobol_second_order(
    func,
    bounds: list[tuple[float, float]],
    param_names: Optional[list[str]] = None,
    n_samples: int = 1024,
    seed: int = 42,
) -> dict:
    """
    Compute Sobol first-order AND second-order interaction indices.

    Uses the Saltelli (2002) estimator. Cost: N(2k+2) evaluations.

    Returns:
        {
            "S1": {param_name: float},         # first-order
            "S2": {(name_i, name_j): float},   # second-order interactions
            "ST": {param_name: float},          # total-order
        }
    """
    rng = np.random.RandomState(seed)
    k = len(bounds)
    names = param_names or [f"p{i}" for i in range(k)]

    def _sample(n):
        raw = rng.uniform(size=(n, k))
        return np.array([
            [lo + raw[j, i] * (hi - lo) for i, (lo, hi) in enumerate(bounds)]
            for j in range(n)
        ])

    A = _sample(n_samples)
    B = _sample(n_samples)
    y_A = np.array([func(A[j]) for j in range(n_samples)])
    y_B = np.array([func(B[j]) for j in range(n_samples)])
    f0_sq = np.mean(y_A) * np.mean(y_B)
    var_y = np.var(np.concatenate([y_A, y_B]))

    if var_y < 1e-12:
        return {
            "S1": {names[i]: 0.0 for i in range(k)},
            "S2": {},
            "ST": {names[i]: 0.0 for i in range(k)},
        }

    # First-order and total-order indices
    S1 = {}
    ST = {}
    y_AB = {}  # cache AB_i evaluations for S2

    for i in range(k):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        y_AB[i] = np.array([func(AB_i[j]) for j in range(n_samples)])

        # S1: Jansen estimator
        s1 = float(np.mean(y_B * (y_AB[i] - y_A)) / var_y)
        S1[names[i]] = max(0.0, min(1.0, s1))

        # ST: total-order (1 - S~i)
        BA_i = B.copy()
        BA_i[:, i] = A[:, i]
        y_BA_i = np.array([func(BA_i[j]) for j in range(n_samples)])
        st = float(0.5 * np.mean((y_A - y_BA_i) ** 2) / var_y)
        ST[names[i]] = max(0.0, min(1.0, st))

    # Second-order interaction indices
    S2 = {}
    for i in range(k):
        for j in range(i + 1, k):
            # AB_ij: take A, replace columns i AND j with B
            AB_ij = A.copy()
            AB_ij[:, i] = B[:, i]
            AB_ij[:, j] = B[:, j]
            y_AB_ij = np.array([func(AB_ij[m]) for m in range(n_samples)])

            # S_ij = V_ij / V(Y) - S_i - S_j
            v_ij = float(np.mean(y_B * (y_AB_ij - y_A)) / var_y)
            s_ij = v_ij - S1[names[i]] - S1[names[j]]
            S2[(names[i], names[j])] = round(s_ij, 4)

    return {"S1": S1, "S2": S2, "ST": ST}


# ═══════════════════════════════════════════════════════════════════════════════
# (E) TRANSFER ENTROPY
# ═══════════════════════════════════════════════════════════════════════════════
#
# Transfer entropy T_{X→Y} measures the directed information flow from
# process X to process Y:
#
#   T_{X→Y} = H(Y_t | Y_{t-1}^{(k)}) - H(Y_t | Y_{t-1}^{(k)}, X_{t-1}^{(l)})
#
# where H is conditional Shannon entropy and k, l are history lengths.
#
#   T > 0: X provides information about Y's future beyond Y's own past
#          (the coupling is informative — the interaction carries signal)
#   T = 0: X and Y are conditionally independent given Y's past
#
# In the simulation context:
#   T_{environment→participant}: how much does knowing the platform state
#     reduce uncertainty about the user's FUTURE score?
#   T_{participant_i→participant_j}: directed peer influence (with coupling)
#
# Implementation: histogram-based discrete entropy estimation.
# No external dependencies (pyunicorn/dit not needed).
# ═══════════════════════════════════════════════════════════════════════════════


def _discretize(series: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Discretize a continuous series into n_bins equal-width bins."""
    mn, mx = np.min(series), np.max(series)
    if mx - mn < 1e-10:
        return np.zeros(len(series), dtype=int)
    return np.clip(
        ((series - mn) / (mx - mn) * n_bins).astype(int),
        0, n_bins - 1
    )


def _joint_entropy(*arrays: np.ndarray) -> float:
    """Compute joint Shannon entropy H(X, Y, ...) from discretized arrays."""
    # Combine arrays into joint state tuples
    n = len(arrays[0])
    states = np.column_stack(arrays)
    # Count unique joint states
    _, counts = np.unique(states, axis=0, return_counts=True)
    probs = counts / n
    return float(-np.sum(probs * np.log2(probs + 1e-15)))


def _conditional_entropy(target: np.ndarray, *conditions: np.ndarray) -> float:
    """Compute H(target | conditions) = H(target, conditions) - H(conditions)."""
    h_joint = _joint_entropy(target, *conditions)
    h_cond = _joint_entropy(*conditions) if conditions else 0.0
    return h_joint - h_cond


def transfer_entropy(
    source: list[float],
    target: list[float],
    k: int = 1,
    l: int = 1,
    n_bins: int = 8,
) -> float:
    """
    Compute transfer entropy T_{source→target}.

    T = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-l})

    Args:
        source: time series X (e.g., environment stimuli or another agent's scores)
        target: time series Y (e.g., participant scores)
        k: target history length (how many past Y values to condition on)
        l: source history length (how many past X values to include)
        n_bins: discretization bins

    Returns:
        Transfer entropy in bits. T > 0 means source → target information flow.
    """
    n = min(len(source), len(target))
    delay = max(k, l)
    if n <= delay + 1:
        return 0.0

    src = np.array(source[:n])
    tgt = np.array(target[:n])
    src_d = _discretize(src, n_bins)
    tgt_d = _discretize(tgt, n_bins)

    # Build lagged arrays
    y_t = tgt_d[delay:]  # target at time t
    y_hist = np.column_stack([tgt_d[delay - i - 1: n - i - 1] for i in range(k)])  # Y_{t-1:t-k}
    x_hist = np.column_stack([src_d[delay - i - 1: n - i - 1] for i in range(l)])  # X_{t-1:t-l}

    # T = H(Y_t | Y_hist) - H(Y_t | Y_hist, X_hist)
    h_y_given_yhist = _conditional_entropy(y_t, *[y_hist[:, i] for i in range(k)])
    h_y_given_both = _conditional_entropy(
        y_t,
        *[y_hist[:, i] for i in range(k)],
        *[x_hist[:, i] for i in range(l)]
    )

    te = h_y_given_yhist - h_y_given_both
    return float(max(0.0, te))  # clip negative noise


def transfer_entropy_matrix(
    score_logs: list[list[float]],
    k: int = 1,
    l: int = 1,
    n_bins: int = 8,
) -> np.ndarray:
    """
    Compute pairwise transfer entropy matrix for all agents.

    Returns N×N matrix where TE[i,j] = T_{agent_i → agent_j}.
    Diagonal is zero (no self-transfer).
    Asymmetric: TE[i,j] ≠ TE[j,i] in general (directed information flow).
    """
    n = len(score_logs)
    te_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            te_matrix[i, j] = transfer_entropy(
                score_logs[i], score_logs[j], k=k, l=l, n_bins=n_bins
            )

    return te_matrix


def net_information_flow(te_matrix: np.ndarray) -> dict:
    """
    Summarize directed information flow from the TE matrix.

    Returns:
        {
            "total_te": float,          # sum of all TE values
            "mean_te": float,           # average TE
            "max_pair": (int, int),     # most informative directed pair
            "max_te": float,            # TE of that pair
            "net_flow": [float],        # per agent: outgoing - incoming TE
                                        #   positive = net information source
                                        #   negative = net information sink
            "asymmetry": float,         # mean |TE[i,j] - TE[j,i]| — directedness
        }
    """
    n = te_matrix.shape[0]
    total = float(np.sum(te_matrix))
    n_pairs = n * (n - 1) if n > 1 else 1
    mean = total / n_pairs

    max_idx = np.unravel_index(np.argmax(te_matrix), te_matrix.shape)

    # Net flow per agent: outgoing - incoming
    outgoing = np.sum(te_matrix, axis=1)  # row sums = what i sends
    incoming = np.sum(te_matrix, axis=0)  # col sums = what i receives
    net = outgoing - incoming

    # Asymmetry: mean directedness
    asym_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            asym_vals.append(abs(te_matrix[i, j] - te_matrix[j, i]))
    asymmetry = float(np.mean(asym_vals)) if asym_vals else 0.0

    return {
        "total_te": round(total, 4),
        "mean_te": round(mean, 4),
        "max_pair": (int(max_idx[0]), int(max_idx[1])),
        "max_te": round(float(te_matrix[max_idx]), 4),
        "net_flow": [round(float(f), 4) for f in net],
        "asymmetry": round(asymmetry, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# (F) EMERGENCE INDEX
# ═══════════════════════════════════════════════════════════════════════════════
#
# Measures how much population-level structure exceeds what individual
# trajectories predict independently.
#
# Three complementary metrics:
#
# 1. Variance ratio:
#      E_var = 1 - Var(population_mean_trajectory) / mean(Var(individual_trajectories))
#      E_var > 0: population behaves differently than individuals predict
#      E_var = 0: population = average of independent agents
#
# 2. Mutual information excess:
#      E_mi = I(X_1; X_2; ...; X_N) - sum(H(X_i))
#      Measures shared structure beyond individual entropy
#
# 3. Correlation dimension:
#      Ratio of multi-agent phase space dimensionality to single-agent
#      dimensionality. Lower ratio = more collective structure.
#
# ═══════════════════════════════════════════════════════════════════════════════


def emergence_variance_ratio(score_logs: list[list[float]]) -> float:
    """
    Variance-based emergence index.

    Compares the variance of the population mean trajectory against the
    mean variance of individual trajectories.

    E = 1 - Var(mean_trajectory) / mean(Var(individual))

    E > 0: population-level smoothing — agents partially cancel out,
           indicating correlated behavior that produces emergent regularity.
    E ≈ 0: no emergence — population = sum of independent parts.
    E < 0: super-variance — population fluctuates MORE than individuals,
           indicating amplifying feedback (possible chaos).
    """
    if not score_logs or len(score_logs) < 2:
        return 0.0

    # Align lengths
    min_len = min(len(log) for log in score_logs)
    if min_len < 3:
        return 0.0

    logs = np.array([log[:min_len] for log in score_logs])  # (N, T)

    # Population mean trajectory
    pop_mean = np.mean(logs, axis=0)  # (T,)
    var_pop = np.var(pop_mean)

    # Mean individual variance
    individual_vars = np.var(logs, axis=1)  # (N,)
    mean_ind_var = np.mean(individual_vars)

    if mean_ind_var < 1e-12:
        return 0.0

    return float(1.0 - var_pop / mean_ind_var)


def emergence_mutual_information(
    score_logs: list[list[float]],
    n_bins: int = 8,
) -> float:
    """
    Information-theoretic emergence index.

    Computes the total correlation (multi-information):
      TC = sum(H(X_i)) - H(X_1, X_2, ..., X_N)

    TC > 0: agents share information — their joint state is more
            structured than independent agents would produce.
    TC = 0: agents are statistically independent.

    Higher TC indicates stronger emergent coupling.
    Normalized by sum(H(X_i)) to give a value in [0, 1].
    """
    if not score_logs or len(score_logs) < 2:
        return 0.0

    min_len = min(len(log) for log in score_logs)
    if min_len < 10:
        return 0.0

    # Use the final portion of trajectories (post-transient)
    tail_start = max(0, min_len - min(min_len, 30))
    tails = [np.array(log[tail_start:min_len]) for log in score_logs]

    # Discretize each agent's trajectory
    discretized = [_discretize(t, n_bins) for t in tails]

    # Sum of individual entropies
    sum_h = sum(_joint_entropy(d) for d in discretized)

    if sum_h < 1e-10:
        return 0.0

    # Joint entropy H(X_1, ..., X_N)
    h_joint = _joint_entropy(*discretized)

    # Total correlation = sum(H_i) - H_joint
    tc = sum_h - h_joint

    # Normalize to [0, 1]
    return float(np.clip(tc / sum_h, 0.0, 1.0))


def emergence_score_clustering(
    final_scores: list[float],
    n_expected_clusters: int = 1,
) -> float:
    """
    Structural emergence index based on score distribution clustering.

    Measures whether the final score distribution shows more structure
    (modes, clusters) than expected from independent agents.

    Uses the gap between observed cluster count and expected.
    Also computes bimodality coefficient (BC):
      BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)(n-3)))
      BC > 5/9 ≈ 0.555 suggests bimodality (population splitting).
    """
    if len(final_scores) < 4:
        return 0.0

    s = np.array(final_scores)
    n = len(s)

    # Bimodality coefficient
    mean = np.mean(s)
    std = np.std(s)
    if std < 1e-10:
        return 0.0

    skew = float(np.mean(((s - mean) / std) ** 3))
    kurt = float(np.mean(((s - mean) / std) ** 4))

    # Adjusted BC (Pfister et al. 2013)
    bc_num = skew ** 2 + 1
    bc_den = kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3)) if n > 3 else kurt + 3.0
    bc = bc_num / bc_den if bc_den > 0 else 0.0

    return float(bc)


def compute_emergence(
    score_logs: list[list[float]],
    final_scores: Optional[list[float]] = None,
) -> dict:
    """
    Compute all emergence metrics for a simulation run.

    Returns:
        {
            "variance_ratio": float,   # population vs individual variance
            "mutual_information": float, # total correlation (normalized)
            "bimodality": float,        # bimodality coefficient
            "is_emergent": bool,        # any metric above threshold
        }
    """
    if final_scores is None:
        final_scores = [log[-1] for log in score_logs if log]

    e_var = emergence_variance_ratio(score_logs)
    e_mi = emergence_mutual_information(score_logs)
    e_bc = emergence_score_clustering(final_scores)

    return {
        "variance_ratio": round(e_var, 4),
        "mutual_information": round(e_mi, 4),
        "bimodality": round(e_bc, 4),
        "is_emergent": bool(e_mi > 0.05 or e_bc > 0.555 or abs(e_var) > 0.3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("\n== (A) Coupled Score Dynamics ==\n")

    # Demonstrate coupling effect
    rng = np.random.RandomState(42)
    n_agents, n_ticks = 6, 30
    for kappa_val in [0.0, 0.1, 0.3]:
        scores = [float(rng.uniform(0.1, 0.5)) for _ in range(n_agents)]
        for t in range(n_ticks):
            signals = [float(np.clip(0.5 + 0.3 * math.sin(t * 0.5 + i * 0.7) + rng.normal(0, 0.1), 0, 1))
                       for i in range(n_agents)]
            scores = coupled_batch_update(scores, signals, kappa=kappa_val)
        spread = max(scores) - min(scores)
        print(f"  kappa={kappa_val:.1f}: final scores={[f'{s:.3f}' for s in scores]}  spread={spread:.3f}")

    print("\n== (B) Bifurcation Sweep (kappa) ==\n")

    sweep = bifurcation_sweep(
        "kappa",
        [round(k * 0.05, 2) for k in range(0, 11)],
        base_params={"alpha": 0.15, "dampening": 1.0, "susceptibility": 0.5,
                     "resilience": 0.1, "logistic_k": 6.0, "score_mode": "ema",
                     "coupling_mode": "linear"},
    )
    for i, k in enumerate(sweep["param_values"]):
        print(f"  kappa={k:.2f}: mean={sweep['means'][i]:.3f} "
              f"std={sweep['stds'][i]:.3f} clusters={sweep['n_clusters'][i]}")

    bifs = detect_bifurcation_points(sweep)
    if bifs:
        print(f"\n  Bifurcation points:")
        for b in bifs:
            print(f"    kappa={b['param_value']:.2f}: std_jump={b['std_jump']:.3f} "
                  f"clusters {b['clusters_before']}→{b['clusters_after']}")

    print("\n== (C) Lyapunov Exponent Estimation ==\n")

    # Generate trajectories with different coupling strengths
    for kappa_val in [0.0, 0.2, 0.5]:
        rng = np.random.RandomState(42)
        logs = []
        scores = [float(rng.uniform(0.1, 0.5)) for _ in range(4)]
        for _ in range(4):
            logs.append([])
        for t in range(60):
            signals = [float(np.clip(0.5 + 0.3 * math.sin(t * 0.5 + i * 0.7) + rng.normal(0, 0.1), 0, 1))
                       for i in range(4)]
            scores = coupled_batch_update(scores, signals, kappa=kappa_val, alpha=0.15)
            for i, s in enumerate(scores):
                logs[i].append(s)

        result = estimate_system_lyapunov(logs)
        print(f"  kappa={kappa_val:.1f}: max_lambda={result['max_lyapunov']:.4f}  "
              f"regime={result['regime']}  per_agent={result['per_agent']}")

    print("\n== (D) Sobol S2 Interaction Indices ==\n")

    from simulation.sim_loop import update_score

    def objective(x):
        alpha, dampening, susceptibility, resilience = x
        rng_obj = np.random.RandomState(42)
        score = 0.3
        for t in range(20):
            signal = float(np.clip(0.5 + 0.3 * math.sin(t * 0.5) + rng_obj.normal(0, 0.1), 0, 1))
            score = update_score(score, signal, dampening, alpha,
                                 susceptibility=susceptibility, resilience=resilience)
        return score

    s2_result = sobol_second_order(
        objective,
        bounds=[(0.1, 0.4), (0.3, 1.0), (0.2, 1.0), (0.0, 0.6)],
        param_names=["alpha", "dampening", "susceptibility", "resilience"],
        n_samples=256,
    )
    print("  First-order (S1):")
    for name, val in s2_result["S1"].items():
        print(f"    {name:>15s}: {val:.4f}")
    print("  Interactions (S2):")
    for pair, val in sorted(s2_result["S2"].items(), key=lambda x: -abs(x[1])):
        marker = " ***" if abs(val) > 0.01 else ""
        print(f"    {str(pair):>40s}: {val:+.4f}{marker}")

    print("\n== (E) Transfer Entropy ==\n")

    rng = np.random.RandomState(42)
    logs_te = []
    scores_te = [float(rng.uniform(0.1, 0.5)) for _ in range(4)]
    for _ in range(4):
        logs_te.append([])
    for t in range(60):
        signals = [float(np.clip(0.5 + 0.3 * math.sin(t * 0.5 + i * 0.7) + rng.normal(0, 0.1), 0, 1))
                   for i in range(4)]
        scores_te = coupled_batch_update(scores_te, signals, kappa=0.2, alpha=0.15)
        for i, s in enumerate(scores_te):
            logs_te[i].append(s)

    te_mat = transfer_entropy_matrix(logs_te)
    flow = net_information_flow(te_mat)
    print(f"  Total TE: {flow['total_te']}  Mean: {flow['mean_te']}  Asymmetry: {flow['asymmetry']}")
    print(f"  Max pair: agent_{flow['max_pair'][0]}→agent_{flow['max_pair'][1]} = {flow['max_te']} bits")
    print(f"  Net flow: {flow['net_flow']}  (positive=source, negative=sink)")

    print("\n== (F) Emergence Index ==\n")

    for kappa_val in [0.0, 0.2, 0.5]:
        rng = np.random.RandomState(42)
        logs_em = []
        scores_em = [float(rng.uniform(0.1, 0.5)) for _ in range(6)]
        for _ in range(6):
            logs_em.append([])
        for t in range(40):
            signals = [float(np.clip(0.5 + 0.3 * math.sin(t * 0.5 + i * 0.7) + rng.normal(0, 0.1), 0, 1))
                       for i in range(6)]
            scores_em = coupled_batch_update(scores_em, signals, kappa=kappa_val, alpha=0.15)
            for i, s in enumerate(scores_em):
                logs_em[i].append(s)

        em = compute_emergence(logs_em)
        print(f"  kappa={kappa_val:.1f}: var_ratio={em['variance_ratio']:.3f}  "
              f"MI={em['mutual_information']:.3f}  bimodality={em['bimodality']:.3f}  "
              f"emergent={em['is_emergent']}")
