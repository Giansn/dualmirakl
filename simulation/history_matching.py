"""
history_matching.py — History Matching for parameter space reduction.

Implements the NROY (Not Ruled Out Yet) framework to narrow the parameter
space before expensive Sobol analysis. Based on Vernon et al. (2010) and
Andrianakis et al. (2015).

Workflow:
    1. Morris screening (sim_loop.py) → identify which parameters matter
    2. History Matching (this module) → eliminate implausible regions
    3. Sobol (sim_loop.py) → run on the narrowed NROY space only

Usage:
    from simulation.history_matching import HistoryMatcher

    # Define target patterns from empirical data or theoretical expectations
    targets = [
        PatternTarget("prevalence", 0.15, 0.25,
                       extractor=lambda ws: fraction_above(ws, 0.7)),
        PatternTarget("mean_score", 0.30, 0.50,
                       extractor=lambda ws: mean_final_score(ws)),
        PatternTarget("score_spread", 0.10, 0.30,
                       extractor=lambda ws: score_std(ws)),
    ]

    hm = HistoryMatcher(targets, bounds, n_waves=3, samples_per_wave=64)
    nroy_space = hm.run(sim_func)
    # nroy_space contains only non-implausible parameter combinations
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatternTarget:
    """
    A single target pattern for POM validation.

    The simulation output must fall within [low, high] to be considered
    non-implausible. Based on Pattern-Oriented Modeling (Grimm et al. 2005):
    validate by requiring simultaneous reproduction of multiple patterns.

    Args:
        name: human-readable identifier
        low: lower bound of acceptable range
        high: upper bound of acceptable range
        extractor: function(WorldState) -> float that computes the pattern
                   metric from simulation output
        tolerance: additional tolerance for implausibility measure
                   (accounts for observation error + model discrepancy)
    """
    name: str
    low: float
    high: float
    extractor: Callable
    tolerance: float = 0.0


@dataclass
class Wave:
    """Results from one wave of History Matching."""
    wave_id: int
    n_samples: int
    n_nroy: int
    nroy_fraction: float
    nroy_points: np.ndarray  # (n_nroy, n_params)
    implausibility_scores: np.ndarray  # (n_samples,)


@dataclass
class HistoryMatcher:
    """
    Iterative History Matching with multiple waves.

    Each wave:
    1. Sample parameter points from the current NROY space
    2. Run simulation for each point → extract pattern metrics
    3. Compute implausibility for each point against all targets
    4. Discard implausible points (implausibility > threshold)
    5. NROY space shrinks → next wave samples from reduced space

    The implausibility measure I(x) for parameter vector x:
        I(x) = max_j [ |f_j(x) - z_j| / sqrt(Var_j + sigma_j^2) ]
    where f_j(x) is the simulation output for target j, z_j is the
    target midpoint, Var_j is the target half-width, sigma_j is tolerance.

    Points with I(x) > threshold (default 3.0, "3-sigma rule") are
    ruled out as implausible.
    """
    targets: list[PatternTarget]
    bounds: list[tuple[float, float]]
    param_names: list[str] = field(default_factory=list)
    n_waves: int = 3
    samples_per_wave: int = 64
    threshold: float = 3.0
    seed: int = 42
    waves: list[Wave] = field(default_factory=list)

    def _sample_initial(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        """Latin Hypercube Sampling for the initial wave."""
        k = len(self.bounds)
        result = np.zeros((n, k))
        for j in range(k):
            lo, hi = self.bounds[j]
            # LHS: divide [0,1] into n equal strata, sample one from each
            cuts = np.linspace(0, 1, n + 1)
            points = rng.uniform(cuts[:-1], cuts[1:])
            rng.shuffle(points)
            result[:, j] = lo + points * (hi - lo)
        return result

    def _sample_nroy(
        self, n: int, nroy_points: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """Sample new points from the NROY region via perturbation."""
        k = len(self.bounds)
        n_nroy = len(nroy_points)
        result = np.zeros((n, k))

        for i in range(n):
            # Pick a random NROY point and perturb it
            base = nroy_points[rng.randint(n_nroy)].copy()
            for j in range(k):
                lo, hi = self.bounds[j]
                spread = (hi - lo) * 0.15  # 15% of range
                base[j] += rng.normal(0, spread)
                base[j] = np.clip(base[j], lo, hi)
            result[i] = base

        return result

    def _compute_implausibility(self, outputs: list[dict[str, float]]) -> np.ndarray:
        """
        Compute maximum implausibility across all targets for each sample.

        I(x) = max over targets [ |output - midpoint| / sqrt(halfwidth^2 + tolerance^2) ]
        """
        n = len(outputs)
        impl = np.zeros(n)

        for i, out in enumerate(outputs):
            max_impl = 0.0
            for target in self.targets:
                if target.name not in out:
                    max_impl = self.threshold + 1  # missing target → implausible
                    break
                value = out[target.name]
                midpoint = (target.low + target.high) / 2.0
                halfwidth = (target.high - target.low) / 2.0
                denom = np.sqrt(halfwidth ** 2 + target.tolerance ** 2)
                if denom < 1e-10:
                    denom = 1e-10
                target_impl = abs(value - midpoint) / denom
                max_impl = max(max_impl, target_impl)
            impl[i] = max_impl

        return impl

    def run(
        self,
        sim_func: Callable[[np.ndarray], dict[str, float]],
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Execute iterative History Matching.

        Args:
            sim_func: function(param_vector) -> dict mapping target names
                      to output values. For async simulations, wrap with
                      asyncio.run() or provide a synchronous summary function.
            verbose: print progress

        Returns:
            NROY parameter points (n_nroy, n_params)
        """
        rng = np.random.RandomState(self.seed)
        nroy_points = None

        for wave_id in range(1, self.n_waves + 1):
            if verbose:
                print(f"\n── History Matching Wave {wave_id}/{self.n_waves} ──")

            # Sample parameter points
            if nroy_points is None or len(nroy_points) < 3:
                points = self._sample_initial(self.samples_per_wave, rng)
                if verbose:
                    print(f"  Sampling {self.samples_per_wave} points (LHS)")
            else:
                points = self._sample_nroy(self.samples_per_wave, nroy_points, rng)
                if verbose:
                    print(f"  Sampling {self.samples_per_wave} points from NROY ({len(nroy_points)} bases)")

            # Evaluate simulation for each point
            outputs = []
            for i, x in enumerate(points):
                try:
                    out = sim_func(x)
                    outputs.append(out)
                except Exception as e:
                    logger.warning(f"  Wave {wave_id} sample {i} failed: {e}")
                    outputs.append({})  # will be marked implausible

            # Compute implausibility
            impl = self._compute_implausibility(outputs)

            # Filter NROY
            nroy_mask = impl <= self.threshold
            nroy_points = points[nroy_mask]
            nroy_frac = np.mean(nroy_mask)

            wave = Wave(
                wave_id=wave_id,
                n_samples=len(points),
                n_nroy=int(np.sum(nroy_mask)),
                nroy_fraction=float(nroy_frac),
                nroy_points=nroy_points,
                implausibility_scores=impl,
            )
            self.waves.append(wave)

            if verbose:
                print(f"  NROY: {wave.n_nroy}/{wave.n_samples} ({nroy_frac:.0%})")
                print(f"  Implausibility: min={impl.min():.2f} median={np.median(impl):.2f} max={impl.max():.2f}")
                if wave.n_nroy > 0:
                    for j, name in enumerate(self.param_names or [f"p{j}" for j in range(len(self.bounds))]):
                        lo, hi = nroy_points[:, j].min(), nroy_points[:, j].max()
                        orig_lo, orig_hi = self.bounds[j]
                        reduction = 1.0 - (hi - lo) / (orig_hi - orig_lo)
                        print(f"    {name:>15s}: [{lo:.3f}, {hi:.3f}] (reduced {reduction:.0%})")

            if wave.n_nroy < 3:
                if verbose:
                    print(f"  WARNING: NROY space nearly empty. Stopping.")
                break

        return nroy_points

    def summary(self) -> dict:
        """Return summary statistics from all waves."""
        return {
            "n_waves": len(self.waves),
            "waves": [
                {
                    "wave_id": w.wave_id,
                    "n_samples": w.n_samples,
                    "n_nroy": w.n_nroy,
                    "nroy_fraction": round(w.nroy_fraction, 3),
                }
                for w in self.waves
            ],
            "final_nroy_size": self.waves[-1].n_nroy if self.waves else 0,
            "param_names": self.param_names,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN EXTRACTORS — compute target metrics from simulation output
# ═══════════════════════════════════════════════════════════════════════════════

def fraction_above_threshold(
    final_scores: list[float], threshold: float = 0.7
) -> float:
    """Fraction of participants with final score above threshold (prevalence)."""
    if not final_scores:
        return 0.0
    return sum(1 for s in final_scores if s >= threshold) / len(final_scores)


def mean_final_score(final_scores: list[float]) -> float:
    if not final_scores:
        return 0.0
    return float(np.mean(final_scores))


def score_std(final_scores: list[float]) -> float:
    if len(final_scores) < 2:
        return 0.0
    return float(np.std(final_scores))


def score_range(final_scores: list[float]) -> float:
    if not final_scores:
        return 0.0
    return float(max(final_scores) - min(final_scores))


def trajectory_monotonicity(score_logs: list[list[float]]) -> float:
    """
    Average fraction of consecutive increases across all trajectories.
    High monotonicity = scores trend upward consistently.
    """
    if not score_logs:
        return 0.0
    monos = []
    for log in score_logs:
        if len(log) < 2:
            continue
        increases = sum(1 for i in range(1, len(log)) if log[i] > log[i - 1])
        monos.append(increases / (len(log) - 1))
    return float(np.mean(monos)) if monos else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE — default targets for media addiction simulation
# ═══════════════════════════════════════════════════════════════════════════════

def default_targets() -> list[PatternTarget]:
    """
    Default POM targets based on media addiction literature.

    Prevalence: 15-25% of adolescents show problematic use (Griffiths 2005,
    WHO ICD-11 prevalence estimates).
    Mean score: population average should be low-to-moderate (0.25-0.45).
    Spread: there should be meaningful individual differences (std 0.10-0.30).
    Monotonicity: trajectories should show mixed trends (0.3-0.7), not
    all-increasing or all-stable.
    """
    return [
        PatternTarget(
            "prevalence", 0.15, 0.25, tolerance=0.05,
            extractor=lambda scores: fraction_above_threshold(scores, 0.7),
        ),
        PatternTarget(
            "mean_score", 0.25, 0.45, tolerance=0.05,
            extractor=lambda scores: mean_final_score(scores),
        ),
        PatternTarget(
            "score_spread", 0.10, 0.30, tolerance=0.03,
            extractor=lambda scores: score_std(scores),
        ),
        PatternTarget(
            "monotonicity", 0.30, 0.70, tolerance=0.10,
            extractor=lambda logs: trajectory_monotonicity(logs),
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION — run HM with the synthetic SA objective from sim_loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_history_matching(
    n_waves: int = 3,
    samples_per_wave: int = 64,
    n_ticks: int = 12,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run History Matching using the synthetic SA objective (no vLLM needed).

    Uses the same simplified scoring model as sim_loop.run_sensitivity_analysis:
    sinusoidal engagement signal + noise, EMA/logistic update.

    Returns summary dict + NROY points.
    """
    import math
    from simulation.sim_loop import update_score

    bounds = [
        (0.1, 0.4),    # alpha
        (1.0, 6.0),    # K
        (0.60, 0.85),  # intervention threshold
        (0.3, 1.0),    # dampening
        (0.2, 1.0),    # susceptibility
        (0.0, 0.6),    # resilience
        (3.0, 10.0),   # logistic_k
    ]
    param_names = ["alpha", "K", "threshold", "dampening",
                   "susceptibility", "resilience", "logistic_k"]

    targets = default_targets()

    def sim_func(x: np.ndarray) -> dict[str, float]:
        alpha, k_float, threshold, dampening, susceptibility, resilience, logistic_k = x
        rng = np.random.RandomState(42)

        # Simulate a small cohort
        n_agents = 8
        score_logs = []
        final_scores = []

        for agent_i in range(n_agents):
            # Per-agent variation around the sampled susceptibility/resilience
            agent_sus = np.clip(susceptibility + rng.normal(0, 0.1), 0.05, 0.95)
            agent_res = np.clip(resilience + rng.normal(0, 0.05), 0.0, 0.8)
            score = rng.uniform(0.1, 0.5)
            log = []

            for t in range(n_ticks):
                signal = 0.5 + 0.3 * math.sin(t * 0.5 + agent_i * 0.3) + rng.normal(0, 0.1)
                signal = max(0.0, min(1.0, signal))
                d = dampening if (t % max(1, int(k_float)) == 0) else 1.0
                score = update_score(
                    score, signal, d, alpha,
                    mode="logistic", logistic_k=logistic_k,
                    susceptibility=agent_sus, resilience=agent_res,
                )
                log.append(score)

            score_logs.append(log)
            final_scores.append(score)

        return {
            "prevalence": fraction_above_threshold(final_scores, 0.7),
            "mean_score": mean_final_score(final_scores),
            "score_spread": score_std(final_scores),
            "monotonicity": trajectory_monotonicity(score_logs),
        }

    hm = HistoryMatcher(
        targets=targets,
        bounds=bounds,
        param_names=param_names,
        n_waves=n_waves,
        samples_per_wave=samples_per_wave,
        threshold=3.0,
        seed=seed,
    )

    nroy_points = hm.run(sim_func, verbose=verbose)

    result = hm.summary()
    result["nroy_points"] = nroy_points.tolist() if len(nroy_points) > 0 else []
    return result


if __name__ == "__main__":
    result = run_history_matching(n_waves=3, samples_per_wave=64, verbose=True)
    print(f"\n── Summary ──")
    print(f"  Final NROY size: {result['final_nroy_size']}")
    print(f"  Waves: {result['n_waves']}")
