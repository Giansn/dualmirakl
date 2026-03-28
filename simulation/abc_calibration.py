"""
ABC-SMC (Approximate Bayesian Computation — Sequential Monte Carlo) for dualmirakl.

Likelihood-free Bayesian inference for calibrating simulation parameters
against observed data. Produces posterior distributions over parameters
without requiring an explicit likelihood function.

Implements Toni et al. (2009) ABC-SMC with adaptive epsilon thresholds
and optimal Gaussian perturbation kernels.

Usage:
    from simulation.abc_calibration import abc_smc, ABCPrior

    priors = [
        ABCPrior("alpha", "uniform", low=0.05, high=0.4),
        ABCPrior("kappa", "uniform", low=0.0, high=0.2),
    ]
    observed = {"mean_score": 0.55, "score_std": 0.12}

    result = abc_smc(sim_func, priors, observed, n_particles=200)
    print(result.posterior_samples.mean(axis=0))  # posterior mean
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ABCPrior:
    """Single parameter prior specification."""
    name: str
    type: str = "uniform"      # "uniform", "normal", "beta"
    low: float = 0.0
    high: float = 1.0
    mean: float = 0.5          # for normal
    std: float = 0.1           # for normal
    alpha: float = 2.0         # for beta
    beta: float = 2.0          # for beta

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Draw n samples from this prior."""
        if self.type == "uniform":
            return rng.uniform(self.low, self.high, n)
        elif self.type == "normal":
            return rng.normal(self.mean, self.std, n)
        elif self.type == "beta":
            return rng.beta(self.alpha, self.beta, n)
        raise ValueError(f"Unknown prior type: {self.type}")

    def log_pdf(self, x: float) -> float:
        """Log probability density at x."""
        if self.type == "uniform":
            if self.low <= x <= self.high:
                return -math.log(self.high - self.low)
            return -float("inf")
        elif self.type == "normal":
            return -0.5 * ((x - self.mean) / self.std) ** 2 - math.log(self.std * math.sqrt(2 * math.pi))
        elif self.type == "beta":
            from scipy.stats import beta as beta_dist
            return float(beta_dist.logpdf(x, self.alpha, self.beta))
        return 0.0

    def in_support(self, x: float) -> bool:
        """Check if x is within the prior support."""
        if self.type == "uniform":
            return self.low <= x <= self.high
        elif self.type == "normal":
            return True  # unbounded
        elif self.type == "beta":
            return 0.0 <= x <= 1.0
        return True


@dataclass
class ABCPopulation:
    """One SMC population (generation)."""
    generation: int
    particles: np.ndarray       # (n_accepted, n_params)
    weights: np.ndarray         # (n_accepted,)
    distances: np.ndarray       # (n_accepted,)
    epsilon: float              # acceptance threshold used
    acceptance_rate: float
    ess: float                  # effective sample size
    n_simulations: int = 0     # total sims attempted


@dataclass
class ABCResult:
    """Result of ABC-SMC calibration."""
    posterior_samples: np.ndarray    # (n_accepted, n_params)
    weights: np.ndarray
    distances: np.ndarray
    acceptance_rate: float
    n_populations: int
    populations: list[ABCPopulation] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)
    computation_time_s: float = 0.0

    def posterior_mean(self) -> np.ndarray:
        """Weighted posterior mean."""
        return np.average(self.posterior_samples, weights=self.weights, axis=0)

    def posterior_std(self) -> np.ndarray:
        """Weighted posterior standard deviation."""
        mean = self.posterior_mean()
        var = np.average((self.posterior_samples - mean) ** 2, weights=self.weights, axis=0)
        return np.sqrt(var)

    def to_dict(self) -> dict:
        return {
            "n_populations": self.n_populations,
            "n_accepted": len(self.posterior_samples),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "computation_time_s": round(self.computation_time_s, 2),
            "param_names": self.param_names,
            "posterior_mean": [round(float(v), 6) for v in self.posterior_mean()],
            "posterior_std": [round(float(v), 6) for v in self.posterior_std()],
            "populations": [
                {
                    "generation": p.generation,
                    "epsilon": round(p.epsilon, 6),
                    "acceptance_rate": round(p.acceptance_rate, 4),
                    "ess": round(p.ess, 1),
                    "n_simulations": p.n_simulations,
                }
                for p in self.populations
            ],
        }


def default_distance(simulated: dict, observed: dict) -> float:
    """Weighted Euclidean distance over shared metric keys."""
    shared = set(simulated) & set(observed)
    if not shared:
        return float("inf")
    diffs = [(simulated[k] - observed[k]) ** 2 for k in shared]
    return math.sqrt(sum(diffs) / len(diffs))


def abc_smc(
    sim_func: Callable[[np.ndarray], dict[str, float]],
    priors: list[ABCPrior],
    observed: dict[str, float],
    distance_func: Optional[Callable] = None,
    n_particles: int = 100,
    n_populations: int = 5,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.01,
    epsilon_quantile: float = 0.5,
    perturbation_scale: float = 2.0,
    max_attempts_per_particle: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> ABCResult:
    """
    ABC-SMC calibration (Toni et al. 2009).

    Produces a weighted posterior sample over parameters by iteratively
    refining acceptance thresholds.

    Args:
        sim_func: Callable taking parameter vector (np.ndarray of shape (n_params,))
            and returning dict of {metric_name: value}.
        priors: Prior distributions for each parameter.
        observed: Observed data to calibrate against.
        distance_func: Distance function (simulated, observed) -> float.
            Default: weighted Euclidean.
        n_particles: Number of accepted particles per population.
        n_populations: Number of SMC populations (generations).
        initial_epsilon: Initial acceptance threshold.
        min_epsilon: Stop when epsilon drops below this.
        epsilon_quantile: Fraction of distances to use as next epsilon.
        perturbation_scale: Multiplier for perturbation kernel covariance.
        max_attempts_per_particle: Max simulation attempts per particle.
        seed: RNG seed.
        verbose: Log progress.

    Returns:
        ABCResult with posterior samples and diagnostics.
    """
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)
    n_params = len(priors)
    dist_fn = distance_func or default_distance
    param_names = [p.name for p in priors]
    populations: list[ABCPopulation] = []

    # ── Population 0: sample from prior ──────────────────────────────────
    particles_0 = []
    distances_0 = []
    n_sims_0 = 0

    while len(particles_0) < n_particles:
        theta = np.array([p.sample(rng, 1)[0] for p in priors])
        n_sims_0 += 1
        try:
            output = sim_func(theta)
            d = dist_fn(output, observed)
        except Exception as e:
            logger.debug("Sim failed for theta=%s: %s", theta, e)
            d = float("inf")

        if d < initial_epsilon:
            particles_0.append(theta)
            distances_0.append(d)

        if n_sims_0 > n_particles * max_attempts_per_particle:
            logger.warning("Population 0: max attempts reached (%d sims, %d accepted)",
                         n_sims_0, len(particles_0))
            break

    if len(particles_0) == 0:
        logger.error("Population 0: no particles accepted. Increase initial_epsilon.")
        return ABCResult(
            posterior_samples=np.empty((0, n_params)),
            weights=np.empty(0),
            distances=np.empty(0),
            acceptance_rate=0.0,
            n_populations=0,
            param_names=param_names,
            computation_time_s=time.monotonic() - t0,
        )

    particles_arr = np.array(particles_0)
    distances_arr = np.array(distances_0)
    weights = np.ones(len(particles_0)) / len(particles_0)

    ess = 1.0 / np.sum(weights ** 2)
    populations.append(ABCPopulation(
        generation=0,
        particles=particles_arr,
        weights=weights,
        distances=distances_arr,
        epsilon=initial_epsilon,
        acceptance_rate=len(particles_0) / max(n_sims_0, 1),
        ess=ess,
        n_simulations=n_sims_0,
    ))

    if verbose:
        logger.info("ABC Pop 0: %d/%d accepted (eps=%.4f, ESS=%.0f)",
                   len(particles_0), n_sims_0, initial_epsilon, ess)

    # ── Populations 1..T: resample, perturb, accept ──────────────────────
    for t in range(1, n_populations):
        prev = populations[-1]
        prev_particles = prev.particles
        prev_weights = prev.weights

        # Adaptive epsilon: quantile of previous distances
        epsilon_t = float(np.quantile(prev.distances, epsilon_quantile))
        epsilon_t = max(epsilon_t, min_epsilon)

        # Perturbation kernel: Gaussian with twice the weighted covariance
        cov = np.cov(prev_particles.T, aweights=prev_weights)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        elif cov.ndim == 1:
            cov = np.diag(cov)
        cov *= perturbation_scale

        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals <= 0):
            cov += np.eye(n_params) * (abs(min(eigvals)) + 1e-6)

        new_particles = []
        new_distances = []
        new_weights_raw = []
        n_sims_t = 0

        while len(new_particles) < n_particles:
            # Resample from previous population
            idx = rng.choice(len(prev_particles), p=prev_weights)
            theta_star = prev_particles[idx]

            # Perturb
            theta = rng.multivariate_normal(theta_star, cov)

            # Check prior support
            in_support = all(p.in_support(theta[i]) for i, p in enumerate(priors))
            if not in_support:
                n_sims_t += 1
                if n_sims_t > n_particles * max_attempts_per_particle:
                    break
                continue

            # Simulate
            n_sims_t += 1
            try:
                output = sim_func(theta)
                d = dist_fn(output, observed)
            except Exception:
                d = float("inf")

            if d < epsilon_t:
                # Importance weight: prior / weighted kernel density
                log_prior = sum(p.log_pdf(theta[i]) for i, p in enumerate(priors))
                kernel_density = 0.0
                for j, prev_theta in enumerate(prev_particles):
                    diff = theta - prev_theta
                    log_k = -0.5 * diff @ np.linalg.solve(cov, diff)
                    kernel_density += prev_weights[j] * np.exp(log_k)
                kernel_density = max(kernel_density, 1e-300)

                w = np.exp(log_prior) / kernel_density
                new_particles.append(theta)
                new_distances.append(d)
                new_weights_raw.append(w)

            if n_sims_t > n_particles * max_attempts_per_particle:
                break

        if len(new_particles) == 0:
            logger.warning("Population %d: no particles accepted (eps=%.4f). Stopping.", t, epsilon_t)
            break

        particles_arr = np.array(new_particles)
        distances_arr = np.array(new_distances)
        weights_arr = np.array(new_weights_raw)
        weights_arr /= weights_arr.sum()  # normalize

        ess = 1.0 / np.sum(weights_arr ** 2)
        populations.append(ABCPopulation(
            generation=t,
            particles=particles_arr,
            weights=weights_arr,
            distances=distances_arr,
            epsilon=epsilon_t,
            acceptance_rate=len(new_particles) / max(n_sims_t, 1),
            ess=ess,
            n_simulations=n_sims_t,
        ))

        if verbose:
            logger.info("ABC Pop %d: %d/%d accepted (eps=%.4f, ESS=%.0f)",
                       t, len(new_particles), n_sims_t, epsilon_t, ess)

        if epsilon_t <= min_epsilon:
            logger.info("ABC: epsilon reached min_epsilon=%.4f. Stopping.", min_epsilon)
            break

    # Final result from last population
    final = populations[-1]
    total_sims = sum(p.n_simulations for p in populations)
    total_accepted = sum(len(p.particles) for p in populations)

    return ABCResult(
        posterior_samples=final.particles,
        weights=final.weights,
        distances=final.distances,
        acceptance_rate=total_accepted / max(total_sims, 1),
        n_populations=len(populations),
        populations=populations,
        param_names=param_names,
        computation_time_s=time.monotonic() - t0,
    )


def make_abc_sim_func(
    param_names: list[str],
    bounds: list[tuple[float, float]],
    n_ticks: int = 20,
    n_agents: int = 8,
    seed: int = 42,
) -> Callable[[np.ndarray], dict[str, float]]:
    """
    Create a lightweight sim_func for ABC-SMC using the synthetic objective
    (same pattern as history_matching.run_history_matching).

    No vLLM needed — uses update_score() directly.
    """
    from simulation.sim_loop import update_score

    def sim_func(theta: np.ndarray) -> dict[str, float]:
        params = dict(zip(param_names, theta))
        alpha = params.get("alpha", 0.15)
        dampening = params.get("dampening", 1.0)
        susceptibility = params.get("susceptibility", 0.5)
        resilience = params.get("resilience", 0.2)
        kappa = params.get("kappa", 0.0)
        logistic_k = params.get("logistic_k", 6.0)

        rng = np.random.RandomState(seed)
        scores = [0.3 + rng.uniform(-0.1, 0.1) for _ in range(n_agents)]

        for t in range(n_ticks):
            signal = 0.5 + 0.3 * np.sin(t * 0.5) + rng.normal(0, 0.1)
            signal = float(np.clip(signal, 0.0, 1.0))
            for i in range(n_agents):
                d = dampening if (t % 4 == 0) else 1.0
                scores[i] = update_score(
                    scores[i], signal, d, alpha,
                    susceptibility=susceptibility,
                    resilience=resilience,
                )

        arr = np.array(scores)
        return {
            "mean_score": float(np.mean(arr)),
            "score_std": float(np.std(arr)),
            "max_score": float(np.max(arr)),
            "min_score": float(np.min(arr)),
        }

    return sim_func


def calibration_pipeline(
    sim_func: Callable[[np.ndarray], dict[str, float]],
    priors: list[ABCPrior],
    observed: dict[str, float],
    param_names: list[str],
    bounds: list[tuple[float, float]],
    hm_waves: int = 3,
    hm_samples: int = 64,
    abc_particles: int = 100,
    abc_populations: int = 5,
    seed: int = 42,
) -> dict:
    """
    Full calibration pipeline: History Matching (NROY) → ABC-SMC (posterior).

    1. Run history matching to eliminate implausible parameter regions.
    2. Run ABC-SMC within the NROY region for posterior inference.

    Returns:
        {hm_result, abc_result, nroy_fraction, posterior_summary}
    """
    from simulation.history_matching import HistoryMatcher, PatternTarget

    # Build HM targets from observed data
    targets = []
    for key, value in observed.items():
        tolerance = abs(value) * 0.2  # 20% tolerance
        targets.append(PatternTarget(
            name=key,
            low=value - tolerance,
            high=value + tolerance,
            extractor=lambda output, k=key: output.get(k, 0.0),
            tolerance=tolerance * 0.5,
        ))

    # Step 1: History Matching
    hm = HistoryMatcher(
        targets=targets,
        bounds=bounds,
        param_names=param_names,
        n_waves=hm_waves,
        samples_per_wave=hm_samples,
        threshold=3.0,
        seed=seed,
    )
    nroy_points = hm.run(sim_func, verbose=True)
    nroy_fraction = len(nroy_points) / (hm_waves * hm_samples) if hm_waves * hm_samples > 0 else 0.0

    # Step 2: ABC-SMC within NROY region
    # Narrow priors to NROY bounds
    if len(nroy_points) > 0:
        nroy_mins = nroy_points.min(axis=0)
        nroy_maxs = nroy_points.max(axis=0)
        narrowed_priors = []
        for i, p in enumerate(priors):
            narrowed_priors.append(ABCPrior(
                name=p.name,
                type="uniform",
                low=float(nroy_mins[i]),
                high=float(nroy_maxs[i]),
            ))
    else:
        narrowed_priors = priors

    abc_result = abc_smc(
        sim_func=sim_func,
        priors=narrowed_priors,
        observed=observed,
        n_particles=abc_particles,
        n_populations=abc_populations,
        seed=seed + 1000,
    )

    return {
        "nroy_fraction": round(nroy_fraction, 4),
        "nroy_n_points": len(nroy_points),
        "abc_result": abc_result.to_dict(),
        "posterior_mean": {
            name: round(float(v), 6)
            for name, v in zip(param_names, abc_result.posterior_mean())
        } if len(abc_result.posterior_samples) > 0 else {},
        "posterior_std": {
            name: round(float(v), 6)
            for name, v in zip(param_names, abc_result.posterior_std())
        } if len(abc_result.posterior_samples) > 0 else {},
    }
