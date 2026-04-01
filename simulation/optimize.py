"""
Optuna-based Bayesian optimization for dualmirakl simulation parameters.

Optional — if optuna is not installed, raises ImportError with install hint.
Works in both 2-GPU (dualmirakl only) and 3-GPU (dualmirakl + FLAME) modes.

Two optimization modes:
    1. Fast (surrogate): Uses update_score() surrogate — no vLLM needed.
       Good for exploring parameter space quickly.
    2. Full (live): Runs actual run_simulation() with vLLM.
       Slower but captures LLM behavioral dynamics.

Usage:
    from simulation.optimize import run_optimization

    # Fast surrogate sweep (no GPU needed)
    study = run_optimization(mode="fast", n_trials=100)

    # Full simulation sweep (needs vLLM running)
    study = run_optimization(mode="full", n_trials=20)

    # With FLAME parameters included
    study = run_optimization(mode="fast", n_trials=100, include_flame=True)
"""

import logging
import math
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Parameter space definitions ──────────────────────────────────────────────

# Core dualmirakl parameters (always optimized)
DUALMIRAKL_PARAMS = {
    "alpha":        {"low": 0.05, "high": 0.40, "type": "float"},
    "K":            {"low": 2,    "high": 8,    "type": "int"},
    "threshold":    {"low": 0.55, "high": 0.90, "type": "float"},
    "dampening":    {"low": 0.3,  "high": 1.0,  "type": "float"},
    "susceptibility": {"low": 0.2, "high": 1.0, "type": "float"},
    "resilience":   {"low": 0.0,  "high": 0.6,  "type": "float"},
    "logistic_k":   {"low": 3.0,  "high": 10.0, "type": "float"},
}

# FLAME parameters (only when include_flame=True)
FLAME_PARAMS = {
    "flame_kappa":            {"low": -0.5, "high": 1.0, "type": "float"},
    "flame_influencer_weight": {"low": 1.0, "high": 20.0, "type": "float"},
    "flame_drift_sigma":      {"low": 0.0, "high": 0.1, "type": "float"},
    "flame_interaction_radius": {"low": 3.0, "high": 30.0, "type": "float"},
    "flame_sub_steps":        {"low": 3,   "high": 30,  "type": "int"},
}


def _suggest_params(trial, param_space: dict) -> dict:
    """Suggest parameters from Optuna trial."""
    params = {}
    for name, spec in param_space.items():
        if spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
    return params


# ── Surrogate objective (fast, no vLLM) ──────────────────────────────────────

def _surrogate_objective(
    trial,
    n_ticks: int = 12,
    n_agents: int = 4,
    include_flame: bool = False,
    target_mean: float = 0.5,
    target_std: float = 0.15,
) -> float:
    """
    Fast surrogate objective using update_score() directly.
    Simulates score dynamics without LLM calls.

    Optimizes for:
        1. Score trajectory diversity (std close to target_std)
        2. Score mean close to target_mean (avoid trivial solutions)
        3. Score stability (low variance in final ticks)
        4. If FLAME: population spread matches agent spread

    Returns combined loss (lower is better).
    """
    from simulation.signal_computation import update_score

    params = _suggest_params(trial, DUALMIRAKL_PARAMS)
    flame_params = {}
    if include_flame:
        flame_params = _suggest_params(trial, FLAME_PARAMS)

    rng = np.random.RandomState(42)
    alpha = params["alpha"]
    k_obs = params["K"]
    dampening_base = params["dampening"]
    susceptibility = params["susceptibility"]
    resilience = params["resilience"]
    logistic_k = params["logistic_k"]

    # Simulate N agents with heterogeneous traits
    scores = [0.1 + 0.4 * rng.random() for _ in range(n_agents)]
    agent_suscept = [max(0.05, susceptibility + rng.normal(0, 0.1)) for _ in range(n_agents)]
    agent_resil = [max(0.0, min(0.8, resilience + rng.normal(0, 0.1))) for _ in range(n_agents)]
    all_trajectories = [[] for _ in range(n_agents)]

    for t in range(1, n_ticks + 1):
        # Synthetic signal (sinusoidal + noise, mimics embedding output)
        base_signal = 0.5 + 0.3 * math.sin(t * 0.5) + rng.normal(0, 0.1)
        dampening = dampening_base if (t % max(1, k_obs) == 0) else 1.0

        for i in range(n_agents):
            signal = max(0.0, min(1.0, base_signal + rng.normal(0, 0.05)))
            scores[i] = update_score(
                scores[i], signal, dampening, alpha,
                mode="logistic", logistic_k=logistic_k,
                susceptibility=agent_suscept[i],
                resilience=agent_resil[i],
            )
            all_trajectories[i].append(scores[i])

    # Metrics
    final_scores = [t[-1] for t in all_trajectories]
    mean_final = np.mean(final_scores)
    std_final = np.std(final_scores)

    # Stability: low variance in last 3 ticks
    stability = np.mean([np.std(t[-3:]) for t in all_trajectories])

    # Loss components
    loss_mean = (mean_final - target_mean) ** 2
    loss_diversity = (std_final - target_std) ** 2
    loss_stability = stability ** 2

    loss = loss_mean + loss_diversity + 0.5 * loss_stability

    # FLAME surrogate: simulate coupled population dynamics
    if include_flame and flame_params:
        kappa = flame_params["flame_kappa"]
        inf_weight = flame_params["flame_influencer_weight"]
        drift = flame_params["flame_drift_sigma"]
        sub_steps = flame_params["flame_sub_steps"]

        # Simple coupled population model (N_pop virtual agents)
        n_pop = 100  # surrogate uses small population
        pop_scores = rng.uniform(0.1, 0.5, n_pop)

        for _ in range(sub_steps):
            pop_mean = np.mean(pop_scores)
            # Weighted mean with influencer scores
            weighted_mean = (
                pop_mean * n_pop + sum(final_scores) * inf_weight
            ) / (n_pop + n_agents * inf_weight)

            for j in range(n_pop):
                coupling = kappa * (weighted_mean - pop_scores[j])
                noise = drift * rng.normal()
                pop_scores[j] = np.clip(
                    pop_scores[j] + coupling + noise, 0.0, 1.0
                )

        pop_mean_final = np.mean(pop_scores)
        pop_std_final = np.std(pop_scores)

        # FLAME loss: population should spread reasonably, not collapse
        loss_pop_collapse = max(0, 0.05 - pop_std_final) ** 2 * 10
        loss_pop_diverge = max(0, pop_std_final - 0.4) ** 2 * 5
        loss += loss_pop_collapse + loss_pop_diverge

        # Report FLAME metrics
        trial.set_user_attr("flame_pop_mean", float(pop_mean_final))
        trial.set_user_attr("flame_pop_std", float(pop_std_final))

    # Report metrics for analysis
    trial.set_user_attr("mean_final", float(mean_final))
    trial.set_user_attr("std_final", float(std_final))
    trial.set_user_attr("stability", float(stability))

    return loss


# ── Full simulation objective (slow, needs vLLM) ─────────────────────────────

def _full_objective(
    trial,
    n_ticks: int = 6,
    n_participants: int = 4,
    include_flame: bool = False,
    target_mean: float = 0.5,
) -> float:
    """
    Full objective running actual simulation with vLLM.
    Much slower but captures real LLM behavioral dynamics.
    """
    import asyncio
    from simulation.sim_loop import run_simulation

    params = _suggest_params(trial, DUALMIRAKL_PARAMS)
    flame_config = None
    if include_flame:
        flame_params = _suggest_params(trial, FLAME_PARAMS)
        flame_config = {
            "kappa": flame_params["flame_kappa"],
            "influencer_weight": flame_params["flame_influencer_weight"],
            "drift_sigma": flame_params["flame_drift_sigma"],
            "interaction_radius": flame_params["flame_interaction_radius"],
            "sub_steps": flame_params["flame_sub_steps"],
        }

    async def _run():
        participants, world_state = await run_simulation(
            n_ticks=n_ticks,
            n_participants=n_participants,
            k=params["K"],
            alpha=params["alpha"],
            intervention_threshold=params["threshold"],
            score_mode="logistic",
            logistic_k=params["logistic_k"],
            seed=trial.number + 42,
            flame_enabled=include_flame,
            flame_config=flame_config,
        )
        return participants, world_state

    participants, world_state = asyncio.run(_run())

    final_scores = [p.behavioral_score for p in participants]
    mean_final = np.mean(final_scores)
    std_final = np.std(final_scores)
    stats = world_state.compute_score_statistics(n_ticks)

    trial.set_user_attr("mean_final", float(mean_final))
    trial.set_user_attr("std_final", float(std_final))

    # Optimize for meaningful dynamics (not trivial convergence)
    loss = (mean_final - target_mean) ** 2 + max(0, 0.05 - std_final) ** 2

    return loss


# ── Public API ───────────────────────────────────────────────────────────────

def run_optimization(
    mode: str = "fast",
    n_trials: int = 100,
    n_ticks: int = 12,
    n_participants: int = 4,
    include_flame: bool = False,
    target_mean: float = 0.5,
    target_std: float = 0.15,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    wandb_project: Optional[str] = None,
) -> "optuna.Study":
    """
    Run Bayesian optimization over simulation parameters.

    Args:
        mode: "fast" (surrogate, no vLLM) or "full" (live simulation)
        n_trials: Number of Optuna trials
        n_ticks: Ticks per trial (shorter = faster for "full" mode)
        n_participants: Number of agents
        include_flame: Include FLAME parameters in search space
        target_mean: Target mean score to optimize toward
        target_std: Target score diversity (fast mode only)
        study_name: Optuna study name (for persistence)
        storage: Optuna storage URL (e.g., "sqlite:///optuna.db")
        wandb_project: If set, log to W&B alongside Optuna

    Returns:
        optuna.Study with results
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "optuna not installed. Install with: pip install optuna\n"
            "Optional: pip install optuna-dashboard  (for web UI)"
        )

    # Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name or f"dualmirakl_{'flame_' if include_flame else ''}{mode}",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    # W&B callback (optional)
    callbacks = []
    if wandb_project:
        try:
            from simulation.tracking import tracker
            tracker.init_run(
                config={
                    "optuna_mode": mode,
                    "n_trials": n_trials,
                    "include_flame": include_flame,
                    "target_mean": target_mean,
                },
                project=wandb_project,
                tags=["optuna", mode],
            )
            callbacks.append(_wandb_callback)
        except Exception as e:
            logger.warning("wandb setup failed for Optuna: %s", e)

    if mode == "fast":
        def objective(trial):
            return _surrogate_objective(
                trial, n_ticks=n_ticks, n_agents=n_participants,
                include_flame=include_flame,
                target_mean=target_mean, target_std=target_std,
            )
    elif mode == "surrogate":
        try:
            from simulation.surrogate import SurrogateModel
        except ImportError:
            raise ImportError("scikit-learn required for surrogate mode: pip install scikit-learn")
        sm = SurrogateModel()
        bench = sm.build(n_samples=max(5000, n_trials * 10), include_flame=include_flame)
        winner_r2 = bench.gp_r2 if bench.winner == "gp" else bench.mlp_r2
        logger.info("Surrogate trained: %s (R²=%.4f, RMSE=%.4f)",
                     bench.winner, winner_r2,
                     bench.gp_rmse if bench.winner == "gp" else bench.mlp_rmse)
        if winner_r2 < 0.7:
            logger.warning("Surrogate R²=%.2f is low — results may be unreliable", winner_r2)

        def objective(trial):
            params = _suggest_params(trial, DUALMIRAKL_PARAMS)
            if include_flame:
                params.update(_suggest_params(trial, FLAME_PARAMS))
            return sm.predict_loss(params)
    elif mode == "full":
        def objective(trial):
            return _full_objective(
                trial, n_ticks=n_ticks, n_participants=n_participants,
                include_flame=include_flame, target_mean=target_mean,
            )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'fast', 'surrogate', or 'full'.")

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

    # Print results
    print(f"\n── Optuna {mode} optimization ({n_trials} trials) ──")
    print(f"  Best loss: {study.best_value:.6f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k:<25} = {v:.4f}" if isinstance(v, float) else f"    {k:<25} = {v}")
    if study.best_trial.user_attrs:
        print(f"  Metrics:")
        for k, v in study.best_trial.user_attrs.items():
            print(f"    {k:<25} = {v:.4f}" if isinstance(v, float) else f"    {k:<25} = {v}")

    # Finish W&B if active
    if wandb_project:
        try:
            from simulation.tracking import tracker
            tracker.log_summary({
                "mean": study.best_trial.user_attrs.get("mean_final", 0),
                "std": study.best_trial.user_attrs.get("std_final", 0),
            })
            tracker.finish()
        except Exception:
            pass

    return study


def _wandb_callback(study, trial):
    """Optuna callback to log each trial to W&B."""
    try:
        from simulation.tracking import tracker
        if tracker.active:
            metrics = {"optuna_loss": trial.value, **trial.params}
            metrics.update(trial.user_attrs)
            tracker._run.log(metrics, step=trial.number)
    except Exception:
        pass


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Optimize dualmirakl parameters")
    parser.add_argument("--mode", choices=["fast", "surrogate", "full"], default="fast")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--flame", action="store_true", help="Include FLAME params")
    parser.add_argument("--target-mean", type=float, default=0.5)
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage (e.g., sqlite:///optuna.db)")
    parser.add_argument("--wandb", type=str, default=None,
                        help="W&B project name for logging")
    args = parser.parse_args()

    study = run_optimization(
        mode=args.mode,
        n_trials=args.trials,
        n_ticks=args.ticks,
        include_flame=args.flame,
        target_mean=args.target_mean,
        storage=args.storage,
        wandb_project=args.wandb,
    )
