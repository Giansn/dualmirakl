"""
FLAME GPU 2 population dynamics engine for dualmirakl.

Runs on GPU 2 as a population amplifier:
    - N influencer agents (mapped 1:1 from LLM participants)
    - N_pop population agents with heterogeneous susceptibility/resilience
    - Spatial messaging for local peer interactions
    - Score dynamics matching dualmirakl (EMA/logistic, coupling κ)

Usage:
    from simulation.flame import FlameEngine, FlameBridge

    engine = FlameEngine(config)
    bridge = FlameBridge(n_influencers=4)

    engine.init()
    bridge.push_influencer_positions(engine)

    # Each dualmirakl tick:
    bridge.push_influencer_scores(engine, participant_scores)
    engine.step(sub_steps=10)
    snapshot = bridge.pull_population_stats(engine, tick, sub_steps=10)

    # Export
    bridge.export_snapshots("data/run_id/flame_population.json")
"""

import math
import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default configuration matching dualmirakl sim_loop defaults
DEFAULT_CONFIG = {
    "n_population": 10_000,
    "n_influencers": 4,
    "space_size": 100.0,
    "interaction_radius": 10.0,
    "alpha": 0.15,
    "kappa": 0.1,
    "dampening": 1.0,
    "influencer_weight": 5.0,
    "score_mode": "ema",
    "logistic_k": 6.0,
    "drift_sigma": 0.01,
    "move_speed": 0.5,
    "sub_steps": 10,
    "gpu_id": 2,
    "seed": 42,
}


class FlameEngine:
    """
    Wrapper around pyflamegpu.CUDASimulation for integration with dualmirakl.

    The engine manages:
        - Model creation via models.build_model_description()
        - Agent population initialization (heterogeneous traits)
        - Influencer score injection (from LLM participants)
        - Sub-stepping (multiple FLAME steps per dualmirakl tick)
        - Population statistics extraction
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._sim = None
        self._model = None
        self._log_cfg = None
        self._step_log = None
        self._initialized = False
        self._total_steps = 0
        self._pyflamegpu = None

    def init(self):
        """
        Build the FLAME GPU 2 model, create the simulation, and populate agents.
        Must be called before step() or any data access.
        """
        try:
            import pyflamegpu
            self._pyflamegpu = pyflamegpu
        except ImportError:
            raise ImportError(
                "pyflamegpu not installed. Install from https://whl.flamegpu.com "
                "or build from source. Requires NVIDIA GPU + CUDA >= 12.0."
            )

        from .models import build_model_description

        logger.info(
            "FLAME GPU 2 init: %d population agents, %d influencers, "
            "space=%.0f, radius=%.1f, gpu=%d",
            self.config["n_population"],
            self.config["n_influencers"],
            self.config["space_size"],
            self.config["interaction_radius"],
            self.config["gpu_id"],
        )

        self._model, self._log_cfg, self._step_log = build_model_description(
            self.config
        )

        # Create simulation
        self._sim = pyflamegpu.CUDASimulation(self._model)
        self._sim.CUDAConfig().device_id = self.config["gpu_id"]
        self._sim.SimulationConfig().random_seed = self.config["seed"]

        # Don't set a fixed step count — we drive stepping manually
        self._sim.SimulationConfig().steps = 0

        self._sim.setStepLog(self._step_log)

        # Populate agents
        self._populate_agents()

        # Apply initial configuration to simulation
        self._sim.applyConfig()

        self._initialized = True
        logger.info("FLAME GPU 2 engine initialized successfully")

    def _populate_agents(self):
        """Create initial agent populations with heterogeneous traits."""
        pyflamegpu = self._pyflamegpu
        rng = random.Random(self.config["seed"])
        space = self.config["space_size"]

        # --- Influencer agents ---
        inf_pop = pyflamegpu.AgentVector(
            self._model.Agent("Influencer"), self.config["n_influencers"]
        )
        for i in range(self.config["n_influencers"]):
            agent = inf_pop[i]
            # Place evenly around center (will be overridden by bridge)
            angle = 2 * math.pi * i / self.config["n_influencers"]
            cx, cy = space / 2, space / 2
            r = space * 0.3
            agent.setVariableFloat("x", cx + r * math.cos(angle))
            agent.setVariableFloat("y", cy + r * math.sin(angle))
            agent.setVariableFloat("score", 0.3)  # default initial
            agent.setVariableInt("llm_index", i)

        self._sim.setPopulationData(inf_pop)

        # --- Population agents ---
        n_pop = self.config["n_population"]
        pop = pyflamegpu.AgentVector(
            self._model.Agent("Population"), n_pop
        )
        for i in range(n_pop):
            agent = pop[i]
            agent.setVariableFloat("x", rng.uniform(0, space))
            agent.setVariableFloat("y", rng.uniform(0, space))
            # Heterogeneous traits matching dualmirakl Beta distributions
            agent.setVariableFloat("score", rng.uniform(0.1, 0.5))
            agent.setVariableFloat(
                "susceptibility", _beta_sample(rng, 2, 3)
            )
            agent.setVariableFloat(
                "resilience", _beta_sample(rng, 2, 5)
            )
            agent.setVariableFloat("score_delta", 0.0)

        self._sim.setPopulationData(pop)

    def step(self, sub_steps: Optional[int] = None):
        """
        Run N FLAME GPU 2 steps (sub-steps within one dualmirakl tick).

        Args:
            sub_steps: Number of FLAME steps to run. Defaults to config value.
        """
        if not self._initialized:
            raise RuntimeError("FlameEngine.init() must be called first")

        n = sub_steps or self.config["sub_steps"]
        for _ in range(n):
            self._sim.step()
            self._total_steps += 1

    def set_influencer_scores(self, scores: list[float]):
        """
        Update influencer agent scores from LLM participant behavioral_scores.
        Called by FlameBridge.push_influencer_scores().
        """
        if not self._initialized:
            raise RuntimeError("FlameEngine.init() must be called first")

        pyflamegpu = self._pyflamegpu
        inf_pop = pyflamegpu.AgentVector(
            self._model.Agent("Influencer"), self.config["n_influencers"]
        )
        self._sim.getPopulationData(inf_pop)

        for i, score in enumerate(scores):
            inf_pop[i].setVariableFloat("score", float(score))

        self._sim.setPopulationData(inf_pop)

    def set_influencer_positions(self, positions: list[tuple[float, float]]):
        """
        Set influencer positions in social space.
        Called by FlameBridge.push_influencer_positions().
        """
        if not self._initialized:
            raise RuntimeError("FlameEngine.init() must be called first")

        pyflamegpu = self._pyflamegpu
        inf_pop = pyflamegpu.AgentVector(
            self._model.Agent("Influencer"), self.config["n_influencers"]
        )
        self._sim.getPopulationData(inf_pop)

        for i, (x, y) in enumerate(positions):
            inf_pop[i].setVariableFloat("x", float(x))
            inf_pop[i].setVariableFloat("y", float(y))

        self._sim.setPopulationData(inf_pop)

    def set_environment(self, **kwargs):
        """
        Update environment properties at runtime.
        Useful for applying interventions (dampening) or changing coupling.
        """
        if not self._initialized:
            raise RuntimeError("FlameEngine.init() must be called first")

        env_manager = self._sim.environment
        type_map = {
            "alpha": ("Float", float),
            "kappa": ("Float", float),
            "dampening": ("Float", float),
            "influencer_weight": ("Float", float),
            "drift_sigma": ("Float", float),
            "move_speed": ("Float", float),
            "score_mode": ("Int", int),
            "logistic_k": ("Float", float),
        }
        for key, value in kwargs.items():
            if key in type_map:
                type_name, cast = type_map[key]
                setter = getattr(env_manager, f"setProperty{type_name}")
                setter(key, cast(value))
            else:
                logger.warning("Unknown environment property: %s", key)

    def get_population_stats(self) -> dict:
        """
        Extract aggregate population statistics from FLAME GPU 2.

        Returns:
            dict with: count, mean_score, std_score, min_score, max_score,
                       influencer_scores, histogram (10 bins)
        """
        if not self._initialized:
            raise RuntimeError("FlameEngine.init() must be called first")

        pyflamegpu = self._pyflamegpu

        # Get population scores
        pop = pyflamegpu.AgentVector(
            self._model.Agent("Population"), self.config["n_population"]
        )
        self._sim.getPopulationData(pop)

        scores = []
        for i in range(pop.size()):
            scores.append(pop[i].getVariableFloat("score"))

        n = len(scores)
        if n == 0:
            return {
                "count": 0, "mean_score": 0, "std_score": 0,
                "min_score": 0, "max_score": 0,
                "influencer_scores": [], "histogram": [0] * 10,
            }

        mean_s = sum(scores) / n
        var_s = sum((s - mean_s) ** 2 for s in scores) / n
        std_s = var_s ** 0.5

        # Histogram (10 bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0])
        histogram = [0] * 10
        for s in scores:
            bin_idx = min(int(s * 10), 9)
            histogram[bin_idx] += 1

        # Get influencer scores
        inf_pop = pyflamegpu.AgentVector(
            self._model.Agent("Influencer"), self.config["n_influencers"]
        )
        self._sim.getPopulationData(inf_pop)
        inf_scores = [
            inf_pop[i].getVariableFloat("score")
            for i in range(inf_pop.size())
        ]

        return {
            "count": n,
            "mean_score": mean_s,
            "std_score": std_s,
            "min_score": min(scores),
            "max_score": max(scores),
            "influencer_scores": inf_scores,
            "histogram": histogram,
        }

    def get_score_trajectories(self) -> list[dict]:
        """
        Extract per-step logged population statistics.

        Returns:
            list of dicts with: step, mean, std, min, max, count
        """
        if not self._initialized:
            return []

        log = self._sim.getRunLog()
        steps = log.getStepLog()
        trajectories = []
        for i, step_log in enumerate(steps):
            pop_log = step_log.getAgent("Population")
            trajectories.append({
                "step": i,
                "mean": pop_log.getMean("score"),
                "std": pop_log.getStandardDev("score"),
                "min": pop_log.getMin("score"),
                "max": pop_log.getMax("score"),
                "count": pop_log.getCount(),
            })
        return trajectories

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def initialized(self) -> bool:
        return self._initialized

    def shutdown(self):
        """Release FLAME GPU 2 resources."""
        self._sim = None
        self._model = None
        self._initialized = False
        self._total_steps = 0
        logger.info("FLAME GPU 2 engine shut down")


def _beta_sample(rng: random.Random, a: float, b: float) -> float:
    """Sample from Beta(a, b) distribution using random.Random."""
    return rng.betavariate(a, b)
