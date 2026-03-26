"""
Evolutionary strategy selection across simulation generations.

After a sim run, the most successful agent strategies are selected,
mutated, and recombined. The next run starts with a fitter population.

(mu+lambda) selection: mu parents + lambda offspring -> top mu survive.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class AgentGenome:
    """Evolvable parameters of an agent."""
    agent_id: str
    personality: dict[str, float]    # Big Five, 0.0-1.0
    initial_stance: float            # -1.0 to 1.0
    influence_weight: float          # 0.0 to 1.0
    strategy_bias: dict[str, float]  # Bandit priors per strategy
    fitness: float = 0.0


class EvolutionEngine:
    """
    (mu+lambda) evolution over agent populations.

    Usage:
        evo = EvolutionEngine(mu=10, lambda_=20)
        evo.load_population("agents.json")

        # After sim run: assign fitness
        evo.set_fitness("agent_01", fitness=0.85)

        # Next generation
        next_gen = evo.evolve()
        evo.save_population("agents_gen2.json", next_gen)
    """

    def __init__(
        self,
        mu: int = 10,
        lambda_: int = 20,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.3,
        seed: int = 42,
    ):
        self.mu = mu
        self.lambda_ = lambda_
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(seed)
        self.population: list[AgentGenome] = []
        self.generation: int = 0

    def load_population(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.population = [
            AgentGenome(
                agent_id=d["id"],
                personality=d.get("personality_vector", {}),
                initial_stance=d.get("initial_stance", 0.0),
                influence_weight=d.get("influence_weight", 0.5),
                strategy_bias=d.get("strategy_bias", {}),
            )
            for d in data
        ]

    def set_fitness(self, agent_id: str, fitness: float) -> None:
        for genome in self.population:
            if genome.agent_id == agent_id:
                genome.fitness = fitness
                return

    def _mutate(self, genome: AgentGenome) -> AgentGenome:
        """Gaussian mutation on numeric parameters."""
        child = copy.deepcopy(genome)
        child.agent_id = f"gen{self.generation + 1}_{self.rng.integers(10000):04d}"

        for key in child.personality:
            if self.rng.random() < self.mutation_rate:
                delta = self.rng.normal(0, 0.1)
                child.personality[key] = float(np.clip(
                    child.personality[key] + delta, 0.0, 1.0
                ))

        if self.rng.random() < self.mutation_rate:
            child.initial_stance = float(np.clip(
                child.initial_stance + self.rng.normal(0, 0.15), -1.0, 1.0
            ))

        if self.rng.random() < self.mutation_rate:
            child.influence_weight = float(np.clip(
                child.influence_weight + self.rng.normal(0, 0.1), 0.0, 1.0
            ))

        child.fitness = 0.0
        return child

    def _crossover(self, parent_a: AgentGenome,
                   parent_b: AgentGenome) -> AgentGenome:
        """Uniform crossover: per parameter randomly from A or B."""
        child = copy.deepcopy(parent_a)
        child.agent_id = f"gen{self.generation + 1}_{self.rng.integers(10000):04d}"

        for key in child.personality:
            if self.rng.random() < 0.5 and key in parent_b.personality:
                child.personality[key] = parent_b.personality[key]

        if self.rng.random() < 0.5:
            child.initial_stance = parent_b.initial_stance
        if self.rng.random() < 0.5:
            child.influence_weight = parent_b.influence_weight

        child.fitness = 0.0
        return child

    def evolve(self) -> list[AgentGenome]:
        """One generation: selection -> variation -> (mu+lambda) survival."""
        ranked = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        parents = ranked[:self.mu]

        offspring = []
        for _ in range(self.lambda_):
            if self.rng.random() < self.crossover_rate and len(parents) >= 2:
                idx = self.rng.choice(len(parents), size=2, replace=False)
                child = self._crossover(parents[idx[0]], parents[idx[1]])
            else:
                parent = parents[self.rng.integers(len(parents))]
                child = self._mutate(parent)
            offspring.append(child)

        combined = parents + offspring
        combined.sort(key=lambda g: g.fitness, reverse=True)
        self.population = combined[:self.mu]
        self.generation += 1
        return self.population

    def save_population(self, path: str,
                        population: list[AgentGenome] | None = None) -> None:
        pop = population or self.population
        data = [
            {
                "id": g.agent_id,
                "personality_vector": g.personality,
                "initial_stance": g.initial_stance,
                "influence_weight": g.influence_weight,
                "strategy_bias": g.strategy_bias,
                "fitness": g.fitness,
                "generation": self.generation,
            }
            for g in pop
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def stats(self) -> dict:
        """Generation statistics."""
        fitnesses = [g.fitness for g in self.population]
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "mean_fitness": float(np.mean(fitnesses)),
            "max_fitness": float(np.max(fitnesses)),
            "min_fitness": float(np.min(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
        }
