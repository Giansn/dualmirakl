"""
Contextual Bandit for agent strategy selection.

Each agent has N strategies (e.g., cooperative, confrontational, observant).
Thompson Sampling selects the strategy with highest expected reward per tick.
The chosen strategy is injected as a constraint into the system prompt.

No external ML framework — pure numpy.
"""

import numpy as np
from dataclasses import dataclass, field

STRATEGIES = {
    "cooperative":     "You actively seek compromises and collaborative solutions.",
    "confrontational": "You defend your position firmly and challenge counterarguments.",
    "observant":       "You observe the dynamics, gather information, and wait.",
    "persuasive":      "You try to convince others of your perspective.",
    "adaptive":        "You adjust your stance to align with the majority opinion.",
}


@dataclass
class BanditState:
    """Beta distribution per strategy: (alpha, beta) = (successes+1, failures+1)."""
    alpha: np.ndarray  # shape: (n_strategies,)
    beta: np.ndarray

    @classmethod
    def uniform(cls, n: int) -> "BanditState":
        return cls(alpha=np.ones(n), beta=np.ones(n))


class ContextualBandit:
    """
    Thompson Sampling over agent strategies.

    Usage:
        bandit = ContextualBandit()
        bandit.register("agent_01")

        # Per tick:
        strategy, prompt_addition = bandit.select("agent_01")
        # ... LLM call with prompt_addition in system prompt ...
        bandit.update("agent_01", strategy, reward=0.7)
    """

    def __init__(self, strategies: dict[str, str] | None = None, seed: int = 42):
        self.strategies = strategies or STRATEGIES
        self.strategy_names = list(self.strategies.keys())
        self.n = len(self.strategy_names)
        self.agents: dict[str, BanditState] = {}
        self.rng = np.random.RandomState(seed)

    def register(self, agent_id: str) -> None:
        self.agents[agent_id] = BanditState.uniform(self.n)

    def select(self, agent_id: str) -> tuple[str, str]:
        """Thompson Sampling: sample from each Beta distribution, pick max."""
        state = self.agents[agent_id]
        samples = self.rng.beta(state.alpha, state.beta)
        idx = int(np.argmax(samples))
        name = self.strategy_names[idx]
        return name, self.strategies[name]

    def update(self, agent_id: str, strategy: str, reward: float) -> None:
        """
        Update after observation.
        reward: 0.0-1.0 (how well did the strategy work?)
        """
        idx = self.strategy_names.index(strategy)
        state = self.agents[agent_id]
        state.alpha[idx] += reward
        state.beta[idx] += (1.0 - reward)

    def get_distribution(self, agent_id: str) -> dict[str, tuple[float, float]]:
        """Current distribution for debugging/analysis."""
        state = self.agents[agent_id]
        return {
            name: (float(state.alpha[i]), float(state.beta[i]))
            for i, name in enumerate(self.strategy_names)
        }

    def decay(self, factor: float = 0.99) -> None:
        """
        Global decay — prevents early rewards from dominating forever.
        Call e.g. every 10 ticks.
        """
        for state in self.agents.values():
            state.alpha = 1.0 + (state.alpha - 1.0) * factor
            state.beta = 1.0 + (state.beta - 1.0) * factor

    def to_dict(self) -> dict:
        """Serialization for DuckDB storage."""
        return {
            agent_id: {
                "alpha": state.alpha.tolist(),
                "beta": state.beta.tolist(),
            }
            for agent_id, state in self.agents.items()
        }

    @classmethod
    def from_dict(cls, data: dict, strategies: dict[str, str] | None = None,
                  seed: int = 42) -> "ContextualBandit":
        bandit = cls(strategies, seed=seed)
        for agent_id, state_data in data.items():
            bandit.agents[agent_id] = BanditState(
                alpha=np.array(state_data["alpha"]),
                beta=np.array(state_data["beta"]),
            )
        return bandit
