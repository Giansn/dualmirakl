"""
Bayesian Belief System for agents.

Each agent has beliefs over key scenario variables.
After each interaction, beliefs are updated via conjugate Beta updates.
The belief state is injected as numerical context into the system prompt.

Example: Agent has belief "trust_institution" as Beta(3, 7) -> 30% trust.
After positive experience: Beta(4, 7) -> 36% trust.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BeliefDimension:
    """A belief dimension with Beta prior."""
    name: str
    description: str       # For system prompt injection
    alpha: float = 2.0     # Prior: slightly positive
    beta: float = 2.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        """How established is the belief? (0=uncertain, 1=very certain)"""
        total = self.alpha + self.beta
        return min(1.0, total / 50.0)

    def update(self, positive: bool, weight: float = 1.0) -> None:
        """Conjugate update: observation -> posterior."""
        if positive:
            self.alpha += weight
        else:
            self.beta += weight

    def update_continuous(self, observation: float, weight: float = 1.0) -> None:
        """Update with continuous observation (0.0-1.0)."""
        self.alpha += observation * weight
        self.beta += (1.0 - observation) * weight

    def to_prompt_string(self) -> str:
        """Human-readable belief for system prompt."""
        level = self.mean
        strength = self.strength
        if level > 0.7:
            stance = "strongly positive"
        elif level > 0.5:
            stance = "slightly positive"
        elif level > 0.3:
            stance = "slightly negative"
        else:
            stance = "strongly negative"
        certainty = ("uncertain" if strength < 0.3
                     else "moderately certain" if strength < 0.6
                     else "very convinced")
        return f"{self.description}: {stance} ({certainty}, {level:.0%})"


class AgentBeliefs:
    """
    Belief system of an agent.

    Usage:
        beliefs = AgentBeliefs("agent_01")
        beliefs.add("trust_gov", "Trust in government", alpha=3, beta=7)
        beliefs.add("risk_tolerance", "Risk tolerance", alpha=5, beta=5)

        # After interaction:
        beliefs.update("trust_gov", positive=False)

        # For system prompt:
        context = beliefs.to_context_string()
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.dimensions: dict[str, BeliefDimension] = {}
        self.history: list[dict] = []

    def add(self, name: str, description: str,
            alpha: float = 2.0, beta: float = 2.0) -> None:
        self.dimensions[name] = BeliefDimension(name, description, alpha, beta)

    def update(self, dimension: str, positive: bool, weight: float = 1.0) -> None:
        if dimension not in self.dimensions:
            return
        self.dimensions[dimension].update(positive, weight)
        self.history.append({
            "dimension": dimension,
            "positive": positive,
            "weight": weight,
            "new_mean": self.dimensions[dimension].mean,
        })

    def update_continuous(self, dimension: str, observation: float,
                          weight: float = 1.0) -> None:
        if dimension not in self.dimensions:
            return
        self.dimensions[dimension].update_continuous(observation, weight)
        self.history.append({
            "dimension": dimension,
            "observation": observation,
            "weight": weight,
            "new_mean": self.dimensions[dimension].mean,
        })

    def to_context_string(self) -> str:
        """Belief state as context for system prompt."""
        lines = ["Your current beliefs:"]
        for dim in self.dimensions.values():
            lines.append(f"  - {dim.to_prompt_string()}")
        return "\n".join(lines)

    def to_vector(self) -> dict[str, float]:
        """Numeric belief vector for statistics."""
        return {name: dim.mean for name, dim in self.dimensions.items()}

    def to_dict(self) -> dict:
        return {
            name: {"alpha": dim.alpha, "beta": dim.beta}
            for name, dim in self.dimensions.items()
        }

    @classmethod
    def from_dict(cls, agent_id: str, data: dict,
                  descriptions: dict[str, str]) -> "AgentBeliefs":
        beliefs = cls(agent_id)
        for name, params in data.items():
            beliefs.add(name, descriptions.get(name, name),
                        params["alpha"], params["beta"])
        return beliefs
