"""
SimPy + LLM simulation loop for media addiction dynamics.

Architecture:
  - SimPy drives discrete-event time steps
  - Each simulated agent calls a vLLM-backed LLM turn per step
  - JAX/NumPyro handles the probabilistic state updates
  - Results feed into FLAME GPU 2 for large-scale population runs
"""

import simpy
import asyncio
import numpy as np
from simulation.agent_roles import AGENT_ROLES
from orchestrator import agent_turn


class MediaAgent:
    """A single simulated individual in the media consumption environment."""

    def __init__(self, env: simpy.Environment, agent_id: str, role: str):
        self.env = env
        self.agent_id = agent_id
        self.role = role
        self.cfg = AGENT_ROLES[role]
        self.history: list[dict] = []
        self.addiction_score: float = np.random.uniform(0.1, 0.5)
        self.engagement_log: list[float] = []

    def run(self):
        while True:
            yield self.env.timeout(1)  # 1 time unit = 1 simulated hour
            asyncio.get_event_loop().run_until_complete(self._step())

    async def _step(self):
        prompt = (
            f"[Hour {self.env.now}] Your current media engagement score is "
            f"{self.addiction_score:.2f}/1.0. Describe your next action."
        )
        response = await agent_turn(
            agent_id=self.agent_id,
            backend=self.cfg["backend"],
            system_prompt=self.cfg["system"],
            user_message=prompt,
            history=self.history[-4:],  # keep last 4 turns
        )
        # Naive score update — replace with NumPyro model
        if any(w in response.lower() for w in ["scroll", "watch", "check", "open"]):
            self.addiction_score = min(1.0, self.addiction_score + 0.05)
        else:
            self.addiction_score = max(0.0, self.addiction_score - 0.02)

        self.history.append({"role": "assistant", "content": response})
        self.engagement_log.append(self.addiction_score)


def run_simulation(n_hours: int = 24, n_agents: int = 4):
    env = simpy.Environment()
    roles = list(AGENT_ROLES.keys())
    agents = [
        MediaAgent(env, f"agent_{i}", roles[i % len(roles)])
        for i in range(n_agents)
    ]
    for agent in agents:
        env.process(agent.run())
    env.run(until=n_hours)
    return agents


if __name__ == "__main__":
    print("Running 12-hour simulation with 4 agents...")
    agents = run_simulation(n_hours=12, n_agents=4)
    for a in agents:
        print(f"{a.agent_id} ({a.role}): final score={a.addiction_score:.2f}")
