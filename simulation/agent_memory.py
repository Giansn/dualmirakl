"""Backward-compat shim -- import from simulation.knowledge.agent_memory instead."""
from simulation.knowledge.agent_memory import *  # noqa: F401,F403
import simulation.knowledge.agent_memory as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
