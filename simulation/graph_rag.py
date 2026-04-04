"""Backward-compat shim -- import from simulation.knowledge.graph_rag instead."""
from simulation.knowledge.graph_rag import *  # noqa: F401,F403
import simulation.knowledge.graph_rag as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
