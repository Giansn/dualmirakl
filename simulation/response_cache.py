"""Backward-compat shim -- import from simulation.knowledge.response_cache instead."""
from simulation.knowledge.response_cache import *  # noqa: F401,F403
import simulation.knowledge.response_cache as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
