"""Backward-compat shim -- import from simulation.config.agents instead."""
from simulation.config.agents import *  # noqa: F401,F403
import simulation.config.agents as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
