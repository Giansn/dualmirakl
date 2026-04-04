"""Backward-compat shim -- import from simulation.config.scenario instead."""
from simulation.config.scenario import *  # noqa: F401,F403
import simulation.config.scenario as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
