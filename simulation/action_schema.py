"""Backward-compat shim -- import from simulation.config.action_schema instead."""
from simulation.config.action_schema import *  # noqa: F401,F403
import simulation.config.action_schema as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
