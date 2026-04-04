"""Backward-compat shim -- import from simulation.core.safety instead."""
from simulation.core.safety import *  # noqa: F401,F403
import simulation.core.safety as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
