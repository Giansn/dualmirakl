"""Backward-compat shim -- import from simulation.analysis.dynamics instead."""
from simulation.analysis.dynamics import *  # noqa: F401,F403
import simulation.analysis.dynamics as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
