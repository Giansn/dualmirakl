"""Backward-compat shim -- import from simulation.optimize.surrogate instead."""
from simulation.optimize.surrogate import *  # noqa: F401,F403
import simulation.optimize.surrogate as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
