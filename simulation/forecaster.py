"""Backward-compat shim -- import from simulation.gpu.forecaster instead."""
from simulation.gpu.forecaster import *  # noqa: F401,F403
import simulation.gpu.forecaster as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
