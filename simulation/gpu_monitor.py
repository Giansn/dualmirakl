"""Backward-compat shim -- import from simulation.gpu.monitor instead."""
from simulation.gpu.monitor import *  # noqa: F401,F403
import simulation.gpu.monitor as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
