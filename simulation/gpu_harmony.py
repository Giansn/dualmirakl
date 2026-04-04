"""Backward-compat shim -- import from simulation.gpu.harmony instead."""
from simulation.gpu.harmony import *  # noqa: F401,F403
import simulation.gpu.harmony as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
