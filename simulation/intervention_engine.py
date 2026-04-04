"""Backward-compat shim -- import from simulation.signal.intervention instead."""
from simulation.signal.intervention import *  # noqa: F401,F403
import simulation.signal.intervention as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
