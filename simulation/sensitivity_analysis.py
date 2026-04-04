"""Backward-compat shim -- import from simulation.signal.sensitivity instead."""
from simulation.signal.sensitivity import *  # noqa: F401,F403
import simulation.signal.sensitivity as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
