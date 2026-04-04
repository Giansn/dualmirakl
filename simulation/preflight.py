"""Backward-compat shim -- import from simulation.signal.preflight instead."""
from simulation.signal.preflight import *  # noqa: F401,F403
import simulation.signal.preflight as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
