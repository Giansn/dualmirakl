"""Backward-compat shim -- import from simulation.signal.computation instead."""
from simulation.signal.computation import *  # noqa: F401,F403
import simulation.signal.computation as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
