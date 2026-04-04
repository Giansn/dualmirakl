"""Backward-compat shim -- import from simulation.core.transitions instead."""
from simulation.core.transitions import *  # noqa: F401,F403
import simulation.core.transitions as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
