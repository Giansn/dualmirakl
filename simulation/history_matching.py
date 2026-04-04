"""Backward-compat shim -- import from simulation.analysis.history_matching instead."""
from simulation.analysis.history_matching import *  # noqa: F401,F403
import simulation.analysis.history_matching as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
