"""Backward-compat shim -- import from simulation.analysis.ensemble instead."""
from simulation.analysis.ensemble import *  # noqa: F401,F403
import simulation.analysis.ensemble as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
