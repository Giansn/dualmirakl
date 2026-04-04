"""Backward-compat shim -- import from simulation.analysis.possibility_report instead."""
from simulation.analysis.possibility_report import *  # noqa: F401,F403
import simulation.analysis.possibility_report as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
