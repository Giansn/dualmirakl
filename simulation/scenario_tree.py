"""Backward-compat shim -- import from simulation.analysis.scenario_tree instead."""
from simulation.analysis.scenario_tree import *  # noqa: F401,F403
import simulation.analysis.scenario_tree as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
