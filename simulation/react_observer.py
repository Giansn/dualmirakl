"""Backward-compat shim -- import from simulation.observe.react_observer instead."""
from simulation.observe.react_observer import *  # noqa: F401,F403
import simulation.observe.react_observer as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
