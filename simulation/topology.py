"""Backward-compat shim -- import from simulation.core.topology instead."""
from simulation.core.topology import *  # noqa: F401,F403
import simulation.core.topology as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
