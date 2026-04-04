"""Backward-compat shim -- import from simulation.core.event_stream instead."""
from simulation.core.event_stream import *  # noqa: F401,F403
import simulation.core.event_stream as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
