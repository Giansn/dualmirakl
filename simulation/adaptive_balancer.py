"""Backward-compat shim -- import from simulation.gpu.balancer instead."""
from simulation.gpu.balancer import *  # noqa: F401,F403
import simulation.gpu.balancer as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
