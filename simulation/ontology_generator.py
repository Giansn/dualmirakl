"""Backward-compat shim -- import from simulation.knowledge.ontology_generator instead."""
from simulation.knowledge.ontology_generator import *  # noqa: F401,F403
import simulation.knowledge.ontology_generator as _mod  # noqa: F811
globals().update({k: getattr(_mod, k) for k in dir(_mod) if k.startswith("_") and not k.startswith("__")})
