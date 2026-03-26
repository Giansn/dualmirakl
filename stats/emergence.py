"""
Emergence metrics — complements simulation/dynamics.py.

This module provides SALib-based Sobol sensitivity analysis (the dynamics.py
version uses a custom implementation). Use this for production SA with
proper Saltelli sampling; use dynamics.py for quick in-simulation estimates.

Transfer entropy is already in dynamics.py — this re-exports it for
convenient stats/ namespace access.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def sobol_sensitivity(
    param_ranges: dict[str, tuple[float, float]],
    model_fn,
    n_samples: int = 1024,
) -> dict:
    """
    Sobol Sensitivity Analysis via SALib.

    param_ranges: {"openness": (0.0, 1.0), "stance": (-1.0, 1.0), ...}
    model_fn: f(params_dict) -> float (sim result, e.g., final polarization)

    Returns: First-order (S1) and Total-order (ST) indices per parameter.

    Requires SALib: pip install SALib
    """
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol
    except ImportError:
        logger.warning("SALib not installed. pip install SALib")
        return {"error": "SALib not installed"}

    problem = {
        "num_vars": len(param_ranges),
        "names": list(param_ranges.keys()),
        "bounds": list(param_ranges.values()),
    }

    param_values = saltelli.sample(problem, n_samples, calc_second_order=True)

    Y = np.zeros(param_values.shape[0])
    for i, params in enumerate(param_values):
        param_dict = {name: float(val) for name, val
                      in zip(problem["names"], params)}
        Y[i] = model_fn(param_dict)

    si = sobol.analyze(problem, Y, calc_second_order=True)

    return {
        "first_order": {name: float(s1) for name, s1
                        in zip(problem["names"], si["S1"])},
        "total_order": {name: float(st) for name, st
                        in zip(problem["names"], si["ST"])},
        "second_order": si["S2"].tolist() if "S2" in si else None,
        "first_order_conf": {name: float(c) for name, c
                             in zip(problem["names"], si["S1_conf"])},
    }


# Re-export dynamics.py transfer entropy for stats/ namespace convenience
def transfer_entropy_matrix(agent_series: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Transfer entropy matrix: TE[i,j] = TE from agent i to agent j.
    Delegates to simulation.dynamics.transfer_entropy_matrix.
    """
    from simulation.dynamics import transfer_entropy_matrix as _te_matrix
    return _te_matrix(agent_series, k=k)
