"""
FLAME intervention macros — named strategies for population parameter tuning.

The Authority observer selects a macro by name. The engine applies the math.
This keeps the LLM in its zone of competence (pattern recognition, strategy
selection) while keeping ODE parameter tuning out of its hands.

Macros are multiplicative relative to BASE values (captured at simulation start),
not the current value. This prevents compounding: 5x "stabilize" gives 1.5x
dampening, not 1.5^5 = 7.59x.

Cooldown prevents oscillation: each parameter can only be adjusted once
every COOLDOWN_TICKS ticks.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Macro definitions ───────────────────────────────────────────────────────
# Each macro maps parameter names to multipliers applied to base values.

FLAME_MACROS = {
    "stabilize": {
        "dampening": 1.5,
        "drift_sigma": 0.5,
    },
    "loosen": {
        "dampening": 0.7,
        "kappa": 0.8,
    },
    "amplify_influencers": {
        "influencer_weight": 1.3,
    },
    "reduce_coupling": {
        "kappa": 0.7,
    },
    "increase_noise": {
        "drift_sigma": 2.0,
    },
}

# Hard clamps — parameters can never leave these bounds
PARAM_CLAMPS = {
    "kappa": (0.01, 0.5),
    "dampening": (0.1, 2.0),
    "influencer_weight": (1.0, 20.0),
    "drift_sigma": (0.001, 0.1),
    "alpha": (0.01, 0.5),
}

# Minimum ticks between adjustments to the same parameter
COOLDOWN_TICKS = 3


class MacroController:
    """
    Applies named intervention macros to a FlameEngine with safety guards.

    Base values are captured at first use. All multipliers are applied
    relative to these base values, preventing compounding.
    """

    def __init__(self):
        self._base_values: dict[str, float] = {}
        self._last_adjusted: dict[str, int] = {}  # param → last tick adjusted

    def apply(
        self,
        engine,
        macro_name: str,
        tick: int,
    ) -> Optional[dict]:
        """
        Apply a named macro to the FLAME engine.

        Args:
            engine: FlameEngine instance
            macro_name: key in FLAME_MACROS
            tick: current simulation tick

        Returns:
            dict of {param: new_value} applied, or None if blocked by cooldown/unknown
        """
        if macro_name not in FLAME_MACROS:
            logger.warning("Unknown FLAME macro: %s (available: %s)",
                           macro_name, list(FLAME_MACROS.keys()))
            return None

        macro = FLAME_MACROS[macro_name]
        updates = {}
        blocked = []

        for param, multiplier in macro.items():
            # Cooldown check
            last = self._last_adjusted.get(param, -COOLDOWN_TICKS - 1)
            if tick - last < COOLDOWN_TICKS:
                blocked.append(param)
                continue

            # Capture base value on first use
            if param not in self._base_values:
                self._base_values[param] = engine.config.get(param, 1.0)

            # Compute new value from base (not current)
            base = self._base_values[param]
            new_val = base * multiplier

            # Clamp
            if param in PARAM_CLAMPS:
                lo, hi = PARAM_CLAMPS[param]
                new_val = max(lo, min(hi, new_val))

            updates[param] = new_val
            self._last_adjusted[param] = tick

        if blocked:
            logger.info("[Tick %d] Macro '%s' — params on cooldown: %s",
                        tick, macro_name, blocked)

        if updates:
            engine.set_environment(**updates)
            logger.info("[Tick %d] FLAME macro '%s' applied: %s", tick, macro_name, updates)

        return updates if updates else None
