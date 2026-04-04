"""
Transition function registry for archetype migration.

Transition functions determine when an agent changes archetype profile
during the simulation. Referenced by name from scenario.yaml.

Signature: fn(agent_state: dict, tick_history: list[dict], **params) -> bool
  agent_state: {"score": float, "archetype": str, "tick": int, ...}
  tick_history: [{"tick": int, "score": float}, ...] (chronological)
  **params: from scenario.yaml transitions[].params

Usage:
    @register_transition("my_custom_transition")
    def my_transition(agent_state, tick_history, **params):
        return agent_state["score"] > params.get("threshold", 0.5)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Callable] = {}


def register_transition(name: str):
    """Decorator to register a transition function by name."""
    def decorator(fn: Callable):
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_transition(name: str) -> Callable:
    """Look up a registered transition function. Raises KeyError if not found."""
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Transition function '{name}' not registered. "
            f"Available: {available}"
        )
    return _REGISTRY[name]


def list_transitions() -> list[str]:
    """List all registered transition function names."""
    return sorted(_REGISTRY.keys())


# ── Built-in transition functions ─────────────────────────────────────────────


@register_transition("escalation_sustained")
def escalation_sustained(
    agent_state: dict,
    tick_history: list[dict],
    **params: Any,
) -> bool:
    """
    Trigger when score stays above threshold for N consecutive ticks.
    Params: threshold (float), consecutive_ticks (int)
    """
    threshold = params.get("threshold", 0.8)
    n = params.get("consecutive_ticks", 3)
    if len(tick_history) < n:
        return False
    return all(
        entry.get("score", 0.0) > threshold
        for entry in tick_history[-n:]
    )


@register_transition("recovery_sustained")
def recovery_sustained(
    agent_state: dict,
    tick_history: list[dict],
    **params: Any,
) -> bool:
    """
    Trigger when score stays below threshold for N consecutive ticks.
    Params: threshold (float), consecutive_ticks (int)
    """
    threshold = params.get("threshold", 0.3)
    n = params.get("consecutive_ticks", 5)
    if len(tick_history) < n:
        return False
    return all(
        entry.get("score", 0.0) < threshold
        for entry in tick_history[-n:]
    )


@register_transition("threshold_cross")
def threshold_cross(
    agent_state: dict,
    tick_history: list[dict],
    **params: Any,
) -> bool:
    """
    Trigger on single-tick threshold crossing.
    Params: threshold (float), direction ("up" | "down")
    """
    threshold = params.get("threshold", 0.5)
    direction = params.get("direction", "up")
    if len(tick_history) < 2:
        return False
    prev = tick_history[-2].get("score", 0.0)
    curr = tick_history[-1].get("score", 0.0)
    if direction == "up":
        return prev < threshold <= curr
    elif direction == "down":
        return prev >= threshold > curr
    return False


@register_transition("oscillation_detect")
def oscillation_detect(
    agent_state: dict,
    tick_history: list[dict],
    **params: Any,
) -> bool:
    """
    Trigger when score variance exceeds amplitude threshold in a window.
    Params: window (int), amplitude (float)
    """
    window = params.get("window", 5)
    amplitude = params.get("amplitude", 0.2)
    if len(tick_history) < window:
        return False
    scores = [e.get("score", 0.0) for e in tick_history[-window:]]
    score_range = max(scores) - min(scores)
    return score_range > amplitude
