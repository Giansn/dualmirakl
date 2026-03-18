"""
Safety classification and observer mode enforcement for dualmirakl.

Combines two patterns from AI coding agent architectures:

  #4 — Observer Mode (from Devin's planning vs execution):
    Externally-controlled mode switching. The orchestrator sets which mode
    the observer is in — the agent never decides. ObserverMode.ANALYSE
    forbids intervention output; ObserverMode.INTERVENE allows it.
    Mandatory reasoning gates at phase transitions.

  #5 — Safety Classification (from Windsurf's SafeToAutoRun):
    Every agent action is classified into a safety tier:
      AUTO    — execute without review (read-only, no side effects)
      REVIEW  — execute and log for post-hoc review
      APPROVE — require explicit approval before execution
    Non-overridable by agents. Allowlist as the only escape hatch.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# #4 — OBSERVER MODE (Devin-inspired)
# ═══════════════════════════════════════════════════════════════════════════════


class ObserverMode(Enum):
    """
    Phase D mode — set by the orchestrator, not the agent.

    ANALYSE:   gather info, describe dynamics — NO interventions allowed.
               Output must include a reasoning artifact.
               Compliance gate: reject if intervention keywords detected.

    INTERVENE: propose interventions based on prior analysis.
               Receives analysis from ANALYSE phase as input.
               Output processed through structured extraction + codebook fallback.
    """
    ANALYSE = "analyse"
    INTERVENE = "intervene"


def validate_observer_output(
    response: str,
    mode: ObserverMode,
    parsed: Optional[dict] = None,
) -> list[str]:
    """
    Validate observer output against the current mode constraints.

    Returns list of violation descriptions (empty = valid).

    ANALYSE mode violations:
      - Missing reasoning field in structured output
      - Intervention keywords in response (already caught by compliance,
        but this adds mode-level enforcement)

    INTERVENE mode violations:
      - None currently — intervene mode is permissive
    """
    violations = []

    if mode == ObserverMode.ANALYSE:
        # Mandatory reasoning gate: structured output must include reasoning
        if parsed and not parsed.get("reasoning"):
            violations.append("analyse_mode: missing required 'reasoning' field")

        # Mode-level intervention keyword check (defense in depth)
        # These are the codebook trigger phrases that should never appear in analyse
        _intervention_phrases = [
            "take a pause", "pause and reflect", "step back for a moment",
            "approaching scenario boundary", "flag boundary issue",
            "dampen the behavioural dynamics", "reduce score momentum",
            "adjust the pacing", "slow down the interaction",
        ]
        response_lower = response.lower()
        for phrase in _intervention_phrases:
            if phrase in response_lower:
                violations.append(f"analyse_mode: contains intervention phrase '{phrase}'")
                break  # one violation is enough

    return violations


# ═══════════════════════════════════════════════════════════════════════════════
# #5 — SAFETY CLASSIFICATION (Windsurf-inspired)
# ═══════════════════════════════════════════════════════════════════════════════


class SafetyTier(Enum):
    """
    Action safety classification.

    AUTO    — execute without review. No side effects beyond internal state.
    REVIEW  — execute and flag in event stream for post-hoc review.
    APPROVE — block execution until explicitly approved. For actions that
              modify scoring formulas, remove agents, or change parameters.

    Non-overridable by agents or users within the simulation.
    Only the allowlist (persistent config) can upgrade APPROVE → AUTO.
    """
    AUTO = "auto"
    REVIEW = "review"
    APPROVE = "approve"


# Default safety classification for all known actions
ACTION_SAFETY: dict[str, SafetyTier] = {
    # ── Participant actions (Phase B) — all auto, no external side effects
    "respond": SafetyTier.AUTO,
    "disengage": SafetyTier.AUTO,
    "escalate": SafetyTier.AUTO,

    # ── Observer A actions (Phase D1) — read-only analysis
    "analyse": SafetyTier.AUTO,

    # ── Observer B actions (Phase D2) — tiered by impact
    "no_intervention": SafetyTier.AUTO,
    "intervene.pause_prompt": SafetyTier.REVIEW,
    "intervene.boundary_warning": SafetyTier.REVIEW,
    "intervene.pacing_adjustment": SafetyTier.REVIEW,
    "intervene.dynamics_dampening": SafetyTier.APPROVE,

    # ── Future actions (when agents get side effects)
    "modify_parameter": SafetyTier.APPROVE,
    "terminate_agent": SafetyTier.APPROVE,
    "inject_event": SafetyTier.REVIEW,
}


class SafetyGate:
    """
    Evaluates and enforces safety tiers for agent actions.

    The allowlist provides the only mechanism to override safety classification.
    This mirrors Windsurf's design: the agent cannot override safety,
    and users can only override via persistent settings (not chat).

    Usage:
        gate = SafetyGate(allowlist={"intervene.dynamics_dampening"})
        result = gate.check("intervene.dynamics_dampening")
        # Returns SafetyTier.AUTO (overridden by allowlist)
    """

    def __init__(self, allowlist: Optional[set[str]] = None):
        self._allowlist = allowlist or set()

    def classify(self, action_key: str) -> SafetyTier:
        """
        Get the safety tier for an action.
        Allowlisted actions are always AUTO.
        Unknown actions default to APPROVE (fail-safe).
        """
        if action_key in self._allowlist:
            return SafetyTier.AUTO
        return ACTION_SAFETY.get(action_key, SafetyTier.APPROVE)

    def check(self, action_key: str, payload: dict) -> dict:
        """
        Evaluate an action and return a decision dict.

        Returns:
            {
                "action": str,
                "tier": SafetyTier,
                "allowed": bool,       # False only for APPROVE tier
                "reason": str,
            }
        """
        tier = self.classify(action_key)
        return {
            "action": action_key,
            "tier": tier,
            "allowed": tier != SafetyTier.APPROVE,
            "reason": (
                f"Action '{action_key}' requires approval (tier={tier.value})"
                if tier == SafetyTier.APPROVE
                else f"Action '{action_key}' is {tier.value}"
            ),
        }

    def evaluate_intervention(self, intervention_type: str) -> dict:
        """
        Convenience method for intervention actions.
        Prepends 'intervene.' to match the ACTION_SAFETY keys.
        """
        action_key = f"intervene.{intervention_type}"
        return self.check(action_key, {})

    @property
    def allowlist(self) -> set[str]:
        return set(self._allowlist)

    def add_to_allowlist(self, action_key: str) -> None:
        self._allowlist.add(action_key)

    def remove_from_allowlist(self, action_key: str) -> None:
        self._allowlist.discard(action_key)
