"""
Tests for observer mode enforcement and safety classification.

Run: python -m pytest tests/test_safety.py -v
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.safety import (
    ObserverMode, SafetyTier, SafetyGate,
    validate_observer_output, ACTION_SAFETY,
)


# ═══════════════════════════════════════════════════════════════════════════════
# #4 — OBSERVER MODE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestObserverMode:
    def test_enum_values(self):
        assert ObserverMode.ANALYSE.value == "analyse"
        assert ObserverMode.INTERVENE.value == "intervene"

    def test_analyse_clean_with_reasoning(self):
        parsed = {
            "action": "analyse",
            "reasoning": "Looking at the data, I see divergence.",
            "trajectory_summary": "participant_0 rising",
            "clustering": "diverging",
            "concern_level": "moderate",
        }
        violations = validate_observer_output(
            "Participant 0 is rising steadily.", ObserverMode.ANALYSE, parsed,
        )
        assert violations == []

    def test_analyse_missing_reasoning(self):
        parsed = {
            "action": "analyse",
            "trajectory_summary": "stable",
            "clustering": "stable",
            "concern_level": "none",
            # no "reasoning" field
        }
        violations = validate_observer_output(
            "Everything looks stable.", ObserverMode.ANALYSE, parsed,
        )
        assert len(violations) == 1
        assert "reasoning" in violations[0]

    def test_analyse_with_intervention_phrase(self):
        response = "I suggest we take a pause and reflect on the trajectory."
        violations = validate_observer_output(
            response, ObserverMode.ANALYSE, None,
        )
        assert len(violations) >= 1
        assert "intervention phrase" in violations[0]

    def test_analyse_boundary_phrase(self):
        response = "We should flag boundary issue here."
        violations = validate_observer_output(
            response, ObserverMode.ANALYSE, None,
        )
        assert len(violations) >= 1

    def test_analyse_dampening_phrase(self):
        response = "We need to dampen the behavioural dynamics."
        violations = validate_observer_output(
            response, ObserverMode.ANALYSE, None,
        )
        assert len(violations) >= 1

    def test_analyse_pacing_phrase(self):
        response = "I recommend to slow down the interaction pace."
        violations = validate_observer_output(
            response, ObserverMode.ANALYSE, None,
        )
        assert len(violations) >= 1

    def test_analyse_clean_no_parsed(self):
        response = "Participant 0 has been stable over the last 3 ticks."
        violations = validate_observer_output(
            response, ObserverMode.ANALYSE, None,
        )
        assert violations == []

    def test_intervene_mode_permissive(self):
        response = "Take a pause and reflect. Dampen the behavioural dynamics."
        violations = validate_observer_output(
            response, ObserverMode.INTERVENE, None,
        )
        assert violations == []

    def test_analyse_case_insensitive(self):
        response = "We should TAKE A PAUSE here."
        violations = validate_observer_output(
            response, ObserverMode.ANALYSE, None,
        )
        assert len(violations) >= 1

    def test_analyse_multiple_violations_reports_one(self):
        """Only reports first intervention phrase found (not all)."""
        response = "Take a pause. Also dampen the behavioural dynamics."
        violations = validate_observer_output(
            response, ObserverMode.ANALYSE, None,
        )
        # Should stop at first match
        assert len(violations) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# #5 — SAFETY TIER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafetyTier:
    def test_enum_values(self):
        assert SafetyTier.AUTO.value == "auto"
        assert SafetyTier.REVIEW.value == "review"
        assert SafetyTier.APPROVE.value == "approve"

    def test_ordering(self):
        # Verify all three tiers exist as distinct values
        assert len(set(t.value for t in SafetyTier)) == 3


class TestActionSafety:
    def test_participant_actions_are_auto(self):
        assert ACTION_SAFETY["respond"] == SafetyTier.AUTO
        assert ACTION_SAFETY["disengage"] == SafetyTier.AUTO
        assert ACTION_SAFETY["escalate"] == SafetyTier.AUTO

    def test_analyse_is_auto(self):
        assert ACTION_SAFETY["analyse"] == SafetyTier.AUTO

    def test_no_intervention_is_auto(self):
        assert ACTION_SAFETY["no_intervention"] == SafetyTier.AUTO

    def test_pause_prompt_is_review(self):
        assert ACTION_SAFETY["intervene.pause_prompt"] == SafetyTier.REVIEW

    def test_boundary_warning_is_review(self):
        assert ACTION_SAFETY["intervene.boundary_warning"] == SafetyTier.REVIEW

    def test_pacing_adjustment_is_review(self):
        assert ACTION_SAFETY["intervene.pacing_adjustment"] == SafetyTier.REVIEW

    def test_dynamics_dampening_is_approve(self):
        assert ACTION_SAFETY["intervene.dynamics_dampening"] == SafetyTier.APPROVE

    def test_future_actions(self):
        assert ACTION_SAFETY["modify_parameter"] == SafetyTier.APPROVE
        assert ACTION_SAFETY["terminate_agent"] == SafetyTier.APPROVE
        assert ACTION_SAFETY["inject_event"] == SafetyTier.REVIEW


class TestSafetyGate:
    def test_classify_known_action(self):
        gate = SafetyGate()
        assert gate.classify("respond") == SafetyTier.AUTO
        assert gate.classify("intervene.dynamics_dampening") == SafetyTier.APPROVE

    def test_classify_unknown_defaults_approve(self):
        gate = SafetyGate()
        assert gate.classify("unknown_action") == SafetyTier.APPROVE

    def test_allowlist_overrides(self):
        gate = SafetyGate(allowlist={"intervene.dynamics_dampening"})
        assert gate.classify("intervene.dynamics_dampening") == SafetyTier.AUTO

    def test_check_auto_allowed(self):
        gate = SafetyGate()
        result = gate.check("respond", {})
        assert result["allowed"] is True
        assert result["tier"] == SafetyTier.AUTO

    def test_check_review_allowed(self):
        gate = SafetyGate()
        result = gate.check("intervene.pause_prompt", {})
        assert result["allowed"] is True
        assert result["tier"] == SafetyTier.REVIEW

    def test_check_approve_blocked(self):
        gate = SafetyGate()
        result = gate.check("intervene.dynamics_dampening", {})
        assert result["allowed"] is False
        assert result["tier"] == SafetyTier.APPROVE
        assert "requires approval" in result["reason"]

    def test_check_approve_with_allowlist(self):
        gate = SafetyGate(allowlist={"intervene.dynamics_dampening"})
        result = gate.check("intervene.dynamics_dampening", {})
        assert result["allowed"] is True
        assert result["tier"] == SafetyTier.AUTO

    def test_evaluate_intervention_convenience(self):
        gate = SafetyGate()
        result = gate.evaluate_intervention("pause_prompt")
        assert result["action"] == "intervene.pause_prompt"
        assert result["tier"] == SafetyTier.REVIEW
        assert result["allowed"] is True

    def test_evaluate_intervention_blocked(self):
        gate = SafetyGate()
        result = gate.evaluate_intervention("dynamics_dampening")
        assert result["allowed"] is False

    def test_allowlist_property(self):
        gate = SafetyGate(allowlist={"a", "b"})
        assert gate.allowlist == {"a", "b"}

    def test_add_to_allowlist(self):
        gate = SafetyGate()
        gate.add_to_allowlist("intervene.dynamics_dampening")
        assert gate.classify("intervene.dynamics_dampening") == SafetyTier.AUTO

    def test_remove_from_allowlist(self):
        gate = SafetyGate(allowlist={"intervene.dynamics_dampening"})
        gate.remove_from_allowlist("intervene.dynamics_dampening")
        assert gate.classify("intervene.dynamics_dampening") == SafetyTier.APPROVE

    def test_remove_nonexistent_safe(self):
        gate = SafetyGate()
        gate.remove_from_allowlist("nonexistent")  # should not raise


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH WORLDSTATE
# ═══════════════════════════════════════════════════════════════════════════════


class TestWorldStateIntegration:
    def test_worldstate_has_safety_gate(self):
        from simulation.sim_loop import WorldState
        ws = WorldState(k=3)
        assert isinstance(ws.safety_gate, SafetyGate)

    def test_worldstate_safety_gate_default(self):
        from simulation.sim_loop import WorldState
        ws = WorldState(k=3)
        # Default gate has no allowlist
        assert ws.safety_gate.allowlist == set()

    def test_worldstate_with_custom_gate(self):
        from simulation.sim_loop import WorldState
        gate = SafetyGate(allowlist={"intervene.dynamics_dampening"})
        ws = WorldState(k=3, safety_gate=gate)
        result = ws.safety_gate.evaluate_intervention("dynamics_dampening")
        assert result["allowed"] is True

    def test_sim_loop_imports(self):
        from simulation.sim_loop import (
            ObserverMode, SafetyTier, SafetyGate,
        )
