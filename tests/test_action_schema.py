"""
Tests for the action schema module.

Run: python -m pytest tests/test_action_schema.py -v
"""

import sys
import os
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.action_schema import (
    PARTICIPANT_ACTIONS, OBSERVER_A_ACTIONS, OBSERVER_B_ACTIONS,
    schema_to_prompt, parse_action, extract_narrative,
)


# ── Schema structure tests ────────────────────────────────────────────────────

class TestSchemaStructure:
    def test_participant_actions_keys(self):
        assert set(PARTICIPANT_ACTIONS.keys()) == {"respond", "disengage", "escalate", "withdraw", "engage"}

    def test_observer_a_actions_keys(self):
        assert set(OBSERVER_A_ACTIONS.keys()) == {"analyse"}

    def test_observer_b_actions_keys(self):
        assert set(OBSERVER_B_ACTIONS.keys()) == {"no_intervention", "intervene"}

    def test_all_schemas_have_required_structure(self):
        for schemas in [PARTICIPANT_ACTIONS, OBSERVER_A_ACTIONS, OBSERVER_B_ACTIONS]:
            for name, schema in schemas.items():
                assert "description" in schema, f"{name} missing description"
                assert "parameters" in schema, f"{name} missing parameters"
                assert "properties" in schema["parameters"], f"{name} missing properties"
                assert "required" in schema["parameters"], f"{name} missing required"

    def test_respond_required_fields(self):
        required = set(PARTICIPANT_ACTIONS["respond"]["parameters"]["required"])
        assert "action" in required
        assert "emotion" in required
        assert "narrative" in required

    def test_intervene_enum_values(self):
        enum = OBSERVER_B_ACTIONS["intervene"]["parameters"]["properties"]["intervention_type"]["enum"]
        assert "pause_prompt" in enum
        assert "boundary_warning" in enum
        assert "pacing_adjustment" in enum
        assert "dynamics_dampening" in enum

    def test_analyse_clustering_enum(self):
        enum = OBSERVER_A_ACTIONS["analyse"]["parameters"]["properties"]["clustering"]["enum"]
        assert set(enum) == {"converging", "diverging", "stable", "chaotic"}

    def test_analyse_concern_enum(self):
        enum = OBSERVER_A_ACTIONS["analyse"]["parameters"]["properties"]["concern_level"]["enum"]
        assert set(enum) == {"none", "low", "moderate", "high"}

    def test_narrative_in_all_participant_actions(self):
        for name, schema in PARTICIPANT_ACTIONS.items():
            assert "narrative" in schema["parameters"]["properties"], \
                f"{name} missing narrative field"


# ── schema_to_prompt tests ────────────────────────────────────────────────────

class TestSchemaToPrompt:
    def test_contains_json_instruction(self):
        prompt = schema_to_prompt(PARTICIPANT_ACTIONS, "participant")
        assert "JSON" in prompt

    def test_contains_action_names(self):
        prompt = schema_to_prompt(PARTICIPANT_ACTIONS, "participant")
        assert '"respond"' in prompt
        assert '"disengage"' in prompt
        assert '"escalate"' in prompt

    def test_contains_required_marker(self):
        prompt = schema_to_prompt(PARTICIPANT_ACTIONS, "participant")
        assert "(REQUIRED)" in prompt

    def test_contains_optional_marker(self):
        prompt = schema_to_prompt(PARTICIPANT_ACTIONS, "participant")
        assert "(optional)" in prompt

    def test_contains_example(self):
        prompt = schema_to_prompt(OBSERVER_B_ACTIONS, "observer_b")
        assert '"action"' in prompt

    def test_observer_a_prompt(self):
        prompt = schema_to_prompt(OBSERVER_A_ACTIONS, "observer_a")
        assert '"analyse"' in prompt
        assert "clustering" in prompt

    def test_observer_b_prompt_has_enum(self):
        prompt = schema_to_prompt(OBSERVER_B_ACTIONS, "observer_b")
        assert "pause_prompt" in prompt


# ── parse_action tests ────────────────────────────────────────────────────────

class TestParseAction:
    def test_parse_valid_respond(self):
        response = json.dumps({
            "action": "respond",
            "action_desc": "I scroll through more posts",
            "emotion": "curious",
            "narrative": "I find myself drawn to the content.",
        })
        # Override: parse_action checks for required fields based on schema
        # The "action" key conflicts with the "action" field in respond schema
        # Let's use a proper response
        response = json.dumps({
            "action": "respond",
            "emotion": "curious",
            "narrative": "I find myself drawn to the content and keep scrolling.",
        })
        parsed = parse_action(response, PARTICIPANT_ACTIONS)
        assert parsed is not None
        assert parsed["action"] == "respond"
        assert parsed["emotion"] == "curious"

    def test_parse_valid_disengage(self):
        response = json.dumps({
            "action": "disengage",
            "reason": "I feel overwhelmed",
            "narrative": "I put the phone down and step away.",
        })
        parsed = parse_action(response, PARTICIPANT_ACTIONS)
        assert parsed is not None
        assert parsed["action"] == "disengage"

    def test_parse_valid_escalate(self):
        response = json.dumps({
            "action": "escalate",
            "trigger": "The new feature caught my attention",
            "narrative": "I dove deeper into exploring the new features.",
        })
        parsed = parse_action(response, PARTICIPANT_ACTIONS)
        assert parsed is not None
        assert parsed["action"] == "escalate"

    def test_parse_with_markdown_wrapper(self):
        response = '```json\n{"action": "respond", "emotion": "calm", "narrative": "I take it in stride."}\n```'
        parsed = parse_action(response, PARTICIPANT_ACTIONS)
        assert parsed is not None
        assert parsed["action"] == "respond"

    def test_parse_observer_a(self):
        response = json.dumps({
            "action": "analyse",
            "reasoning": "Looking at the last 3 ticks...",
            "trajectory_summary": "participant_0 rising, participant_1 stable",
            "clustering": "diverging",
            "concern_level": "moderate",
            "flagged_participants": ["participant_0"],
        })
        parsed = parse_action(response, OBSERVER_A_ACTIONS)
        assert parsed is not None
        assert parsed["clustering"] == "diverging"
        assert parsed["concern_level"] == "moderate"

    def test_parse_observer_b_intervene(self):
        response = json.dumps({
            "action": "intervene",
            "intervention_type": "pause_prompt",
            "target": "participant_0",
            "rationale": "Participant is escalating without reflection.",
        })
        parsed = parse_action(response, OBSERVER_B_ACTIONS)
        assert parsed is not None
        assert parsed["intervention_type"] == "pause_prompt"
        assert parsed["target"] == "participant_0"

    def test_parse_observer_b_no_intervention(self):
        response = json.dumps({
            "action": "no_intervention",
            "rationale": "Trajectories are stable.",
        })
        parsed = parse_action(response, OBSERVER_B_ACTIONS)
        assert parsed is not None
        assert parsed["action"] == "no_intervention"

    def test_parse_missing_required_returns_none(self):
        response = json.dumps({
            "action": "respond",
            "emotion": "happy",
            # missing "narrative"
        })
        parsed = parse_action(response, PARTICIPANT_ACTIONS)
        assert parsed is None

    def test_parse_invalid_json_returns_none(self):
        parsed = parse_action("I just keep going, you know?", PARTICIPANT_ACTIONS)
        assert parsed is None

    def test_parse_unknown_action_infers_from_keys(self):
        # Unknown action name, but has engage's required keys (action + narrative)
        response = json.dumps({"action": "teleport", "narrative": "whoosh"})
        parsed = parse_action(response, PARTICIPANT_ACTIONS)
        # Infers first matching schema with (action, narrative) as required fields
        assert parsed is not None
        assert parsed["action"] in ("engage", "escalate")

    def test_parse_truly_unknown_returns_none(self):
        # No matching required field set
        response = json.dumps({"action": "teleport", "x": 1, "y": 2})
        parsed = parse_action(response, PARTICIPANT_ACTIONS)
        assert parsed is None

    def test_parse_infers_action_from_keys(self):
        # Has all of respond's required keys: action, emotion, narrative
        response = json.dumps({
            "action": "unknown_thing",
            "emotion": "anxious",
            "narrative": "I feel uneasy about this.",
        })
        parsed = parse_action(response, PARTICIPANT_ACTIONS)
        # Should infer "respond" since action + emotion + narrative match
        assert parsed is not None
        assert parsed["action"] == "respond"

    def test_parse_intensity_clamped(self):
        response = json.dumps({
            "action": "respond",
            "emotion": "excited",
            "narrative": "This is amazing!",
            "intensity": 1.5,
        })
        parsed = parse_action(response, PARTICIPANT_ACTIONS)
        assert parsed is not None
        assert parsed["intensity"] == 1.0

    def test_parse_intensity_string_coerced(self):
        response = json.dumps({
            "action": "respond",
            "emotion": "calm",
            "narrative": "Taking it easy.",
            "intensity": "0.3",
        })
        parsed = parse_action(response, PARTICIPANT_ACTIONS)
        assert parsed is not None
        assert abs(parsed["intensity"] - 0.3) < 1e-6

    def test_parse_json_embedded_in_text(self):
        response = 'Here is my response: {"action": "respond", "emotion": "curious", "narrative": "Interesting."}'
        parsed = parse_action(response, PARTICIPANT_ACTIONS)
        assert parsed is not None
        assert parsed["action"] == "respond"


# ── extract_narrative tests ───────────────────────────────────────────────────

class TestExtractNarrative:
    def test_from_parsed_with_narrative(self):
        parsed = {"action": "respond", "narrative": "I stepped back carefully."}
        assert extract_narrative(parsed, "raw text") == "I stepped back carefully."

    def test_fallback_to_raw(self):
        assert extract_narrative(None, "raw text here") == "raw text here"

    def test_from_parsed_without_narrative(self):
        parsed = {"action": "respond", "thought": "I need to think.", "reason": "Too fast."}
        result = extract_narrative(parsed, "raw text")
        assert "I need to think" in result

    def test_empty_parsed(self):
        parsed = {}
        result = extract_narrative(parsed, "fallback text")
        assert result == "fallback text"


# ── Integration with sim_loop imports ─────────────────────────────────────────

class TestSimLoopIntegration:
    def test_sim_loop_imports_action_schema(self):
        from simulation.sim_loop import (
            PARTICIPANT_ACTIONS, OBSERVER_A_ACTIONS, OBSERVER_B_ACTIONS,
            schema_to_prompt, parse_action, extract_narrative,
        )

    def test_participant_agent_has_schema_in_prompt(self):
        from simulation.sim_loop import ParticipantAgent, set_seed
        set_seed(42)
        p = ParticipantAgent("p_0")
        system = p._build_system_prompt()
        # Schema is injected in step(), not _build_system_prompt()
        # So the base prompt should NOT contain JSON instructions
        assert "JSON" not in system
