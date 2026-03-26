"""
Tests for the ontology generator module.

Run: python -m pytest tests/test_ontology_generator.py -v
"""

import sys
import os
import json
import asyncio

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, AsyncMock

from simulation.ontology_generator import (
    build_prompt,
    parse_llm_output,
    validate_ontology,
    generate_ontology,
    REGISTERED_FUNCTIONS,
)
from simulation.scenario import ScenarioConfig


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio needed)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Sample LLM output for mocking ────────────────────────────────────────────

def _valid_llm_output(n_profiles: int = 4) -> str:
    """Return a valid JSON string that matches the expected ontology schema."""
    profiles = []
    dist = {}
    fraction = round(1.0 / n_profiles, 2)
    # Adjust last fraction so they sum to exactly 1.0
    for i in range(n_profiles):
        pid = f"K{i + 1}"
        susc = ["low", "medium", "high"][i % 3]
        resil = ["high", "medium", "low"][i % 3]
        profiles.append({
            "id": pid,
            "label": f"Archetype {i + 1}",
            "description": f"Description for archetype {i + 1}",
            "properties": {
                "susceptibility": susc,
                "resilience": resil,
            },
        })
        if i < n_profiles - 1:
            dist[pid] = fraction
        else:
            dist[pid] = round(1.0 - fraction * (n_profiles - 1), 2)

    data = {
        "archetypes": {
            "profiles": profiles,
            "distribution": dist,
        },
        "transitions": [
            {
                "from": "K1",
                "to": "K2",
                "function": "escalation_sustained",
                "params": {"threshold": 0.8, "consecutive_ticks": 3},
            },
            {
                "from": "K2",
                "to": "K1",
                "function": "recovery_sustained",
                "params": {"threshold": 0.3, "consecutive_ticks": 5},
            },
        ],
    }
    return json.dumps(data)


def _valid_llm_output_with_fences() -> str:
    """Return valid output wrapped in markdown code fences."""
    return "```json\n" + _valid_llm_output() + "\n```"


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildPrompt:
    def test_returns_valid_json(self):
        result = build_prompt("Some document text", "test-scenario", 4)
        data = json.loads(result)
        assert "system" in data
        assert "user" in data

    def test_includes_scenario_name(self):
        result = build_prompt("doc", "my-scenario", 3)
        data = json.loads(result)
        assert "my-scenario" in data["user"]

    def test_includes_n_profiles(self):
        result = build_prompt("doc", "test", 5)
        data = json.loads(result)
        assert "5" in data["user"]

    def test_includes_document_text(self):
        result = build_prompt("The domain is about urban planning.", "test", 4)
        data = json.loads(result)
        assert "urban planning" in data["user"]

    def test_includes_registered_functions(self):
        result = build_prompt("doc", "test", 4)
        data = json.loads(result)
        for fn in REGISTERED_FUNCTIONS:
            assert fn in data["user"], f"Missing function name: {fn}"

    def test_system_prompt_mentions_json(self):
        result = build_prompt("doc", "test", 4)
        data = json.loads(result)
        assert "JSON" in data["system"]

    def test_user_prompt_specifies_distribution_sum(self):
        result = build_prompt("doc", "test", 4)
        data = json.loads(result)
        assert "sum to" in data["user"].lower() or "1.0" in data["user"]


# ══════════════════════════════════════════════════════════════════════════════
# PARSING TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestParseLlmOutput:
    def test_parse_valid_json(self):
        raw = _valid_llm_output()
        data = parse_llm_output(raw)
        assert "archetypes" in data
        assert "transitions" in data

    def test_parse_with_markdown_fences(self):
        raw = _valid_llm_output_with_fences()
        data = parse_llm_output(raw)
        assert "archetypes" in data
        assert len(data["archetypes"]["profiles"]) == 4

    def test_parse_with_whitespace(self):
        raw = "  \n  " + _valid_llm_output() + "  \n  "
        data = parse_llm_output(raw)
        assert "archetypes" in data

    def test_parse_invalid_json_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_llm_output("This is not JSON at all")

    def test_parse_missing_archetypes_raises(self):
        raw = json.dumps({"transitions": []})
        with pytest.raises(ValueError, match="Missing 'archetypes'"):
            parse_llm_output(raw)

    def test_parse_missing_transitions_raises(self):
        raw = json.dumps({"archetypes": {"profiles": [], "distribution": {}}})
        with pytest.raises(ValueError, match="Missing 'transitions'"):
            parse_llm_output(raw)

    def test_parse_non_object_raises(self):
        with pytest.raises(ValueError, match="Expected JSON object"):
            parse_llm_output("[1, 2, 3]")

    def test_parse_profiles_not_list_raises(self):
        raw = json.dumps({
            "archetypes": {"profiles": "not a list", "distribution": {}},
            "transitions": [],
        })
        with pytest.raises(ValueError, match="profiles must be a list"):
            parse_llm_output(raw)

    def test_parse_distribution_not_dict_raises(self):
        raw = json.dumps({
            "archetypes": {"profiles": [], "distribution": [0.5, 0.5]},
            "transitions": [],
        })
        with pytest.raises(ValueError, match="distribution must be a dict"):
            parse_llm_output(raw)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateOntology:
    def test_valid_ontology_no_errors(self):
        data = json.loads(_valid_llm_output())
        errors = validate_ontology(data)
        assert errors == []

    def test_distribution_sum_not_one(self):
        data = json.loads(_valid_llm_output(2))
        data["archetypes"]["distribution"] = {"K1": 0.3, "K2": 0.3}
        errors = validate_ontology(data)
        assert any("sum" in e for e in errors)

    def test_distribution_sum_close_to_one(self):
        """Distribution within tolerance (0.01) should pass."""
        data = json.loads(_valid_llm_output(2))
        data["archetypes"]["distribution"] = {"K1": 0.505, "K2": 0.5}
        errors = validate_ontology(data)
        sum_errors = [e for e in errors if "sum" in e]
        assert sum_errors == []

    def test_distribution_references_undefined_profile(self):
        data = json.loads(_valid_llm_output(2))
        data["archetypes"]["distribution"]["K99"] = 0.1
        errors = validate_ontology(data)
        assert any("K99" in e for e in errors)

    def test_profile_missing_distribution(self):
        data = json.loads(_valid_llm_output(2))
        # Remove K2 from distribution
        del data["archetypes"]["distribution"]["K2"]
        errors = validate_ontology(data)
        assert any("K2" in e and "no distribution" in e for e in errors)

    def test_transition_invalid_from_profile(self):
        data = json.loads(_valid_llm_output())
        data["transitions"][0]["from"] = "NONEXISTENT"
        errors = validate_ontology(data)
        assert any("NONEXISTENT" in e for e in errors)

    def test_transition_invalid_to_profile(self):
        data = json.loads(_valid_llm_output())
        data["transitions"][0]["to"] = "NONEXISTENT"
        errors = validate_ontology(data)
        assert any("NONEXISTENT" in e for e in errors)

    def test_transition_unregistered_function(self):
        data = json.loads(_valid_llm_output())
        data["transitions"][0]["function"] = "made_up_function"
        errors = validate_ontology(data)
        assert any("made_up_function" in e for e in errors)

    def test_all_registered_functions_accepted(self):
        """Each built-in function name should pass validation."""
        for fn_name in REGISTERED_FUNCTIONS:
            data = json.loads(_valid_llm_output(2))
            data["transitions"] = [{
                "from": "K1",
                "to": "K2",
                "function": fn_name,
                "params": {},
            }]
            errors = validate_ontology(data)
            fn_errors = [e for e in errors if "unregistered" in e]
            assert fn_errors == [], f"{fn_name} should be accepted"

    def test_duplicate_profile_id(self):
        data = json.loads(_valid_llm_output(2))
        # Duplicate the first profile
        data["archetypes"]["profiles"].append(
            data["archetypes"]["profiles"][0].copy()
        )
        errors = validate_ontology(data)
        assert any("Duplicate" in e for e in errors)

    def test_profile_missing_id(self):
        data = json.loads(_valid_llm_output(2))
        data["archetypes"]["profiles"][0]["id"] = ""
        errors = validate_ontology(data)
        assert any("missing 'id'" in e for e in errors)

    def test_profile_missing_label(self):
        data = json.loads(_valid_llm_output(2))
        data["archetypes"]["profiles"][0]["label"] = ""
        errors = validate_ontology(data)
        assert any("missing 'label'" in e for e in errors)


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE_ONTOLOGY INTEGRATION TESTS (mocked LLM)
# ══════════════════════════════════════════════════════════════════════════════


class TestGenerateOntology:
    def test_generates_valid_ontology(self):
        mock_output = _valid_llm_output(4)

        with patch("orchestrator.agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = mock_output
            result = _run(generate_ontology("domain doc text", "test-scenario", 4))

        assert "archetypes" in result
        assert "transitions" in result
        assert len(result["archetypes"]["profiles"]) == 4
        assert len(result["transitions"]) == 2

        # Verify agent_turn was called with correct backend
        mock_turn.assert_called_once()
        call_kwargs = mock_turn.call_args
        assert call_kwargs[1]["backend"] == "authority" or call_kwargs[0][1] == "authority"

    def test_handles_markdown_fenced_output(self):
        mock_output = _valid_llm_output_with_fences()

        with patch("orchestrator.agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = mock_output
            result = _run(generate_ontology("doc", "test", 4))

        assert len(result["archetypes"]["profiles"]) == 4

    def test_raises_on_invalid_llm_output(self):
        with patch("orchestrator.agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = "Sorry, I cannot do that."
            with pytest.raises(ValueError, match="not valid JSON"):
                _run(generate_ontology("doc", "test"))

    def test_raises_on_validation_failure(self):
        bad_data = json.loads(_valid_llm_output(2))
        bad_data["transitions"][0]["function"] = "nonexistent_fn"
        mock_output = json.dumps(bad_data)

        with patch("orchestrator.agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = mock_output
            with pytest.raises(ValueError, match="validation failed"):
                _run(generate_ontology("doc", "test", 2))

    def test_raises_runtime_error_on_llm_failure(self):
        with patch("orchestrator.agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.side_effect = ConnectionError("server down")
            with pytest.raises(RuntimeError, match="LLM call failed"):
                _run(generate_ontology("doc", "test"))

    def test_distribution_sums_to_one(self):
        mock_output = _valid_llm_output(4)

        with patch("orchestrator.agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = mock_output
            result = _run(generate_ontology("doc", "test", 4))

        total = sum(result["archetypes"]["distribution"].values())
        assert abs(total - 1.0) <= 0.01

    def test_n_profiles_parameter(self):
        for n in [2, 3, 5]:
            mock_output = _valid_llm_output(n)
            with patch("orchestrator.agent_turn", new_callable=AsyncMock) as mock_turn:
                mock_turn.return_value = mock_output
                result = _run(generate_ontology("doc", "test", n))
            assert len(result["archetypes"]["profiles"]) == n

    def test_prompt_includes_document_text(self):
        mock_output = _valid_llm_output()
        doc_text = "Unique document about coral reef ecosystems"

        with patch("orchestrator.agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = mock_output
            _run(generate_ontology(doc_text, "reef-sim", 4))

        # Check that the user_message sent to agent_turn contains the document
        call_args = mock_turn.call_args
        # agent_turn signature: (agent_id, backend, system_prompt, user_message, ...)
        # Could be positional or keyword
        if call_args[1]:
            user_msg = call_args[1].get("user_message", "")
        else:
            user_msg = ""
        if not user_msg and len(call_args[0]) >= 4:
            user_msg = call_args[0][3]
        assert "coral reef" in user_msg


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIOCONFIG INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestScenarioConfigIntegration:
    """Verify that generated ontology output can be loaded into ScenarioConfig."""

    def _base_scenario(self) -> dict:
        """Minimal scenario dict without archetypes/transitions."""
        return {
            "meta": {"name": "ontology-test", "description": "Generated ontology test"},
            "agents": {
                "roles": [
                    {
                        "id": "participant_template",
                        "slot": "swarm",
                        "type": "participant",
                        "system_prompt": "You are a test agent.",
                        "count": 4,
                    },
                ],
            },
        }

    def test_merge_into_scenario_config(self):
        ontology = json.loads(_valid_llm_output(3))
        scenario = self._base_scenario()
        scenario["archetypes"] = ontology["archetypes"]
        scenario["transitions"] = ontology["transitions"]

        config = ScenarioConfig.from_dict(scenario)
        assert len(config.archetypes.profiles) == 3
        assert len(config.transitions) == 2

    def test_merged_config_validates(self):
        ontology = json.loads(_valid_llm_output(3))
        scenario = self._base_scenario()
        scenario["archetypes"] = ontology["archetypes"]
        scenario["transitions"] = ontology["transitions"]

        config = ScenarioConfig.from_dict(scenario)
        report = config.validate_scenario(strict=False)
        assert report["valid"] is True, f"Errors: {report['errors']}"

    def test_merged_profiles_accessible(self):
        ontology = json.loads(_valid_llm_output(3))
        scenario = self._base_scenario()
        scenario["archetypes"] = ontology["archetypes"]
        scenario["transitions"] = ontology["transitions"]

        config = ScenarioConfig.from_dict(scenario)
        profile = config.get_profile("K1")
        assert profile is not None
        assert profile.label == "Archetype 1"
        assert "susceptibility" in profile.properties

    def test_distribution_fractions_preserved(self):
        ontology = json.loads(_valid_llm_output(3))
        scenario = self._base_scenario()
        scenario["archetypes"] = ontology["archetypes"]
        scenario["transitions"] = ontology["transitions"]

        config = ScenarioConfig.from_dict(scenario)
        total = sum(config.archetypes.distribution.values())
        assert abs(total - 1.0) <= 0.01

    def test_transition_rules_preserved(self):
        ontology = json.loads(_valid_llm_output(3))
        scenario = self._base_scenario()
        scenario["archetypes"] = ontology["archetypes"]
        scenario["transitions"] = ontology["transitions"]

        config = ScenarioConfig.from_dict(scenario)
        assert config.transitions[0].from_profile == "K1"
        assert config.transitions[0].to_profile == "K2"
        assert config.transitions[0].function == "escalation_sustained"

    def test_full_roundtrip_with_mock_llm(self):
        """End-to-end: generate_ontology -> merge -> ScenarioConfig.from_dict."""
        mock_output = _valid_llm_output(4)

        with patch("orchestrator.agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = mock_output
            ontology = _run(generate_ontology("domain text", "roundtrip", 4))

        scenario = self._base_scenario()
        scenario["archetypes"] = ontology["archetypes"]
        scenario["transitions"] = ontology["transitions"]

        config = ScenarioConfig.from_dict(scenario)
        report = config.validate_scenario(strict=False)
        assert report["valid"] is True, f"Errors: {report['errors']}"
        assert config.participant_count() == 4
        assert len(config.archetypes.profiles) == 4
        assert len(config.transitions) == 2
