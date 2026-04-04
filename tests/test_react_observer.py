"""
Tests for the ReACT observer module.

Run: python -m pytest tests/test_react_observer.py -v
"""

import sys
import os
import json
import asyncio

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.react_observer import (
    OBSERVER_TOOLS,
    TOOL_SAFETY,
    ReactObserver,
    _parse_react_response,
    _tools_to_prompt,
    _exec_query_scores,
    _exec_query_events,
    _exec_check_interventions,
    _exec_query_memory,
)
from simulation.event_stream import EventStream, TOOL_USE, ALL_EVENT_TYPES
from simulation.safety import SafetyTier, SafetyGate, ACTION_SAFETY
from simulation.action_schema import OBSERVER_A_ACTIONS


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio needed)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Tool definition tests ────────────────────────────────────────────────────

class TestToolDefinitions:
    def test_all_tools_have_required_structure(self):
        for name, schema in OBSERVER_TOOLS.items():
            assert "description" in schema, f"{name} missing description"
            assert "parameters" in schema, f"{name} missing parameters"
            assert "properties" in schema["parameters"], f"{name} missing properties"

    def test_tool_names(self):
        expected = {
            "query_scores", "query_events", "check_interventions",
            "query_memory", "interview_agent", "query_graph",
        }
        assert set(OBSERVER_TOOLS.keys()) == expected

    def test_interview_agent_required_fields(self):
        required = set(OBSERVER_TOOLS["interview_agent"]["parameters"]["required"])
        assert "agent_id" in required
        assert "question" in required

    def test_query_memory_required_fields(self):
        required = set(OBSERVER_TOOLS["query_memory"]["parameters"]["required"])
        assert "agent_id" in required
        assert "query" in required

    def test_query_events_enum_values(self):
        enum = OBSERVER_TOOLS["query_events"]["parameters"]["properties"]["event_type"]["enum"]
        assert "stimulus" in enum
        assert "response" in enum
        assert "score" in enum
        assert "observation" in enum
        assert "intervention" in enum

    def test_all_tools_have_safety_tiers(self):
        for name in OBSERVER_TOOLS:
            assert name in TOOL_SAFETY, f"Tool '{name}' missing safety tier"

    def test_interview_is_review_tier(self):
        assert TOOL_SAFETY["interview_agent"] == SafetyTier.REVIEW

    def test_read_only_tools_are_auto(self):
        for tool in ["query_scores", "query_events", "check_interventions", "query_memory"]:
            assert TOOL_SAFETY[tool] == SafetyTier.AUTO


# ── Safety integration ───────────────────────────────────────────────────────

class TestToolSafetyIntegration:
    def test_tool_safety_in_action_safety_map(self):
        assert "tool.query_scores" in ACTION_SAFETY
        assert "tool.interview_agent" in ACTION_SAFETY

    def test_tool_safety_tiers_match(self):
        assert ACTION_SAFETY["tool.query_scores"] == SafetyTier.AUTO
        assert ACTION_SAFETY["tool.interview_agent"] == SafetyTier.REVIEW


# ── Event stream integration ────────────────────────────────────────────────

class TestToolUseEventType:
    def test_tool_use_in_all_event_types(self):
        assert TOOL_USE in ALL_EVENT_TYPES

    def test_tool_use_value(self):
        assert TOOL_USE == "tool_use"

    def test_emit_tool_use_event(self):
        stream = EventStream()
        stream.emit(5, "D", TOOL_USE, "observer_a", {
            "tool": "query_scores",
            "args": {"last_n_ticks": 3},
            "result_length": 120,
            "step": 0,
        })
        events = stream.query(event_type=TOOL_USE)
        assert len(events) == 1
        assert events[0].payload["tool"] == "query_scores"
        assert events[0].agent_id == "observer_a"


# ── ReACT response parsing ──────────────────────────────────────────────────

class TestParseReactResponse:
    def test_parse_tool_call(self):
        response = json.dumps({"action": "query_scores", "last_n_ticks": 5})
        parsed = _parse_react_response(response, OBSERVER_TOOLS)
        assert parsed is not None
        assert parsed["action"] == "query_scores"
        assert parsed["last_n_ticks"] == 5

    def test_parse_final_answer(self):
        response = json.dumps({
            "action": "final_answer",
            "reasoning": "Analysis complete.",
            "trajectory_summary": "All stable.",
            "clustering": "stable",
            "concern_level": "none",
        })
        parsed = _parse_react_response(response, OBSERVER_TOOLS)
        assert parsed is not None
        assert parsed["action"] == "final_answer"
        assert parsed["reasoning"] == "Analysis complete."

    def test_parse_implicit_final_answer(self):
        """If response has all OBSERVER_A required fields but no action, infer final_answer."""
        response = json.dumps({
            "reasoning": "Everything is fine.",
            "trajectory_summary": "Stable.",
            "clustering": "stable",
            "concern_level": "none",
        })
        parsed = _parse_react_response(response, OBSERVER_TOOLS)
        assert parsed is not None
        assert parsed["action"] == "final_answer"

    def test_parse_markdown_wrapped(self):
        response = '```json\n{"action": "query_scores"}\n```'
        parsed = _parse_react_response(response, OBSERVER_TOOLS)
        assert parsed is not None
        assert parsed["action"] == "query_scores"

    def test_parse_unknown_tool(self):
        response = json.dumps({"action": "nonexistent_tool"})
        parsed = _parse_react_response(response, OBSERVER_TOOLS)
        assert parsed is None

    def test_parse_invalid_json(self):
        parsed = _parse_react_response("not json at all", OBSERVER_TOOLS)
        assert parsed is None

    def test_parse_no_action_field(self):
        response = json.dumps({"some_key": "some_value"})
        parsed = _parse_react_response(response, OBSERVER_TOOLS)
        assert parsed is None

    def test_parse_embedded_json(self):
        response = 'I will query scores: {"action": "query_scores", "last_n_ticks": 3}'
        parsed = _parse_react_response(response, OBSERVER_TOOLS)
        assert parsed is not None
        assert parsed["action"] == "query_scores"


# ── Tool prompt generation ───────────────────────────────────────────────────

class TestToolsToPrompt:
    def test_prompt_contains_all_tools(self):
        prompt = _tools_to_prompt()
        for name in OBSERVER_TOOLS:
            assert f'"{name}"' in prompt

    def test_prompt_contains_format_instruction(self):
        prompt = _tools_to_prompt()
        assert "TOOL CALL FORMAT" in prompt
        assert "final_answer" in prompt

    def test_prompt_is_string(self):
        assert isinstance(_tools_to_prompt(), str)


# ── Tool execution tests ────────────────────────────────────────────────────

def _make_tool_world_state():
    """Create a minimal WorldState-like object for tool execution tests."""
    stream = EventStream()
    for tick in range(1, 4):
        stream.emit(tick, "C", "score", "participant_0", {
            "score_before": 0.3 + tick * 0.05,
            "score_after": 0.3 + tick * 0.06,
            "signal": 0.5,
            "signal_se": 0.01,
        })
        stream.emit(tick, "C", "score", "participant_1", {
            "score_before": 0.4 + tick * 0.03,
            "score_after": 0.4 + tick * 0.04,
            "signal": 0.6,
            "signal_se": 0.02,
        })
    stream.emit(2, "A", "stimulus", "participant_0", {"content": "test stimulus"})
    stream.emit(2, "B", "response", "participant_0", {"content": "test response"})

    class FakeWorldState:
        def __init__(self):
            self.stream = stream
            self.active_interventions = []
            self.memory = None
    return FakeWorldState()


def _make_participants():
    """Create fake participant objects."""
    class FakeParticipant:
        def __init__(self, agent_id):
            self.agent_id = agent_id
    return [FakeParticipant("participant_0"), FakeParticipant("participant_1")]


class TestToolExecution:
    def test_query_scores_all(self):
        ws = _make_tool_world_state()
        participants = _make_participants()
        result = _run(_exec_query_scores({}, ws, participants, tick=3))
        assert "participant_0" in result
        assert "participant_1" in result
        assert "latest=" in result

    def test_query_scores_specific_agent(self):
        ws = _make_tool_world_state()
        participants = _make_participants()
        result = _run(_exec_query_scores(
            {"agent_ids": ["participant_0"]}, ws, participants, tick=3,
        ))
        assert "participant_0" in result
        assert "participant_1" not in result

    def test_query_scores_last_n(self):
        ws = _make_tool_world_state()
        participants = _make_participants()
        result = _run(_exec_query_scores(
            {"last_n_ticks": 1}, ws, participants, tick=3,
        ))
        assert "participant_0" in result

    def test_query_events(self):
        ws = _make_tool_world_state()
        result = _run(_exec_query_events(
            {"event_type": "score", "last_n": 2}, ws, [], tick=3,
        ))
        assert "score" in result
        assert "participant" in result

    def test_query_events_by_agent(self):
        ws = _make_tool_world_state()
        result = _run(_exec_query_events(
            {"agent_id": "participant_0"}, ws, [], tick=3,
        ))
        assert "participant_0" in result

    def test_query_events_no_results(self):
        ws = _make_tool_world_state()
        result = _run(_exec_query_events(
            {"event_type": "flame_snapshot"}, ws, [], tick=3,
        ))
        assert "No matching events" in result

    def test_check_interventions_empty(self):
        ws = _make_tool_world_state()
        result = _run(_exec_check_interventions({}, ws, [], tick=3))
        assert "Active interventions (0)" in result
        assert "none" in result

    def test_check_interventions_with_active(self):
        ws = _make_tool_world_state()

        class FakeIntervention:
            type = "participant_nudge"
            description = "Take a pause"
            activated_at = 2
            duration = 5
            source = "observer_b"
        ws.active_interventions = [FakeIntervention()]

        result = _run(_exec_check_interventions({}, ws, [], tick=3))
        assert "Active interventions (1)" in result
        assert "Take a pause" in result

    def test_query_memory_disabled(self):
        ws = _make_tool_world_state()
        result = _run(_exec_query_memory(
            {"agent_id": "p0", "query": "test"}, ws, [], tick=3,
        ))
        assert "not enabled" in result


# ── ReactObserver construction ───────────────────────────────────────────────

class TestReactObserverConstruction:
    def test_default_construction(self):
        obs = ReactObserver("observer_a", "observer_a")
        assert obs.agent_id == "observer_a"
        assert obs.max_steps >= 2
        assert len(obs._tools) == len(OBSERVER_TOOLS)

    def test_filtered_tools(self):
        obs = ReactObserver(
            "observer_a", "observer_a",
            enabled_tools=["query_scores", "query_events"],
        )
        assert set(obs._tools.keys()) == {"query_scores", "query_events"}

    def test_set_participants(self):
        obs = ReactObserver("observer_a", "observer_a")
        obs.set_participants(["p0", "p1"])
        assert obs.participants == ["p0", "p1"]

    def test_min_max_steps(self):
        obs = ReactObserver("observer_a", "observer_a", max_steps=1)
        assert obs.max_steps >= 2  # Enforced minimum

    def test_has_analyse_method(self):
        obs = ReactObserver("observer_a", "observer_a")
        assert hasattr(obs, "analyse")
        assert callable(obs.analyse)

    def test_has_history_and_analyses(self):
        obs = ReactObserver("observer_a", "observer_a")
        assert obs.history == []
        assert obs.analyses == []


# ── ReactObserver ReACT loop (mocked LLM) ───────────────────────────────────

def _make_react_world_state():
    """WorldState-like object for ReACT loop tests."""
    stream = EventStream()
    for tick in range(1, 4):
        stream.emit(tick, "C", "score", "participant_0", {
            "score_before": 0.3, "score_after": 0.35,
            "signal": 0.5, "signal_se": 0.01,
        })

    class FakeWorldState:
        def __init__(self):
            self.stream = stream
            self.active_interventions = []
            self.memory = None
            self.k = 3
            self._log = []
            self._log_by_tick = {}
            self._compliance_log = []
            self.safety_gate = SafetyGate()

        def observer_prompt_window(self, tick, n):
            return "No observations yet."

        def compute_score_statistics(self, tick):
            return {}

    return FakeWorldState()


class TestReactLoop:
    def test_single_step_final_answer(self):
        """If LLM immediately returns final_answer, loop terminates."""
        from unittest.mock import AsyncMock, patch

        final = json.dumps({
            "action": "final_answer",
            "reasoning": "Quick scan shows stability.",
            "trajectory_summary": "All flat.",
            "clustering": "stable",
            "concern_level": "none",
        })

        with patch(
            "simulation.core.agents_impl.resilient_agent_turn",
            new_callable=AsyncMock,
            return_value=final,
        ):
            obs = ReactObserver("observer_a", "observer_a", max_steps=5)
            ws = _make_react_world_state()
            result = _run(obs.analyse(1, ws, 2))

        assert "stable" in result or "stability" in result
        assert obs._last_parsed is not None
        assert obs._last_parsed.get("action") == "analyse"

    def test_tool_then_final_answer(self):
        """LLM calls a tool first, then final_answer."""
        from unittest.mock import AsyncMock, patch

        call_count = 0

        async def mock_agent_turn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({"action": "query_scores"})
            else:
                return json.dumps({
                    "action": "final_answer",
                    "reasoning": "After checking scores, all is well.",
                    "trajectory_summary": "Stable scores.",
                    "clustering": "stable",
                    "concern_level": "none",
                })

        with patch(
            "simulation.core.agents_impl.resilient_agent_turn",
            side_effect=mock_agent_turn,
        ):
            obs = ReactObserver("observer_a", "observer_a", max_steps=5)
            ws = _make_react_world_state()
            result = _run(obs.analyse(1, ws, 2))

        assert call_count == 2
        tool_events = ws.stream.query(event_type="tool_use")
        assert len(tool_events) == 1
        assert tool_events[0].payload["tool"] == "query_scores"

    def test_max_steps_enforced(self):
        """After max_steps, loop terminates even without final_answer."""
        from unittest.mock import AsyncMock, patch

        call_count = 0

        async def mock_agent_turn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return json.dumps({"action": "query_scores"})

        with patch(
            "simulation.core.agents_impl.resilient_agent_turn",
            side_effect=mock_agent_turn,
        ):
            obs = ReactObserver("observer_a", "observer_a", max_steps=3)
            ws = _make_react_world_state()
            result = _run(obs.analyse(1, ws, 2))

        assert call_count == 3

    def test_unknown_tool_handled(self):
        """Unknown tool name gets an error message, loop continues."""
        from unittest.mock import AsyncMock, patch

        call_count = 0

        async def mock_agent_turn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({"action": "nonexistent_tool"})
            else:
                return json.dumps({
                    "action": "final_answer",
                    "reasoning": "OK.",
                    "trajectory_summary": "OK.",
                    "clustering": "stable",
                    "concern_level": "none",
                })

        with patch(
            "simulation.core.agents_impl.resilient_agent_turn",
            side_effect=mock_agent_turn,
        ):
            obs = ReactObserver("observer_a", "observer_a", max_steps=5)
            ws = _make_react_world_state()
            result = _run(obs.analyse(1, ws, 2))

        assert call_count == 2

    def test_react_metadata_in_parsed(self):
        """Final parsed output includes _react_steps and _tools_used."""
        from unittest.mock import AsyncMock, patch

        call_count = 0

        async def mock_agent_turn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({"action": "check_interventions"})
            else:
                return json.dumps({
                    "action": "final_answer",
                    "reasoning": "Done.",
                    "trajectory_summary": "Done.",
                    "clustering": "stable",
                    "concern_level": "low",
                })

        with patch(
            "simulation.core.agents_impl.resilient_agent_turn",
            side_effect=mock_agent_turn,
        ):
            obs = ReactObserver("observer_a", "observer_a", max_steps=5)
            ws = _make_react_world_state()
            _run(obs.analyse(1, ws, 2))

        assert obs._last_parsed is not None
        assert obs._last_parsed["_react_steps"] == 1
        assert "check_interventions" in obs._last_parsed["_tools_used"]

    def test_analyse_stores_in_history(self):
        """Each analyse call appends to history and analyses lists."""
        from unittest.mock import AsyncMock, patch

        final = json.dumps({
            "action": "final_answer",
            "reasoning": "Done.",
            "trajectory_summary": "Done.",
            "clustering": "stable",
            "concern_level": "none",
        })

        with patch(
            "simulation.core.agents_impl.resilient_agent_turn",
            new_callable=AsyncMock,
            return_value=final,
        ):
            obs = ReactObserver("observer_a", "observer_a", max_steps=5)
            ws = _make_react_world_state()
            _run(obs.analyse(1, ws, 2))

        assert len(obs.history) == 1
        assert len(obs.analyses) == 1


# ── ScenarioConfig integration ───────────────────────────────────────────────

class TestReactConfig:
    def test_react_config_defaults(self):
        from simulation.scenario import ReactConfig
        cfg = ReactConfig()
        assert cfg.enabled is False
        assert cfg.max_steps == 5
        assert "query_scores" in cfg.tools

    def test_react_in_scenario_config(self):
        from simulation.scenario import ScenarioConfig
        config = ScenarioConfig.from_dict({
            "meta": {"name": "test", "version": "1.0"},
            "agents": {"roles": [
                {"id": "p", "slot": "swarm", "type": "participant",
                 "system_prompt": "test", "count": 2},
            ]},
            "react": {"enabled": True, "max_steps": 3},
        })
        assert config.react.enabled is True
        assert config.react.max_steps == 3

    def test_react_default_in_scenario(self):
        from simulation.scenario import ScenarioConfig
        config = ScenarioConfig.from_dict({
            "meta": {"name": "test", "version": "1.0"},
            "agents": {"roles": [
                {"id": "p", "slot": "swarm", "type": "participant",
                 "system_prompt": "test", "count": 2},
            ]},
        })
        assert config.react.enabled is False

    def test_react_custom_tools(self):
        from simulation.scenario import ScenarioConfig
        config = ScenarioConfig.from_dict({
            "meta": {"name": "test", "version": "1.0"},
            "agents": {"roles": [
                {"id": "p", "slot": "swarm", "type": "participant",
                 "system_prompt": "test", "count": 2},
            ]},
            "react": {"enabled": True, "tools": ["query_scores", "query_events"]},
        })
        assert len(config.react.tools) == 2
        assert "query_scores" in config.react.tools
