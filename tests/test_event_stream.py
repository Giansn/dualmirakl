"""
Tests for the EventStream module.

Run: python -m pytest tests/test_event_stream.py -v
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.event_stream import (
    EventStream, SimEvent,
    STIMULUS, RESPONSE, SCORE, OBSERVATION, INTERVENTION,
    COMPLIANCE, FLAME_SNAPSHOT, PERSONA, CONTEXT,
    ALL_EVENT_TYPES,
)


# ── SimEvent tests ────────────────────────────────────────────────────────────

class TestSimEvent:
    def test_create(self):
        e = SimEvent(tick=1, phase="A", event_type=STIMULUS,
                     agent_id="participant_0", payload={"content": "hello"})
        assert e.tick == 1
        assert e.phase == "A"
        assert e.event_type == STIMULUS
        assert e.agent_id == "participant_0"
        assert e.payload == {"content": "hello"}
        assert e.timestamp > 0

    def test_to_dict(self):
        e = SimEvent(tick=2, phase="B", event_type=RESPONSE,
                     agent_id="participant_1", payload={"content": "world"})
        d = e.to_dict()
        assert d["tick"] == 2
        assert d["phase"] == "B"
        assert d["event_type"] == RESPONSE
        assert d["agent_id"] == "participant_1"
        assert d["payload"] == {"content": "world"}
        assert "timestamp" not in d  # not exported

    def test_event_type_constants(self):
        assert len(ALL_EVENT_TYPES) == 9
        assert STIMULUS in ALL_EVENT_TYPES
        assert FLAME_SNAPSHOT in ALL_EVENT_TYPES


# ── EventStream emit tests ───────────────────────────────────────────────────

class TestEmit:
    def test_emit_returns_event(self):
        stream = EventStream()
        event = stream.emit(1, "A", STIMULUS, "environment", {"content": "go"})
        assert isinstance(event, SimEvent)
        assert event.tick == 1

    def test_emit_increments_length(self):
        stream = EventStream()
        assert len(stream) == 0
        stream.emit(1, "A", STIMULUS, "environment", {})
        assert len(stream) == 1
        stream.emit(1, "B", RESPONSE, "participant_0", {})
        assert len(stream) == 2

    def test_emit_indexes_by_tick(self):
        stream = EventStream()
        stream.emit(1, "A", STIMULUS, "env", {})
        stream.emit(2, "A", STIMULUS, "env", {})
        stream.emit(2, "B", RESPONSE, "p_0", {})
        assert len(stream.tick_events(1)) == 1
        assert len(stream.tick_events(2)) == 2
        assert len(stream.tick_events(3)) == 0

    def test_emit_indexes_by_agent(self):
        stream = EventStream()
        stream.emit(1, "A", STIMULUS, "p_0", {})
        stream.emit(1, "A", STIMULUS, "p_1", {})
        stream.emit(2, "B", RESPONSE, "p_0", {})
        assert len(stream.query(agent_id="p_0")) == 2
        assert len(stream.query(agent_id="p_1")) == 1

    def test_emit_indexes_by_type(self):
        stream = EventStream()
        stream.emit(1, "A", STIMULUS, "env", {})
        stream.emit(1, "B", RESPONSE, "p_0", {})
        stream.emit(1, "C", SCORE, "p_0", {})
        assert len(stream.query(event_type=STIMULUS)) == 1
        assert len(stream.query(event_type=SCORE)) == 1


# ── EventStream query tests ──────────────────────────────────────────────────

class TestQuery:
    @pytest.fixture
    def populated_stream(self):
        stream = EventStream()
        # Tick 1
        stream.emit(1, "A", STIMULUS, "environment", {"content": "s1"})
        stream.emit(1, "B", RESPONSE, "participant_0", {"content": "r1"})
        stream.emit(1, "B", RESPONSE, "participant_1", {"content": "r2"})
        stream.emit(1, "C", SCORE, "participant_0", {"score_after": 0.4})
        stream.emit(1, "C", SCORE, "participant_1", {"score_after": 0.5})
        # Tick 2
        stream.emit(2, "A", STIMULUS, "environment", {"content": "s2"})
        stream.emit(2, "B", RESPONSE, "participant_0", {"content": "r3"})
        stream.emit(2, "B", RESPONSE, "participant_1", {"content": "r4"})
        stream.emit(2, "C", SCORE, "participant_0", {"score_after": 0.45})
        stream.emit(2, "C", SCORE, "participant_1", {"score_after": 0.55})
        # Tick 2 observer
        stream.emit(2, "D", OBSERVATION, "observer_a", {"content": "analysis"})
        stream.emit(2, "D", INTERVENTION, "observer_b", {
            "type": "pause_prompt", "activated_at": 2, "duration": -1,
        })
        return stream

    def test_query_by_agent(self, populated_stream):
        events = populated_stream.query(agent_id="participant_0")
        assert len(events) == 4  # 2 responses + 2 scores
        assert all(e.agent_id == "participant_0" for e in events)

    def test_query_by_type(self, populated_stream):
        events = populated_stream.query(event_type=SCORE)
        assert len(events) == 4
        assert all(e.event_type == SCORE for e in events)

    def test_query_by_agent_and_type(self, populated_stream):
        events = populated_stream.query(agent_id="participant_0", event_type=SCORE)
        assert len(events) == 2
        assert all(e.agent_id == "participant_0" and e.event_type == SCORE for e in events)

    def test_query_since_tick(self, populated_stream):
        events = populated_stream.query(since_tick=2)
        assert all(e.tick >= 2 for e in events)
        assert len(events) == 7  # all tick-2 events

    def test_query_until_tick(self, populated_stream):
        events = populated_stream.query(until_tick=1)
        assert all(e.tick <= 1 for e in events)
        assert len(events) == 5  # all tick-1 events

    def test_query_tick_range(self, populated_stream):
        events = populated_stream.query(since_tick=1, until_tick=1)
        assert len(events) == 5
        assert all(e.tick == 1 for e in events)

    def test_query_last_n(self, populated_stream):
        events = populated_stream.query(event_type=SCORE, last_n=2)
        assert len(events) == 2
        # Should be the last two score events (tick 2)
        assert events[0].tick == 2
        assert events[1].tick == 2

    def test_query_combined_filters(self, populated_stream):
        events = populated_stream.query(
            agent_id="participant_0", event_type=RESPONSE, since_tick=2
        )
        assert len(events) == 1
        assert events[0].payload["content"] == "r3"

    def test_query_empty(self, populated_stream):
        events = populated_stream.query(agent_id="nonexistent")
        assert events == []

    def test_query_no_filters(self, populated_stream):
        events = populated_stream.query()
        assert len(events) == 12  # all events


# ── Window & latest ──────────────────────────────────────────────────────────

class TestWindowAndLatest:
    def test_window(self):
        stream = EventStream()
        for t in range(1, 6):
            stream.emit(t, "C", SCORE, "p_0", {"score_after": t * 0.1})
        window = stream.window("p_0", tick=5, window_size=3)
        assert len(window) == 3
        assert window[0].tick == 3
        assert window[-1].tick == 5

    def test_window_at_start(self):
        stream = EventStream()
        stream.emit(1, "C", SCORE, "p_0", {"score_after": 0.1})
        stream.emit(2, "C", SCORE, "p_0", {"score_after": 0.2})
        window = stream.window("p_0", tick=2, window_size=5)
        assert len(window) == 2  # only 2 ticks exist

    def test_latest(self):
        stream = EventStream()
        stream.emit(1, "D", OBSERVATION, "observer_a", {"content": "first"})
        stream.emit(3, "D", OBSERVATION, "observer_a", {"content": "second"})
        latest = stream.latest(OBSERVATION, agent_id="observer_a")
        assert latest.payload["content"] == "second"

    def test_latest_none(self):
        stream = EventStream()
        assert stream.latest(OBSERVATION) is None


# ── Aggregate helpers ─────────────────────────────────────────────────────────

class TestAggregates:
    def test_score_trajectory(self):
        stream = EventStream()
        for t in range(1, 5):
            stream.emit(t, "C", SCORE, "p_0", {"score_after": round(t * 0.1, 4)})
        trajectory = stream.score_trajectory("p_0")
        assert trajectory == [0.1, 0.2, 0.3, 0.4]

    def test_score_trajectory_empty(self):
        stream = EventStream()
        assert stream.score_trajectory("p_0") == []

    def test_active_interventions_permanent(self):
        stream = EventStream()
        stream.emit(2, "D", INTERVENTION, "observer_b", {
            "type": "pause_prompt", "activated_at": 2, "duration": -1,
        })
        active = stream.active_interventions(tick=10)
        assert len(active) == 1

    def test_active_interventions_expired(self):
        stream = EventStream()
        stream.emit(2, "D", INTERVENTION, "observer_b", {
            "type": "pacing_adjustment", "activated_at": 2, "duration": 3,
        })
        assert len(stream.active_interventions(tick=4)) == 1  # 2+3=5 > 4
        assert len(stream.active_interventions(tick=5)) == 0  # 2+3=5 == 5, expired

    def test_response_texts(self):
        stream = EventStream()
        stream.emit(1, "B", RESPONSE, "p_0", {"content": "hello"})
        stream.emit(1, "B", RESPONSE, "p_1", {"content": "world"})
        stream.emit(1, "A", STIMULUS, "env", {"content": "go"})  # not a response
        texts = stream.response_texts(1)
        assert texts == {"p_0": "hello", "p_1": "world"}

    def test_response_texts_empty_tick(self):
        stream = EventStream()
        assert stream.response_texts(99) == {}


# ── Batch emit ────────────────────────────────────────────────────────────────

class TestBatchEmit:
    def test_batch_emits_all(self):
        stream = EventStream()
        with stream.batch() as b:
            b.emit(1, "A", STIMULUS, "p_0", {"content": "a"})
            b.emit(1, "A", STIMULUS, "p_1", {"content": "b"})
            b.emit(1, "A", STIMULUS, "p_2", {"content": "c"})
        assert len(stream) == 3

    def test_batch_indexes_correctly(self):
        stream = EventStream()
        with stream.batch() as b:
            b.emit(1, "A", STIMULUS, "p_0", {"content": "a"})
            b.emit(1, "B", RESPONSE, "p_0", {"content": "b"})
        assert len(stream.query(event_type=STIMULUS)) == 1
        assert len(stream.query(event_type=RESPONSE)) == 1
        assert len(stream.query(agent_id="p_0")) == 2
        assert len(stream.tick_events(1)) == 2

    def test_batch_not_visible_during(self):
        stream = EventStream()
        stream.emit(0, "A", STIMULUS, "env", {"content": "pre"})
        with stream.batch() as b:
            b.emit(1, "A", STIMULUS, "p_0", {"content": "a"})
            # During batch, events are pending — only pre-existing visible
            assert len(stream) == 1
        # After batch, all visible
        assert len(stream) == 2

    def test_batch_empty(self):
        stream = EventStream()
        with stream.batch() as b:
            pass  # no emissions
        assert len(stream) == 0

    def test_batch_returns_events(self):
        stream = EventStream()
        with stream.batch() as b:
            e = b.emit(1, "A", STIMULUS, "p_0", {"content": "test"})
        assert e.agent_id == "p_0"


# ── Introspection ─────────────────────────────────────────────────────────────

class TestIntrospection:
    def test_ticks(self):
        stream = EventStream()
        stream.emit(3, "A", STIMULUS, "env", {})
        stream.emit(1, "A", STIMULUS, "env", {})
        stream.emit(2, "A", STIMULUS, "env", {})
        assert stream.ticks == [1, 2, 3]

    def test_agents(self):
        stream = EventStream()
        stream.emit(1, "A", STIMULUS, "env", {})
        stream.emit(1, "B", RESPONSE, "p_0", {})
        assert set(stream.agents) == {"env", "p_0"}


# ── Export ────────────────────────────────────────────────────────────────────

class TestExport:
    def test_export_list(self):
        stream = EventStream()
        stream.emit(1, "A", STIMULUS, "env", {"content": "go"})
        stream.emit(1, "B", RESPONSE, "p_0", {"content": "ok"})
        exported = stream.export()
        assert len(exported) == 2
        assert exported[0]["event_type"] == STIMULUS
        assert exported[1]["payload"]["content"] == "ok"

    def test_export_by_tick(self):
        stream = EventStream()
        stream.emit(1, "A", STIMULUS, "env", {})
        stream.emit(2, "A", STIMULUS, "env", {})
        by_tick = stream.export_by_tick()
        assert set(by_tick.keys()) == {1, 2}
        assert len(by_tick[1]) == 1
        assert len(by_tick[2]) == 1

    def test_export_empty(self):
        stream = EventStream()
        assert stream.export() == []
        assert stream.export_by_tick() == {}


# ── Integration with WorldState ───────────────────────────────────────────────

class TestWorldStateIntegration:
    def test_worldstate_has_stream(self):
        from simulation.sim_loop import WorldState
        ws = WorldState(k=3)
        assert isinstance(ws.stream, EventStream)
        assert len(ws.stream) == 0

    def test_worldstate_stream_persists(self):
        from simulation.sim_loop import WorldState
        ws = WorldState(k=3)
        ws.stream.emit(1, "A", STIMULUS, "env", {"content": "test"})
        assert len(ws.stream) == 1
        # Stream survives across method calls
        events = ws.stream.query(event_type=STIMULUS)
        assert len(events) == 1

    def test_stream_and_log_coexist(self):
        from simulation.sim_loop import WorldState, ObsEntry
        ws = WorldState(k=3)
        entry = ObsEntry(tick=1, participant_id="p_0", score_before=0.3,
                         score_after=0.35, stimulus="hi", response="hello",
                         signal=0.5, signal_se=0.02)
        ws.log(entry)
        ws.stream.emit(1, "C", SCORE, "p_0", {
            "score_before": 0.3, "score_after": 0.35,
            "signal": 0.5, "signal_se": 0.02,
        })
        # Both systems have the data
        assert len(ws.full_log()) == 1
        assert len(ws.stream.query(event_type=SCORE)) == 1
