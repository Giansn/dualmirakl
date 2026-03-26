"""
Tests for the graph memory feedback loop module.

Run: python -m pytest tests/test_graph_memory.py -v
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.graph_memory import (
    GraphMemory,
    TemporalEdge,
    ALL_EDGE_TYPES,
    SCORED_AT, RESPONDED_TO, INTERVENTION_ON, ESCALATED,
    DISENGAGED, FLAGGED_BY, CLUSTERED_WITH, RECEIVED_STIMULUS,
)
from simulation.event_stream import EventStream, GRAPH_UPDATE, ALL_EVENT_TYPES


# ── TemporalEdge tests ──────────────────────────────────────────────────────

class TestTemporalEdge:
    def test_creation(self):
        edge = TemporalEdge(edge_type=SCORED_AT, valid_at=5, invalid_at=-1)
        assert edge.is_active
        assert edge.valid_at == 5

    def test_expire(self):
        edge = TemporalEdge(edge_type=SCORED_AT, valid_at=5, invalid_at=-1)
        edge.expire(10)
        assert not edge.is_active
        assert edge.invalid_at == 10

    def test_expire_idempotent(self):
        edge = TemporalEdge(edge_type=SCORED_AT, valid_at=5, invalid_at=10)
        edge.expire(15)  # already expired, should not change
        assert edge.invalid_at == 10

    def test_to_dict(self):
        edge = TemporalEdge(
            edge_type=ESCALATED, valid_at=3, invalid_at=-1,
            properties={"trigger": "high score"},
        )
        d = edge.to_dict()
        assert d["edge_type"] == ESCALATED
        assert d["valid_at"] == 3
        assert d["properties"]["trigger"] == "high score"

    def test_properties_default_empty(self):
        edge = TemporalEdge(edge_type=SCORED_AT, valid_at=1, invalid_at=-1)
        assert edge.properties == {}


# ── Edge type constants ──────────────────────────────────────────────────────

class TestEdgeTypes:
    def test_all_edge_types_count(self):
        assert len(ALL_EDGE_TYPES) == 8

    def test_key_edge_types_present(self):
        assert SCORED_AT in ALL_EDGE_TYPES
        assert ESCALATED in ALL_EDGE_TYPES
        assert INTERVENTION_ON in ALL_EDGE_TYPES
        assert FLAGGED_BY in ALL_EDGE_TYPES


# ── Event stream GRAPH_UPDATE ────────────────────────────────────────────────

class TestGraphUpdateEvent:
    def test_graph_update_in_all_types(self):
        assert GRAPH_UPDATE in ALL_EVENT_TYPES

    def test_graph_update_value(self):
        assert GRAPH_UPDATE == "graph_update"


# ── GraphMemory node operations ─────────────────────────────────────────────

class TestGraphNodes:
    def test_add_node(self):
        g = GraphMemory()
        g.add_node("p_0", "agent", label="Participant 0", tick=1)
        assert g.n_nodes == 1
        node = g.get_node("p_0")
        assert node["type"] == "agent"
        assert node["label"] == "Participant 0"

    def test_update_node(self):
        g = GraphMemory()
        g.add_node("p_0", "agent", tick=1, score=0.3)
        g.add_node("p_0", "agent", tick=5, score=0.7)
        assert g.n_nodes == 1  # no duplicate
        node = g.get_node("p_0")
        assert node["properties"]["score"] == 0.7
        assert node["tick_updated"] == 5

    def test_get_nonexistent_node(self):
        g = GraphMemory()
        assert g.get_node("nonexistent") is None

    def test_nodes_by_type(self):
        g = GraphMemory()
        g.add_node("p_0", "agent", tick=1)
        g.add_node("p_1", "agent", tick=1)
        g.add_node("high_engagement", "score_region", tick=1)
        assert len(g.nodes_by_type("agent")) == 2
        assert len(g.nodes_by_type("score_region")) == 1
        assert len(g.nodes_by_type("unknown")) == 0

    def test_default_label(self):
        g = GraphMemory()
        g.add_node("p_0", "agent", tick=1)
        assert g.get_node("p_0")["label"] == "p_0"  # defaults to node_id


# ── GraphMemory edge operations ─────────────────────────────────────────────

class TestGraphEdges:
    def test_add_edge(self):
        g = GraphMemory()
        g.add_node("p_0", "agent", tick=1)
        g.add_node("high", "region", tick=1)
        edge = g.add_edge("p_0", "high", SCORED_AT, tick=1, score=0.85)
        assert edge.is_active
        assert g.n_edges == 1

    def test_get_edges(self):
        g = GraphMemory()
        g.add_node("p_0", "agent", tick=1)
        g.add_node("high", "region", tick=1)
        g.add_edge("p_0", "high", SCORED_AT, tick=1)
        g.add_edge("p_0", "high", RESPONDED_TO, tick=2)

        all_edges = g.get_edges("p_0")
        assert len(all_edges) == 2

        score_edges = g.get_edges("p_0", edge_type=SCORED_AT)
        assert len(score_edges) == 1

    def test_get_edges_active_only(self):
        g = GraphMemory()
        g.add_node("p_0", "agent", tick=1)
        g.add_node("high", "region", tick=1)
        g.add_edge("p_0", "high", SCORED_AT, tick=1)
        g.expire_edges("p_0", "high", SCORED_AT, tick=5)
        g.add_edge("p_0", "high", SCORED_AT, tick=5)

        active = g.get_edges("p_0", active_only=True)
        assert len(active) == 1
        assert active[0][2].valid_at == 5

    def test_expire_edges(self):
        g = GraphMemory()
        g.add_node("p_0", "agent", tick=1)
        g.add_node("high", "region", tick=1)
        g.add_edge("p_0", "high", SCORED_AT, tick=1)

        count = g.expire_edges("p_0", "high", SCORED_AT, tick=5)
        assert count == 1

        active = g.get_edges("p_0", active_only=True)
        assert len(active) == 0


# ── GraphMemory query operations ────────────────────────────────────────────

class TestGraphQueries:
    def _build_graph(self):
        g = GraphMemory()
        g.add_node("p_0", "agent", tick=1)
        g.add_node("p_1", "agent", tick=1)
        g.add_node("obs_a", "agent", tick=1)
        g.add_node("high", "score_region", tick=1)
        g.add_edge("p_0", "high", SCORED_AT, tick=1)
        g.add_edge("p_1", "high", SCORED_AT, tick=1)
        g.add_edge("p_0", "obs_a", FLAGGED_BY, tick=2)
        return g

    def test_neighbors(self):
        g = self._build_graph()
        n = g.neighbors("p_0")
        assert "high" in n
        assert "obs_a" in n

    def test_neighbors_filtered(self):
        g = self._build_graph()
        n = g.neighbors("p_0", edge_type=SCORED_AT)
        assert n == ["high"]

    def test_relationship_path(self):
        g = self._build_graph()
        path = g.relationship_path("p_0", "p_1")
        assert path is not None
        assert path[0] == "p_0"
        assert path[-1] == "p_1"
        assert len(path) <= 4

    def test_relationship_path_self(self):
        g = self._build_graph()
        path = g.relationship_path("p_0", "p_0")
        assert path == ["p_0"]

    def test_relationship_path_not_found(self):
        g = GraphMemory()
        g.add_node("a", "agent", tick=1)
        g.add_node("b", "agent", tick=1)
        # No edges connecting a and b
        path = g.relationship_path("a", "b")
        assert path is None

    def test_relationship_path_nonexistent_node(self):
        g = GraphMemory()
        path = g.relationship_path("a", "b")
        assert path is None

    def test_summarize_agent(self):
        g = self._build_graph()
        summary = g.summarize_agent("p_0")
        assert "p_0" in summary
        assert "agent" in summary

    def test_summarize_nonexistent(self):
        g = GraphMemory()
        summary = g.summarize_agent("unknown")
        assert "not found" in summary


# ── Event stream distillation ────────────────────────────────────────────────

class TestDistillation:
    def _make_stream_with_scores(self):
        stream = EventStream()
        stream.emit(1, "C", "score", "participant_0", {
            "score_before": 0.3, "score_after": 0.45,
            "signal": 0.5, "signal_se": 0.01,
        })
        stream.emit(1, "C", "score", "participant_1", {
            "score_before": 0.6, "score_after": 0.85,
            "signal": 0.8, "signal_se": 0.02,
        })
        return stream

    def test_distill_score_events(self):
        g = GraphMemory()
        stream = self._make_stream_with_scores()
        ops = g.distill_tick(1, stream)
        assert ops > 0
        assert g.get_node("participant_0") is not None
        assert g.get_node("participant_1") is not None
        # p_1 should be in high_engagement (score 0.85)
        n = g.neighbors("participant_1", edge_type=SCORED_AT)
        assert "high_engagement" in n

    def test_distill_score_region_transitions(self):
        """Score region edges expire when agent moves to new region."""
        g = GraphMemory()
        stream = EventStream()
        # Tick 1: p_0 in moderate (score 0.55)
        stream.emit(1, "C", "score", "participant_0", {
            "score_before": 0.3, "score_after": 0.55,
            "signal": 0.5, "signal_se": 0.01,
        })
        g.distill_tick(1, stream)
        n1 = g.neighbors("participant_0", edge_type=SCORED_AT)
        assert "moderate_engagement" in n1

        # Tick 2: p_0 moves to high (score 0.85)
        stream.emit(2, "C", "score", "participant_0", {
            "score_before": 0.55, "score_after": 0.85,
            "signal": 0.9, "signal_se": 0.01,
        })
        g.distill_tick(2, stream)
        n2 = g.neighbors("participant_0", edge_type=SCORED_AT)
        assert "high_engagement" in n2
        assert "moderate_engagement" not in n2  # expired

    def test_distill_response_escalation(self):
        g = GraphMemory()
        stream = EventStream()
        stream.emit(1, "B", "response", "participant_0", {
            "content": "I am escalating",
            "structured": {"action": "escalate", "trigger": "boredom"},
        })
        ops = g.distill_tick(1, stream)
        assert ops > 0
        assert g.get_node("escalation_pattern") is not None
        n = g.neighbors("participant_0", edge_type=ESCALATED)
        assert "escalation_pattern" in n

    def test_distill_response_disengage(self):
        g = GraphMemory()
        stream = EventStream()
        stream.emit(1, "B", "response", "participant_0", {
            "content": "I am leaving",
            "structured": {
                "action": "disengage",
                "reason": "tired",
                "duration": "extended",
            },
        })
        ops = g.distill_tick(1, stream)
        assert ops > 0
        assert g.get_node("disengagement_pattern") is not None

    def test_distill_observation_flagged(self):
        g = GraphMemory()
        g.add_node("observer_a", "agent", tick=1)
        stream = EventStream()
        stream.emit(1, "D", "observation", "observer_a", {
            "content": "Analysis",
            "structured": {
                "action": "analyse",
                "reasoning": "...",
                "trajectory_summary": "...",
                "clustering": "diverging",
                "concern_level": "high",
                "flagged_participants": ["participant_0", "participant_2"],
            },
        })
        ops = g.distill_tick(1, stream)
        assert ops > 0
        # Flagged participants should have edges
        edges = g.get_edges("participant_0", edge_type=FLAGGED_BY)
        assert len(edges) >= 1
        # Population state node
        state = g.get_node("population_state")
        assert state is not None
        assert state["properties"]["clustering"] == "diverging"

    def test_distill_intervention(self):
        g = GraphMemory()
        g.add_node("participant_0", "agent", tick=1)
        stream = EventStream()
        stream.emit(3, "D", "intervention", "observer_b", {
            "type": "pause_prompt",
            "description": "Take a moment to reflect.",
            "modifier": {},
            "activated_at": 3,
            "duration": 5,
            "source": "observer_b",
        })
        ops = g.distill_tick(3, stream)
        assert ops > 0
        iv_nodes = g.nodes_by_type("intervention")
        assert len(iv_nodes) == 1
        assert "pause_prompt" in iv_nodes[0]

    def test_distill_stimulus(self):
        g = GraphMemory()
        stream = EventStream()
        stream.emit(1, "A", "stimulus", "participant_0", {
            "content": "test stimulus",
        })
        ops = g.distill_tick(1, stream)
        assert ops > 0
        assert g.get_node("environment") is not None
        n = g.neighbors("participant_0", edge_type=RECEIVED_STIMULUS)
        assert "environment" in n

    def test_distill_response_no_structured(self):
        """Response without structured data should not create edges."""
        g = GraphMemory()
        stream = EventStream()
        stream.emit(1, "B", "response", "participant_0", {
            "content": "just a plain response",
        })
        ops = g.distill_tick(1, stream)
        assert ops == 0

    def test_distill_empty_tick(self):
        g = GraphMemory()
        stream = EventStream()
        ops = g.distill_tick(5, stream)
        assert ops == 0

    def test_last_tick_tracked(self):
        g = GraphMemory()
        stream = EventStream()
        stream.emit(7, "A", "stimulus", "p_0", {"content": "x"})
        g.distill_tick(7, stream)
        assert g.last_tick == 7


# ── GraphMemory introspection ────────────────────────────────────────────────

class TestGraphIntrospection:
    def test_stats(self):
        g = GraphMemory()
        g.add_node("a", "agent", tick=1)
        g.add_node("b", "agent", tick=1)
        g.add_edge("a", "b", SCORED_AT, tick=1)
        stats = g.stats()
        assert stats["n_nodes"] == 2
        assert stats["n_edges"] == 1
        assert stats["n_active_edges"] == 1
        assert stats["node_types"]["agent"] == 2

    def test_export(self):
        g = GraphMemory()
        g.add_node("a", "agent", tick=1, foo="bar")
        g.add_node("b", "pattern", tick=2)
        g.add_edge("a", "b", ESCALATED, tick=2, trigger="x")
        data = g.export()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["edges"][0]["edge_type"] == ESCALATED

    def test_query_text_overview(self):
        g = GraphMemory()
        g.add_node("participant_0", "agent", tick=1, latest_score=0.45)
        g.add_node("participant_1", "agent", tick=1, latest_score=0.82)
        text = g.query_text()
        assert "Graph:" in text
        assert "participant_0" in text

    def test_query_text_specific_node(self):
        g = GraphMemory()
        g.add_node("participant_0", "agent", label="P0", tick=1)
        text = g.query_text(node_id="participant_0")
        assert "P0" in text

    def test_query_text_nonexistent(self):
        g = GraphMemory()
        text = g.query_text(node_id="unknown")
        assert "not found" in text

    def test_query_text_no_graph(self):
        g = GraphMemory()
        text = g.query_text()
        assert "0 nodes" in text
