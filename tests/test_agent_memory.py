"""
Tests for the agent memory store.

Run: python -m pytest tests/test_agent_memory.py -v
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.agent_memory import AgentMemoryStore, Memory, _cosine_sim


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_embed(text: str) -> np.ndarray:
    """Deterministic mock embedding: hash-based 32-dim vector."""
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(32).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def _identity_embed(text: str) -> np.ndarray:
    """Embed that returns same vector for same text (for dedup testing)."""
    if "same" in text:
        return np.ones(32, dtype=np.float32) / np.sqrt(32)
    return _mock_embed(text)


# ── Memory dataclass tests ────────────────────────────────────────────────────

class TestMemory:
    def test_to_dict(self):
        mem = Memory(
            id="abc123", agent_id="p_0", title="First impression",
            content="The scenario felt familiar.", tags=["early", "impression"],
            tick_created=1, tick_accessed=3,
            embedding=np.zeros(32),
        )
        d = mem.to_dict()
        assert d["id"] == "abc123"
        assert d["agent_id"] == "p_0"
        assert d["title"] == "First impression"
        assert d["tags"] == ["early", "impression"]
        assert d["tick_created"] == 1
        assert d["tick_accessed"] == 3
        assert "embedding" not in d  # not serialized

    def test_cosine_sim_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_sim(v, v) - 1.0) < 1e-6

    def test_cosine_sim_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_sim(a, b)) < 1e-6

    def test_cosine_sim_zero_vector(self):
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(a, b) == 0.0


# ── Create tests ──────────────────────────────────────────────────────────────

class TestCreate:
    def test_basic_create(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        mem = store.create("p_0", "Test", "Some content", ["tag1"], tick=1)
        assert isinstance(mem, Memory)
        assert mem.agent_id == "p_0"
        assert mem.title == "Test"
        assert mem.tick_created == 1

    def test_create_increments_count(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        assert len(store) == 0
        store.create("p_0", "A", "Content A", [], tick=1)
        assert len(store) == 1
        store.create("p_0", "B", "Content B", [], tick=2)
        assert len(store) == 2

    def test_create_different_agents(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        store.create("p_0", "A", "Content A", [], tick=1)
        store.create("p_1", "B", "Content B", [], tick=1)
        assert len(store) == 2
        assert set(store.agents) == {"p_0", "p_1"}

    def test_dedup_updates_existing(self):
        store = AgentMemoryStore(embed_fn=_identity_embed, dedup_threshold=0.9)
        mem1 = store.create("p_0", "Original", "same content", ["tag1"], tick=1)
        mem2 = store.create("p_0", "Updated", "same thing", ["tag2"], tick=3)
        # Should update, not create new
        assert len(store) == 1
        assert mem2.title == "Updated"
        assert mem2.tick_accessed == 3
        assert "tag1" in mem2.tags  # tags merged

    def test_eviction_when_over_limit(self):
        store = AgentMemoryStore(embed_fn=_mock_embed, max_per_agent=3)
        store.create("p_0", "A", "Content A", [], tick=1)
        store.create("p_0", "B", "Content B", [], tick=2)
        store.create("p_0", "C", "Content C", [], tick=3)
        assert len(store) == 3
        store.create("p_0", "D", "Content D", [], tick=4)
        assert len(store) == 3  # oldest evicted

    def test_eviction_removes_lru(self):
        store = AgentMemoryStore(embed_fn=_mock_embed, max_per_agent=2)
        mem_old = store.create("p_0", "Old", "Old content here", [], tick=1)
        mem_new = store.create("p_0", "New", "New content here", [], tick=2)
        # Manually set tick_accessed to make "Old" more recently accessed
        mem_old.tick_accessed = 5
        mem_new.tick_accessed = 2  # "New" is now LRU
        # Now add another — "New" should be evicted (LRU = lowest tick_accessed)
        store.create("p_0", "Newest", "Newest content here", [], tick=6)
        titles = [m.title for m in store.get_all("p_0")]
        assert "Old" in titles
        assert "Newest" in titles
        assert len(titles) == 2


# ── Retrieve tests ────────────────────────────────────────────────────────────

class TestRetrieve:
    def test_semantic_retrieve(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        store.create("p_0", "Pressure", "I felt pressured by the pace", ["emotion"], tick=1)
        store.create("p_0", "Calm", "Everything seemed peaceful today", ["emotion"], tick=2)
        store.create("p_0", "Technical", "The interface had three buttons", ["detail"], tick=3)
        results = store.retrieve("p_0", "feeling stressed and pressured", tick=4, top_k=2)
        assert len(results) == 2
        # tick_accessed updated
        assert all(r.tick_accessed == 4 for r in results)

    def test_retrieve_empty_agent(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        results = store.retrieve("p_nonexistent", "anything", tick=1)
        assert results == []

    def test_retrieve_top_k(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        for i in range(10):
            store.create("p_0", f"M{i}", f"Content number {i}", [], tick=i)
        results = store.retrieve("p_0", "content", tick=11, top_k=3)
        assert len(results) == 3

    def test_retrieve_by_tags(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        store.create("p_0", "A", "Content A", ["emotion", "early"], tick=1)
        store.create("p_0", "B", "Content B", ["decision"], tick=2)
        store.create("p_0", "C", "Content C", ["emotion", "late"], tick=3)
        results = store.retrieve_by_tags("p_0", ["emotion"])
        assert len(results) == 2
        titles = {m.title for m in results}
        assert titles == {"A", "C"}

    def test_retrieve_by_tags_no_match(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        store.create("p_0", "A", "Content A", ["tag1"], tick=1)
        results = store.retrieve_by_tags("p_0", ["nonexistent"])
        assert results == []

    def test_get_all(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        store.create("p_0", "A", "Content A", [], tick=1)
        store.create("p_0", "B", "Content B", [], tick=2)
        store.create("p_1", "C", "Content C", [], tick=3)
        assert len(store.get_all("p_0")) == 2
        assert len(store.get_all("p_1")) == 1
        assert len(store.get_all("p_2")) == 0


# ── Delete tests ──────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_by_id(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        mem = store.create("p_0", "A", "Content A", [], tick=1)
        assert len(store) == 1
        assert store.delete("p_0", mem.id) is True
        assert len(store) == 0

    def test_delete_nonexistent(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        assert store.delete("p_0", "fake-id") is False

    def test_clear_agent(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        store.create("p_0", "A", "Content A", [], tick=1)
        store.create("p_1", "B", "Content B", [], tick=2)
        store.clear("p_0")
        assert len(store.get_all("p_0")) == 0
        assert len(store.get_all("p_1")) == 1

    def test_clear_all(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        store.create("p_0", "A", "Content A", [], tick=1)
        store.create("p_1", "B", "Content B", [], tick=2)
        store.clear()
        assert len(store) == 0


# ── Prompt formatting tests ──────────────────────────────────────────────────

class TestFormatForPrompt:
    def test_empty_returns_empty_string(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        assert store.format_for_prompt("p_0", "anything", tick=1) == ""

    def test_returns_formatted_block(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        store.create("p_0", "Pressure", "I felt pressured", ["emotion"], tick=1)
        result = store.format_for_prompt("p_0", "stress", tick=2)
        assert "MEMORIES" in result
        assert "Pressure" in result
        assert "I felt pressured" in result
        assert "emotion" in result


# ── Structured output integration ─────────────────────────────────────────────

class TestStructuredOutput:
    def test_process_with_memory(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        parsed = {
            "action": "respond",
            "narrative": "I noticed something important.",
            "memory": {
                "title": "Important observation",
                "content": "The scenario changed direction unexpectedly.",
                "tags": ["turning_point"],
            },
        }
        mem = store.process_agent_memory_output("p_0", parsed, tick=3)
        assert mem is not None
        assert mem.title == "Important observation"
        assert "turning_point" in mem.tags
        assert len(store) == 1

    def test_process_without_memory(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        parsed = {"action": "respond", "narrative": "Nothing special."}
        mem = store.process_agent_memory_output("p_0", parsed, tick=1)
        assert mem is None
        assert len(store) == 0

    def test_process_none_parsed(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        mem = store.process_agent_memory_output("p_0", None, tick=1)
        assert mem is None

    def test_process_empty_memory(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        parsed = {"action": "respond", "memory": {"title": "", "content": ""}}
        mem = store.process_agent_memory_output("p_0", parsed, tick=1)
        assert mem is None

    def test_process_string_tags(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        parsed = {
            "action": "respond",
            "memory": {"title": "T", "content": "C", "tags": "single_tag"},
        }
        mem = store.process_agent_memory_output("p_0", parsed, tick=1)
        assert mem is not None
        assert mem.tags == ["single_tag"]


# ── Export tests ──────────────────────────────────────────────────────────────

class TestExport:
    def test_export_empty(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        assert store.export() == []

    def test_export_sorted(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        store.create("p_1", "B", "Content B", [], tick=2)
        store.create("p_0", "A", "Content A", [], tick=1)
        exported = store.export()
        assert len(exported) == 2
        assert exported[0]["agent_id"] == "p_0"
        assert exported[1]["agent_id"] == "p_1"

    def test_export_no_embedding(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        store.create("p_0", "A", "Content A", ["tag"], tick=1)
        exported = store.export()
        assert "embedding" not in exported[0]


# ── Stats tests ───────────────────────────────────────────────────────────────

class TestStats:
    def test_stats(self):
        store = AgentMemoryStore(embed_fn=_mock_embed)
        store.create("p_0", "A", "Content A", [], tick=1)
        store.create("p_0", "B", "Content B", [], tick=2)
        store.create("p_1", "C", "Content C", [], tick=3)
        stats = store.stats()
        assert stats["total_memories"] == 3
        assert stats["agents"] == 2
        assert stats["per_agent"]["p_0"] == 2
        assert stats["per_agent"]["p_1"] == 1


# ── Integration with WorldState ───────────────────────────────────────────────

class TestWorldStateIntegration:
    def test_worldstate_has_memory_field(self):
        from simulation.sim_loop import WorldState
        ws = WorldState(k=3)
        # Default is None (initialized in run_simulation)
        assert ws.memory is None

    def test_worldstate_with_memory_store(self):
        from simulation.sim_loop import WorldState
        store = AgentMemoryStore(embed_fn=_mock_embed)
        ws = WorldState(k=3, memory=store)
        assert ws.memory is store
        ws.memory.create("p_0", "Test", "Content", [], tick=1)
        assert len(ws.memory) == 1
