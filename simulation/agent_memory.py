"""
Structured agent memory store for dualmirakl.

Inspired by Windsurf's create_memory tool: persistent memory with titles,
tags, workspace scoping, and CRUD operations. Layers structured memory
on top of e5-small-v2 embeddings for agents to save/recall explicit facts
between ticks.

Design principles:
  - Memories persist across ticks within a simulation run
  - Each agent has its own memory scope (like Windsurf's CorpusNames)
  - Semantic retrieval via cosine similarity (reuses e5-small-v2)
  - Tag-based retrieval for deterministic lookups
  - Deduplication: cosine > 0.9 triggers update instead of new memory
  - Eviction: LRU when agent exceeds max_per_agent
  - Proactive creation: agents can store memories via structured output
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Single memory entry for an agent."""
    id: str
    agent_id: str
    title: str
    content: str
    tags: list[str]
    tick_created: int
    tick_accessed: int
    embedding: np.ndarray

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "title": self.title,
            "content": self.content,
            "tags": self.tags,
            "tick_created": self.tick_created,
            "tick_accessed": self.tick_accessed,
        }


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Delegates to signal_computation._cosine for consistency.
    Kept as a named function for backward compatibility with tests.
    """
    from simulation.signal_computation import _cosine
    return _cosine(a, b)


class AgentMemoryStore:
    """
    Per-agent memory store with semantic and tag-based retrieval.

    Args:
        embed_fn: function that takes a string and returns a numpy vector
                  (e.g., model.encode wrapped in a lambda)
        max_per_agent: maximum memories per agent before LRU eviction
        dedup_threshold: cosine similarity above which a memory is updated
                         instead of creating a new one
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        max_per_agent: int = 20,
        dedup_threshold: float = 0.9,
    ):
        self._memories: dict[str, list[Memory]] = defaultdict(list)
        self._embed = embed_fn
        self._max = max_per_agent
        self._dedup_threshold = dedup_threshold

    # ── Create / Update ───────────────────────────────────────────────────

    def create(
        self,
        agent_id: str,
        title: str,
        content: str,
        tags: list[str],
        tick: int,
    ) -> Memory:
        """
        Store a new memory. Deduplicates by cosine similarity:
        if an existing memory is > dedup_threshold similar, update it instead.
        """
        vec = self._embed(content)
        agent_mems = self._memories[agent_id]

        # Check for duplicate
        for existing in agent_mems:
            sim = _cosine_sim(vec, existing.embedding)
            if sim > self._dedup_threshold:
                logger.debug(
                    f"[memory] {agent_id}: dedup update '{existing.title}' "
                    f"(sim={sim:.3f})"
                )
                existing.title = title
                existing.content = content
                existing.tags = list(set(existing.tags + tags))
                existing.tick_accessed = tick
                existing.embedding = vec
                return existing

        # Create new memory
        mem = Memory(
            id=str(uuid4()),
            agent_id=agent_id,
            title=title,
            content=content,
            tags=list(tags),
            tick_created=tick,
            tick_accessed=tick,
            embedding=vec,
        )
        agent_mems.append(mem)

        # Evict oldest-accessed if over limit
        if len(agent_mems) > self._max:
            self._evict_lru(agent_id)

        return mem

    def _evict_lru(self, agent_id: str) -> None:
        """Remove the least recently accessed memory for this agent."""
        mems = self._memories[agent_id]
        if not mems:
            return
        oldest = min(mems, key=lambda m: m.tick_accessed)
        mems.remove(oldest)
        logger.debug(f"[memory] {agent_id}: evicted '{oldest.title}' (LRU)")

    # ── Retrieve ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        agent_id: str,
        query: str,
        tick: int,
        top_k: int = 3,
    ) -> list[Memory]:
        """
        Semantic retrieval — find most relevant memories for a query.
        Updates tick_accessed on returned memories.
        """
        agent_mems = self._memories.get(agent_id, [])
        if not agent_mems:
            return []

        q_vec = self._embed(query)
        scored = [(_cosine_sim(q_vec, m.embedding), m) for m in agent_mems]
        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        for sim, mem in scored[:top_k]:
            mem.tick_accessed = tick
            result.append(mem)
        return result

    def retrieve_by_tags(
        self,
        agent_id: str,
        tags: list[str],
    ) -> list[Memory]:
        """
        Tag-based retrieval — deterministic, no embedding needed.
        Returns memories that match ANY of the given tags.
        """
        agent_mems = self._memories.get(agent_id, [])
        tag_set = set(tags)
        return [m for m in agent_mems if set(m.tags) & tag_set]

    def get_all(self, agent_id: str) -> list[Memory]:
        """Get all memories for an agent."""
        return list(self._memories.get(agent_id, []))

    # ── Delete ────────────────────────────────────────────────────────────

    def delete(self, agent_id: str, memory_id: str) -> bool:
        """Delete a specific memory by ID. Returns True if found."""
        agent_mems = self._memories.get(agent_id, [])
        for mem in agent_mems:
            if mem.id == memory_id:
                agent_mems.remove(mem)
                return True
        return False

    def clear(self, agent_id: Optional[str] = None) -> None:
        """Clear all memories for an agent, or all agents if None."""
        if agent_id:
            self._memories[agent_id] = []
        else:
            self._memories.clear()

    # ── Prompt injection ──────────────────────────────────────────────────

    def format_for_prompt(
        self,
        agent_id: str,
        query: str,
        tick: int,
        top_k: int = 3,
    ) -> str:
        """
        Retrieve and format memories for injection into agent's prompt.
        Returns empty string if no memories exist.
        """
        memories = self.retrieve(agent_id, query, tick, top_k)
        if not memories:
            return ""

        lines = ["[MEMORIES — things you remember from earlier:]"]
        for mem in memories:
            tag_str = f" [{', '.join(mem.tags)}]" if mem.tags else ""
            lines.append(f"  - {mem.title}{tag_str}: {mem.content}")
        lines.append("")
        return "\n".join(lines)

    # ── Structured output integration ─────────────────────────────────────

    def process_agent_memory_output(
        self,
        agent_id: str,
        parsed: Optional[dict],
        tick: int,
    ) -> Optional[Memory]:
        """
        Extract memory from structured agent output and store it.
        Looks for a "memory" field in the parsed action output:
          {"memory": {"title": "...", "content": "...", "tags": [...]}}
        Returns the created/updated Memory, or None if no memory in output.
        """
        if not parsed:
            return None

        mem_data = parsed.get("memory")
        if not mem_data or not isinstance(mem_data, dict):
            return None

        title = mem_data.get("title", "").strip()
        content = mem_data.get("content", "").strip()
        if not title or not content:
            return None

        tags = mem_data.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        return self.create(agent_id, title, content, tags, tick)

    # ── Introspection ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return sum(len(mems) for mems in self._memories.values())

    @property
    def agents(self) -> list[str]:
        return [a for a, mems in self._memories.items() if mems]

    def stats(self) -> dict:
        """Summary statistics for the memory store."""
        return {
            "total_memories": len(self),
            "agents": len(self.agents),
            "per_agent": {
                agent_id: len(mems)
                for agent_id, mems in self._memories.items()
                if mems
            },
        }

    # ── Export ────────────────────────────────────────────────────────────

    def export(self) -> list[dict]:
        """Export all memories as list of dicts (for JSON serialization)."""
        result = []
        for mems in self._memories.values():
            for mem in mems:
                result.append(mem.to_dict())
        result.sort(key=lambda m: (m["agent_id"], m["tick_created"]))
        return result


# ── DuckDB persistence layer ─────────────────────────────────────────────────

class DuckDBMemoryBackend:
    """
    Persistent memory backend using DuckDB.

    Sits alongside the in-memory AgentMemoryStore as a write-behind layer.
    During simulation: memories accumulate in-memory (fast path).
    At tick boundaries: new memories are batch-flushed to DuckDB.
    Across runs: memories can be loaded from prior runs via load_from_run().

    Usage:
        backend = DuckDBMemoryBackend(run_id="run_20260325_120000_s42")
        backend.flush(memory_store)          # after each tick
        backend.flush_all(memory_store)      # at simulation end

        # For a new run continuing from a prior one:
        prior_memories = backend.load_from_run(
            prior_run_id="run_20260324_...",
            scenario_context="...",
            embed_fn=lambda texts: model.encode(texts),
            top_k=5,
        )
    """

    def __init__(self, run_id: str, db=None):
        self.run_id = run_id
        self._db = db
        self._flushed_ids: set[str] = set()

    @property
    def db(self):
        if self._db is None:
            from simulation.storage import get_db
            self._db = get_db()
        return self._db

    def flush(self, store: AgentMemoryStore) -> int:
        """
        Flush new (unflushed) memories from the in-memory store to DuckDB.
        Call this at tick boundaries. Returns count of newly persisted memories.
        """
        count = 0
        for agent_id in store.agents:
            for mem in store.get_all(agent_id):
                if mem.id in self._flushed_ids:
                    continue
                self._persist_memory(mem)
                self._flushed_ids.add(mem.id)
                count += 1
        return count

    def flush_all(self, store: AgentMemoryStore) -> int:
        """Flush all memories (call at simulation end)."""
        return self.flush(store)

    def _persist_memory(self, mem: Memory) -> None:
        """Insert a single memory into DuckDB."""
        embedding_list = mem.embedding.tolist() if mem.embedding is not None else None
        self.db.execute(
            """INSERT OR REPLACE INTO agent_memories
               (id, run_id, agent_id, tick, memory_type, title, content, tags,
                embedding, importance, decay_rate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [mem.id, self.run_id, mem.agent_id, mem.tick_created,
             "episodic", mem.title, mem.content, mem.tags,
             embedding_list, 1.0, 0.05],
        )

    def load_from_run(
        self,
        prior_run_id: str,
        scenario_context: str,
        embed_fn,
        top_k: int = 5,
        decay_weight: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Load relevant memories from a prior run for each agent.

        Retrieves top-K memories per agent from the prior run, ranked by
        cosine similarity to the new scenario context, optionally weighted
        by importance and temporal decay.

        Args:
            prior_run_id: Run ID to load memories from.
            scenario_context: Current scenario description for relevance filtering.
            embed_fn: Batch embedding function.
            top_k: Max memories per agent.
            decay_weight: Apply temporal decay to importance.

        Returns:
            Dict of agent_id -> list of memory dicts.
        """
        from simulation.storage import cosine_similarity_sql

        # Embed the scenario context
        q_vec = embed_fn([scenario_context])[0].tolist()
        sim_expr = cosine_similarity_sql("embedding", q_vec)

        rows = self.db.execute(f"""
            SELECT agent_id, title, content, tags, tick,
                   importance, decay_rate,
                   {sim_expr} AS similarity
            FROM agent_memories
            WHERE run_id = ?
              AND embedding IS NOT NULL
            ORDER BY agent_id, similarity DESC
        """, [prior_run_id]).fetchall()

        # Group by agent, take top_k per agent
        from collections import defaultdict
        agent_memories: dict[str, list] = defaultdict(list)

        for agent_id, title, content, tags, tick, importance, decay_rate, sim in rows:
            if len(agent_memories[agent_id]) >= top_k:
                continue

            # Apply decay if requested
            effective_importance = importance
            if decay_weight:
                import math
                effective_importance = importance * math.exp(-decay_rate * tick)

            agent_memories[agent_id].append({
                "title": title,
                "content": content,
                "tags": tags or [],
                "tick_created": tick,
                "similarity": float(sim),
                "effective_importance": float(effective_importance),
            })

        return dict(agent_memories)

    def get_run_ids(self) -> list[str]:
        """List all run IDs that have persisted memories."""
        rows = self.db.execute(
            "SELECT DISTINCT run_id FROM agent_memories ORDER BY run_id DESC"
        ).fetchall()
        return [r[0] for r in rows]

    def memory_stats(self, run_id: Optional[str] = None) -> dict:
        """Get memory statistics for a run (or all runs)."""
        if run_id:
            row = self.db.execute("""
                SELECT COUNT(*) as total,
                       COUNT(DISTINCT agent_id) as agents,
                       COUNT(DISTINCT run_id) as runs
                FROM agent_memories WHERE run_id = ?
            """, [run_id]).fetchone()
        else:
            row = self.db.execute("""
                SELECT COUNT(*) as total,
                       COUNT(DISTINCT agent_id) as agents,
                       COUNT(DISTINCT run_id) as runs
                FROM agent_memories
            """).fetchone()
        return {"total_memories": row[0], "agents": row[1], "runs": row[2]}
