"""
Real-time graph memory feedback loop for dualmirakl.

Inspired by MiroFish's Zep-backed knowledge graph: agent actions during
simulation are continuously distilled into a shared graph that evolves as
the simulation progresses. Unlike per-agent AgentMemoryStore (private,
semantic), GraphMemory is shared world-state knowledge with temporal edges.

Architecture:
  - NetworkX DiGraph (no external dependency — self-contained)
  - Entities: agents, archetypes, interventions, stimuli patterns
  - Relationships: scored_at, responded_to, intervention_on, escalated,
                   clustered_with, flagged_by
  - Temporal edges: valid_at (tick created), invalid_at (tick expired)
  - Distiller: EventStream → graph updates after each tick

Usage:
    graph = GraphMemory()
    graph.distill_tick(tick, world_state.stream)   # after each tick
    neighbors = graph.neighbors("participant_0")    # query
    path = graph.relationship_path("p_0", "p_1")   # path finding
    summary = graph.summarize_agent("participant_0") # LLM-ready summary
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Edge types ───────────────────────────────────────────────────────────────

SCORED_AT = "scored_at"
RESPONDED_TO = "responded_to"
INTERVENTION_ON = "intervention_on"
ESCALATED = "escalated"
DISENGAGED = "disengaged"
FLAGGED_BY = "flagged_by"
CLUSTERED_WITH = "clustered_with"
RECEIVED_STIMULUS = "received_stimulus"

ALL_EDGE_TYPES = frozenset({
    SCORED_AT, RESPONDED_TO, INTERVENTION_ON, ESCALATED,
    DISENGAGED, FLAGGED_BY, CLUSTERED_WITH, RECEIVED_STIMULUS,
})


# ── Temporal edge ────────────────────────────────────────────────────────────

@dataclass
class TemporalEdge:
    """Edge with temporal validity — knows when it was created and when it expired."""
    edge_type: str
    valid_at: int       # tick when this relationship became active
    invalid_at: int     # tick when this relationship expired (-1 = still active)
    properties: dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.invalid_at == -1

    def expire(self, tick: int) -> None:
        if self.invalid_at == -1:
            self.invalid_at = tick

    def to_dict(self) -> dict:
        return {
            "edge_type": self.edge_type,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "properties": self.properties,
        }


# ── GraphMemory ──────────────────────────────────────────────────────────────

class GraphMemory:
    """
    Shared knowledge graph that evolves with the simulation.

    Nodes represent entities (agents, intervention types, score regions).
    Edges represent relationships with temporal metadata.

    The graph is updated after each tick by distilling events from the
    EventStream. This creates a feedback loop: agent actions → graph →
    observer context → richer analysis.
    """

    def __init__(self):
        # Node storage: {node_id: {type, label, properties, tick_created, tick_updated}}
        self._nodes: dict[str, dict] = {}
        # Edge storage: {(src, dst): [TemporalEdge, ...]}
        self._edges: dict[tuple[str, str], list[TemporalEdge]] = defaultdict(list)
        # Index: node_id → set of connected node_ids
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        # Track last distilled tick
        self._last_tick: int = 0

    # ── Node operations ──────────────────────────────────────────────────

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str = "",
        tick: int = 0,
        **properties,
    ) -> None:
        """Add or update a node."""
        if node_id in self._nodes:
            self._nodes[node_id]["tick_updated"] = tick
            self._nodes[node_id]["properties"].update(properties)
        else:
            self._nodes[node_id] = {
                "type": node_type,
                "label": label or node_id,
                "properties": dict(properties),
                "tick_created": tick,
                "tick_updated": tick,
            }

    def get_node(self, node_id: str) -> Optional[dict]:
        """Get node data by ID."""
        return self._nodes.get(node_id)

    def nodes_by_type(self, node_type: str) -> list[str]:
        """Get all node IDs of a given type."""
        return [
            nid for nid, data in self._nodes.items()
            if data["type"] == node_type
        ]

    # ── Edge operations ──────────────────────────────────────────────────

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        tick: int,
        **properties,
    ) -> TemporalEdge:
        """Add a temporal edge between two nodes."""
        edge = TemporalEdge(
            edge_type=edge_type,
            valid_at=tick,
            invalid_at=-1,
            properties=dict(properties),
        )
        self._edges[(src, dst)].append(edge)
        self._adjacency[src].add(dst)
        self._adjacency[dst].add(src)
        return edge

    def get_edges(
        self,
        src: str,
        dst: Optional[str] = None,
        edge_type: Optional[str] = None,
        active_only: bool = False,
    ) -> list[tuple[str, str, TemporalEdge]]:
        """Query edges from a source node with optional filters."""
        results = []
        for (s, d), edges in self._edges.items():
            if s != src and d != src:
                continue
            if dst is not None and d != dst and s != dst:
                continue
            for edge in edges:
                if edge_type is not None and edge.edge_type != edge_type:
                    continue
                if active_only and not edge.is_active:
                    continue
                results.append((s, d, edge))
        return results

    def expire_edges(
        self,
        src: str,
        dst: str,
        edge_type: str,
        tick: int,
    ) -> int:
        """Expire all active edges of a given type between src and dst. Returns count."""
        count = 0
        for edge in self._edges.get((src, dst), []):
            if edge.edge_type == edge_type and edge.is_active:
                edge.expire(tick)
                count += 1
        return count

    # ── Query operations ─────────────────────────────────────────────────

    def neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        active_only: bool = True,
    ) -> list[str]:
        """Get neighboring node IDs, optionally filtered by edge type."""
        if edge_type is None and not active_only:
            return list(self._adjacency.get(node_id, set()))

        result = set()
        for (s, d), edges in self._edges.items():
            if s != node_id and d != node_id:
                continue
            other = d if s == node_id else s
            for edge in edges:
                if edge_type is not None and edge.edge_type != edge_type:
                    continue
                if active_only and not edge.is_active:
                    continue
                result.add(other)
                break
        return list(result)

    def relationship_path(
        self,
        src: str,
        dst: str,
        max_depth: int = 3,
    ) -> Optional[list[str]]:
        """BFS shortest path between two nodes. Returns node path or None."""
        if src not in self._nodes or dst not in self._nodes:
            return None
        if src == dst:
            return [src]

        visited = {src}
        queue = [(src, [src])]

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                break
            for neighbor in self._adjacency.get(current, set()):
                if neighbor == dst:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def summarize_agent(self, agent_id: str) -> str:
        """
        Generate an LLM-ready summary of an agent's graph context.
        Includes: current relationships, score trajectory edges, interventions.
        """
        node = self.get_node(agent_id)
        if node is None:
            return f"Agent '{agent_id}' not found in graph."

        lines = [f"{node['label']} ({node['type']})"]
        props = node["properties"]
        if props:
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
            lines.append(f"  Properties: {prop_str}")

        # Active relationships
        active_edges = self.get_edges(agent_id, active_only=True)
        if active_edges:
            lines.append(f"  Active relationships ({len(active_edges)}):")
            for s, d, edge in active_edges[:10]:
                other = d if s == agent_id else s
                lines.append(
                    f"    -> {other} [{edge.edge_type}] "
                    f"(since T{edge.valid_at})"
                )

        # Flagged by observers
        flags = self.get_edges(agent_id, edge_type=FLAGGED_BY)
        if flags:
            lines.append(f"  Flagged {len(flags)} time(s)")

        return "\n".join(lines)

    # ── Event stream distillation ────────────────────────────────────────

    def distill_tick(self, tick: int, stream) -> int:
        """
        Distill events from a single tick into graph updates.

        Processes: score events, response events, observation events,
        intervention events, compliance events.

        Returns number of graph operations performed.
        """
        ops = 0
        tick_events = stream.tick_events(tick)

        for event in tick_events:
            if event.event_type == "score":
                ops += self._distill_score(tick, event)
            elif event.event_type == "response":
                ops += self._distill_response(tick, event)
            elif event.event_type == "observation":
                ops += self._distill_observation(tick, event)
            elif event.event_type == "intervention":
                ops += self._distill_intervention(tick, event)
            elif event.event_type == "stimulus":
                ops += self._distill_stimulus(tick, event)

        self._last_tick = tick
        if ops > 0:
            logger.debug(f"[graph] tick {tick}: {ops} graph operations")
        return ops

    def _distill_score(self, tick: int, event) -> int:
        """Score event → update agent node + score_region edges."""
        agent_id = event.agent_id
        score = event.payload.get("score_after", 0.0)
        signal = event.payload.get("signal", 0.0)

        # Ensure agent node exists
        self.add_node(agent_id, "agent", tick=tick, latest_score=score)

        # Determine score region
        if score >= 0.8:
            region = "high_engagement"
        elif score >= 0.5:
            region = "moderate_engagement"
        else:
            region = "low_engagement"

        self.add_node(region, "score_region", label=region, tick=tick)

        # Expire old score_region edges for this agent
        for region_id in ["high_engagement", "moderate_engagement", "low_engagement"]:
            if region_id != region:
                self.expire_edges(agent_id, region_id, SCORED_AT, tick)

        # Add current score edge (if not already active)
        active_in_region = self.get_edges(
            agent_id, dst=region, edge_type=SCORED_AT, active_only=True,
        )
        if not active_in_region:
            self.add_edge(agent_id, region, SCORED_AT, tick, score=score, signal=signal)
            return 2  # node update + edge

        # Update existing edge properties
        for _, _, edge in active_in_region:
            edge.properties["score"] = score
            edge.properties["signal"] = signal
        return 1  # node update only

    def _distill_response(self, tick: int, event) -> int:
        """Response event → check for escalation/disengagement signals."""
        agent_id = event.agent_id
        structured = event.payload.get("structured")
        if not structured:
            return 0

        action = structured.get("action", "")

        if action == "escalate":
            self.add_node(agent_id, "agent", tick=tick)
            self.add_node("escalation_pattern", "pattern", tick=tick)
            self.add_edge(
                agent_id, "escalation_pattern", ESCALATED, tick,
                trigger=structured.get("trigger", ""),
            )
            return 2

        if action == "disengage":
            self.add_node(agent_id, "agent", tick=tick)
            self.add_node("disengagement_pattern", "pattern", tick=tick)
            self.add_edge(
                agent_id, "disengagement_pattern", DISENGAGED, tick,
                reason=structured.get("reason", ""),
                duration=structured.get("duration", ""),
            )
            return 2

        return 0

    def _distill_observation(self, tick: int, event) -> int:
        """Observation event → extract flagged participants + clustering."""
        structured = event.payload.get("structured")
        if not structured:
            return 0

        ops = 0

        # Flagged participants
        flagged = structured.get("flagged_participants", [])
        for pid in flagged:
            self.add_node(pid, "agent", tick=tick)
            self.add_edge(pid, "observer_a", FLAGGED_BY, tick)
            ops += 1

        # Clustering state
        clustering = structured.get("clustering")
        if clustering:
            self.add_node(
                "population_state", "state",
                label=f"population ({clustering})",
                tick=tick,
                clustering=clustering,
                concern_level=structured.get("concern_level", "unknown"),
            )
            ops += 1

        return ops

    def _distill_intervention(self, tick: int, event) -> int:
        """Intervention event → create intervention node + edges."""
        iv_type = event.payload.get("type", "unknown")
        description = event.payload.get("description", "")
        source = event.payload.get("source", "observer_b")

        iv_node = f"intervention_T{tick}_{iv_type}"
        self.add_node(
            iv_node, "intervention",
            label=f"{iv_type} (T{tick})",
            tick=tick,
            intervention_type=iv_type,
            description=description,
        )

        # Edge from source observer
        self.add_node(source, "agent", tick=tick)
        self.add_edge(source, iv_node, INTERVENTION_ON, tick)

        # Edge to all agents (interventions affect population)
        for agent_id in self.nodes_by_type("agent"):
            if isinstance(agent_id, str) and agent_id.startswith("participant_"):
                self.add_edge(iv_node, agent_id, INTERVENTION_ON, tick)

        return 2

    def _distill_stimulus(self, tick: int, event) -> int:
        """Stimulus event → environment → participant edge."""
        agent_id = event.agent_id
        self.add_node(agent_id, "agent", tick=tick)
        self.add_node("environment", "agent", label="environment", tick=tick)
        self.add_edge("environment", agent_id, RECEIVED_STIMULUS, tick)
        return 1

    # ── GraphRAG seeding ────────────────────────────────────────────────

    def seed_from_graphrag(self, entities: list, relations: list) -> int:
        """
        Pre-populate the graph with entities and relations extracted from
        documents via the GraphRAG pipeline (simulation/graph_rag.py).

        Called before tick 0 to give the graph initial structure from
        domain knowledge. Runtime events (distill_tick) then layer on top.

        Args:
            entities: List of graph_rag.Entity objects.
            relations: List of graph_rag.Relation objects.

        Returns:
            Number of graph operations performed.
        """
        ops = 0
        entity_map = {}  # name -> node_id

        for ent in entities:
            node_id = f"kg_{ent.name.lower().replace(' ', '_')}"
            entity_map[ent.name] = node_id
            self.add_node(
                node_id,
                node_type=f"kg_{ent.type}",
                label=ent.name,
                tick=0,
                **ent.properties,
            )
            ops += 1

        for rel in relations:
            src_id = entity_map.get(rel.source)
            tgt_id = entity_map.get(rel.target)
            if src_id and tgt_id:
                self.add_edge(
                    src_id, tgt_id,
                    edge_type=f"kg_{rel.rel_type}",
                    tick=0,
                    context=rel.context,
                    weight=rel.weight,
                )
                ops += 1

        if ops > 0:
            logger.info(
                "[graph] seeded with %d entities + %d relations from GraphRAG",
                len(entities), len(relations),
            )

        return ops

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        return sum(len(edges) for edges in self._edges.values())

    @property
    def last_tick(self) -> int:
        return self._last_tick

    def stats(self) -> dict:
        """Summary statistics."""
        type_counts = defaultdict(int)
        for data in self._nodes.values():
            type_counts[data["type"]] += 1

        active_edges = sum(
            1 for edges in self._edges.values()
            for e in edges if e.is_active
        )

        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_active_edges": active_edges,
            "node_types": dict(type_counts),
            "last_tick": self._last_tick,
        }

    # ── Export ────────────────────────────────────────────────────────────

    def export(self) -> dict:
        """Export full graph as JSON-serializable dict."""
        nodes = {}
        for nid, data in self._nodes.items():
            nodes[nid] = {
                "type": data["type"],
                "label": data["label"],
                "properties": data["properties"],
                "tick_created": data["tick_created"],
                "tick_updated": data["tick_updated"],
            }

        edges = []
        for (src, dst), edge_list in self._edges.items():
            for edge in edge_list:
                edges.append({
                    "src": src,
                    "dst": dst,
                    **edge.to_dict(),
                })

        return {"nodes": nodes, "edges": edges}

    def query_text(
        self,
        node_id: Optional[str] = None,
        edge_type: Optional[str] = None,
        active_only: bool = True,
        include_properties: bool = True,
    ) -> str:
        """
        Human/LLM-readable text summary of graph state.
        Used by the query_graph observer tool.
        """
        lines = []

        if node_id:
            # Summarize a specific node
            node = self.get_node(node_id)
            if not node:
                return f"Node '{node_id}' not found."
            lines.append(self.summarize_agent(node_id))
        else:
            # Overall graph summary
            s = self.stats()
            lines.append(
                f"Graph: {s['n_nodes']} nodes, {s['n_edges']} edges "
                f"({s['n_active_edges']} active), last tick: {s['last_tick']}"
            )
            lines.append(f"Node types: {s['node_types']}")

            # List agents and their current regions
            agents = self.nodes_by_type("agent")
            if agents:
                lines.append(f"\nAgents ({len(agents)}):")
                for aid in sorted(agents):
                    if not isinstance(aid, str) or not aid.startswith("participant_"):
                        continue
                    props = self._nodes[aid].get("properties", {})
                    score = props.get("latest_score", "?")
                    score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
                    lines.append(f"  {aid}: score={score_str}")

            # Active interventions
            iv_nodes = self.nodes_by_type("intervention")
            if iv_nodes:
                lines.append(f"\nInterventions ({len(iv_nodes)}):")
                for iv_id in iv_nodes[-5:]:  # last 5
                    iv = self._nodes[iv_id]
                    lines.append(
                        f"  {iv['label']}: {iv['properties'].get('description', '')[:60]}"
                    )

            # Patterns
            patterns = self.nodes_by_type("pattern")
            if patterns:
                lines.append(f"\nBehavioral patterns: {', '.join(patterns)}")

        return "\n".join(lines)
