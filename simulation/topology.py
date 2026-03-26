"""
Dual-environment topology manager for dualmirakl.

Inspired by MiroFish's dual-platform architecture (Twitter broadcast +
Reddit community): running agents through multiple interaction topologies
simultaneously produces richer emergent behavior than a single topology.

Supported topologies:
  - independent: each participant gets unique stimuli (current default)
  - clustered: participants grouped into clusters, stimuli flow through
               cluster context (agents see neighbors' presence)

Implementation follows Atlas Option A — "dual-context, single-response":
  - Multiple topologies generate stimuli in Phase A
  - Participant prompt combines all topology stimuli
  - Single response per participant (no GPU cost increase on swarm)
  - Single score signal (no signal merging needed)

Backward compatible: no topologies config = single independent topology.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Cluster ──────────────────────────────────────────────────────────────────

@dataclass
class Cluster:
    """A group of participants that share stimulus context."""
    id: str
    members: list[str]

    @property
    def size(self) -> int:
        return len(self.members)

    def contains(self, agent_id: str) -> bool:
        return agent_id in self.members

    def neighbors(self, agent_id: str) -> list[str]:
        """Get cluster members excluding the given agent."""
        return [m for m in self.members if m != agent_id]


# ── TopologyManager ─────────────────────────────────────────────────────────

class TopologyManager:
    """
    Manages topology assignments for the simulation.

    Holds cluster assignments per topology and provides methods to
    query cluster membership and build topology-aware prompts.
    """

    def __init__(self):
        self._clusters: dict[str, list[Cluster]] = {}  # topology_id → clusters

    def assign_clusters(
        self,
        topology_id: str,
        participant_ids: list[str],
        cluster_size: int = 2,
        rng=None,
    ) -> list[Cluster]:
        """
        Assign participants to clusters for a given topology.

        Handles remainders: last cluster may be larger than cluster_size.
        Uses rng for shuffle if provided (for reproducibility).
        """
        ids = list(participant_ids)
        if rng is not None:
            rng.shuffle(ids)

        clusters = []
        n_clusters = max(1, math.ceil(len(ids) / cluster_size))

        for i in range(n_clusters):
            start = i * cluster_size
            end = start + cluster_size
            if i == n_clusters - 1:
                # Last cluster gets all remaining
                members = ids[start:]
            else:
                members = ids[start:end]
            if members:
                clusters.append(Cluster(
                    id=f"{topology_id}_cluster_{i}",
                    members=members,
                ))

        self._clusters[topology_id] = clusters
        logger.debug(
            f"[topology] {topology_id}: {len(clusters)} clusters "
            f"from {len(ids)} participants (size={cluster_size})"
        )
        return clusters

    def get_clusters(self, topology_id: str) -> list[Cluster]:
        """Get clusters for a topology."""
        return self._clusters.get(topology_id, [])

    def get_cluster_for(self, topology_id: str, agent_id: str) -> Optional[Cluster]:
        """Find which cluster an agent belongs to."""
        for cluster in self._clusters.get(topology_id, []):
            if cluster.contains(agent_id):
                return cluster
        return None

    def get_neighbors(self, topology_id: str, agent_id: str) -> list[str]:
        """Get an agent's cluster neighbors for a topology."""
        cluster = self.get_cluster_for(topology_id, agent_id)
        if cluster is None:
            return []
        return cluster.neighbors(agent_id)

    @property
    def topology_ids(self) -> list[str]:
        return list(self._clusters.keys())

    def stats(self) -> dict:
        return {
            tid: {
                "n_clusters": len(clusters),
                "sizes": [c.size for c in clusters],
            }
            for tid, clusters in self._clusters.items()
        }


# ── Stimulus combination ────────────────────────────────────────────────────

def combine_stimuli(
    all_stimuli: dict[str, dict[str, str]],
    agent_id: str,
    topology_configs: list,
) -> str:
    """
    Combine stimuli from multiple topologies into a single participant prompt.

    Args:
        all_stimuli: {topology_id: {agent_id: stimulus_text}}
        agent_id: participant to build prompt for
        topology_configs: list of topology config objects (with .id, .type)

    Returns:
        Combined stimulus text with topology labels.
    """
    if len(all_stimuli) == 1:
        # Single topology — no labels needed, backward compatible
        topo_id = next(iter(all_stimuli))
        return all_stimuli[topo_id].get(agent_id, "")

    parts = []
    for topo_cfg in topology_configs:
        stim = all_stimuli.get(topo_cfg.id, {}).get(agent_id, "")
        if stim:
            label = _topology_label(topo_cfg)
            parts.append(f"[{label}] {stim}")

    return "\n".join(parts)


def _topology_label(topo_cfg) -> str:
    """Human-readable label for a topology."""
    if hasattr(topo_cfg, 'id'):
        tid = topo_cfg.id
    else:
        tid = str(topo_cfg)
    # Capitalize and make friendly
    return tid.replace("_", " ").title()
