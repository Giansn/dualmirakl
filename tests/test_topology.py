"""
Tests for the dual-environment topology module.

Run: python -m pytest tests/test_topology.py -v
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.topology import (
    Cluster,
    TopologyManager,
    combine_stimuli,
    _topology_label,
)


# ── Cluster tests ────────────────────────────────────────────────────────────

class TestCluster:
    def test_creation(self):
        c = Cluster(id="c_0", members=["p_0", "p_1"])
        assert c.size == 2
        assert c.contains("p_0")
        assert not c.contains("p_2")

    def test_neighbors(self):
        c = Cluster(id="c_0", members=["p_0", "p_1", "p_2"])
        assert c.neighbors("p_0") == ["p_1", "p_2"]
        assert c.neighbors("p_1") == ["p_0", "p_2"]

    def test_neighbors_single_member(self):
        c = Cluster(id="c_0", members=["p_0"])
        assert c.neighbors("p_0") == []

    def test_neighbors_nonexistent(self):
        c = Cluster(id="c_0", members=["p_0", "p_1"])
        assert c.neighbors("p_99") == ["p_0", "p_1"]


# ── TopologyManager tests ───────────────────────────────────────────────────

class TestTopologyManager:
    def test_assign_clusters_even(self):
        tm = TopologyManager()
        clusters = tm.assign_clusters(
            "community", ["p_0", "p_1", "p_2", "p_3"], cluster_size=2,
        )
        assert len(clusters) == 2
        assert all(c.size == 2 for c in clusters)

    def test_assign_clusters_remainder(self):
        """Last cluster gets extra members when count doesn't divide evenly."""
        tm = TopologyManager()
        clusters = tm.assign_clusters(
            "community", ["p_0", "p_1", "p_2", "p_3", "p_4"], cluster_size=2,
        )
        assert len(clusters) == 3
        sizes = sorted([c.size for c in clusters])
        assert sizes == [1, 2, 2]

    def test_assign_clusters_single(self):
        tm = TopologyManager()
        clusters = tm.assign_clusters("community", ["p_0"], cluster_size=2)
        assert len(clusters) == 1
        assert clusters[0].size == 1

    def test_assign_clusters_large_size(self):
        """cluster_size larger than participant count = one cluster."""
        tm = TopologyManager()
        clusters = tm.assign_clusters(
            "community", ["p_0", "p_1"], cluster_size=10,
        )
        assert len(clusters) == 1
        assert clusters[0].size == 2

    def test_get_clusters(self):
        tm = TopologyManager()
        tm.assign_clusters("community", ["p_0", "p_1"], cluster_size=2)
        assert len(tm.get_clusters("community")) == 1
        assert tm.get_clusters("nonexistent") == []

    def test_get_cluster_for(self):
        tm = TopologyManager()
        tm.assign_clusters("community", ["p_0", "p_1", "p_2", "p_3"], cluster_size=2)
        c = tm.get_cluster_for("community", "p_0")
        assert c is not None
        assert c.contains("p_0")

    def test_get_cluster_for_not_found(self):
        tm = TopologyManager()
        tm.assign_clusters("community", ["p_0", "p_1"], cluster_size=2)
        assert tm.get_cluster_for("community", "p_99") is None

    def test_get_neighbors(self):
        tm = TopologyManager()
        tm.assign_clusters("community", ["p_0", "p_1", "p_2", "p_3"], cluster_size=2)
        neighbors = tm.get_neighbors("community", "p_0")
        assert len(neighbors) == 1  # one neighbor in cluster of 2

    def test_get_neighbors_not_found(self):
        tm = TopologyManager()
        assert tm.get_neighbors("community", "p_99") == []

    def test_topology_ids(self):
        tm = TopologyManager()
        tm.assign_clusters("community", ["p_0", "p_1"], cluster_size=2)
        tm.assign_clusters("broadcast", ["p_0", "p_1"], cluster_size=2)
        assert set(tm.topology_ids) == {"community", "broadcast"}

    def test_stats(self):
        tm = TopologyManager()
        tm.assign_clusters("community", ["p_0", "p_1", "p_2", "p_3"], cluster_size=2)
        s = tm.stats()
        assert s["community"]["n_clusters"] == 2
        assert s["community"]["sizes"] == [2, 2]

    def test_assign_with_rng(self):
        """Using a seeded rng produces deterministic clusters."""
        import numpy as np
        tm1 = TopologyManager()
        tm2 = TopologyManager()
        ids = [f"p_{i}" for i in range(6)]
        c1 = tm1.assign_clusters("x", ids, cluster_size=2, rng=np.random.default_rng(42))
        c2 = tm2.assign_clusters("x", ids, cluster_size=2, rng=np.random.default_rng(42))
        assert [c.members for c in c1] == [c.members for c in c2]


# ── Stimulus combination tests ──────────────────────────────────────────────

class _FakeTopoConfig:
    def __init__(self, id, type="independent"):
        self.id = id
        self.type = type


class TestCombineStimuli:
    def test_single_topology_no_label(self):
        """Single topology returns raw stimulus, no labels."""
        stimuli = {"broadcast": {"p_0": "Hello from broadcast"}}
        result = combine_stimuli(stimuli, "p_0", [_FakeTopoConfig("broadcast")])
        assert result == "Hello from broadcast"
        assert "[" not in result  # no label prefix

    def test_multi_topology_labels(self):
        """Multiple topologies get labeled."""
        stimuli = {
            "broadcast": {"p_0": "News flash"},
            "community": {"p_0": "Your neighbor said hi"},
        }
        configs = [_FakeTopoConfig("broadcast"), _FakeTopoConfig("community", "clustered")]
        result = combine_stimuli(stimuli, "p_0", configs)
        assert "[Broadcast]" in result
        assert "[Community]" in result
        assert "News flash" in result
        assert "Your neighbor said hi" in result

    def test_missing_agent_returns_empty(self):
        stimuli = {"broadcast": {"p_0": "Hello"}}
        result = combine_stimuli(stimuli, "p_99", [_FakeTopoConfig("broadcast")])
        assert result == ""

    def test_topology_label(self):
        cfg = _FakeTopoConfig("my_topology")
        assert _topology_label(cfg) == "My Topology"


# ── ScenarioConfig integration ──────────────────────────────────────────────

class TestTopologyConfig:
    def test_default_topology(self):
        from simulation.scenario import ScenarioConfig
        config = ScenarioConfig.from_dict({
            "meta": {"name": "test", "version": "1.0"},
            "agents": {"roles": [
                {"id": "p", "slot": "swarm", "type": "participant",
                 "system_prompt": "test", "count": 2},
            ]},
        })
        assert len(config.topologies) == 1
        assert config.topologies[0].type == "independent"

    def test_dual_topology(self):
        from simulation.scenario import ScenarioConfig
        config = ScenarioConfig.from_dict({
            "meta": {"name": "test", "version": "1.0"},
            "agents": {"roles": [
                {"id": "p", "slot": "swarm", "type": "participant",
                 "system_prompt": "test", "count": 4},
            ]},
            "topologies": [
                {"id": "broadcast", "type": "independent"},
                {"id": "community", "type": "clustered", "cluster_size": 2},
            ],
        })
        assert len(config.topologies) == 2
        assert config.topologies[1].type == "clustered"
        assert config.topologies[1].cluster_size == 2

    def test_invalid_topology_type(self):
        from simulation.scenario import TopologyConfig
        with pytest.raises(Exception):
            TopologyConfig(id="bad", type="unknown")

    def test_topology_defaults(self):
        from simulation.scenario import TopologyConfig
        cfg = TopologyConfig()
        assert cfg.id == "broadcast"
        assert cfg.type == "independent"
        assert cfg.weight == 1.0
        assert cfg.cluster_size == 2
        assert cfg.recluster_interval == 0
