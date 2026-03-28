"""
Scenario tree construction from ensemble trajectory data.

Builds probability-weighted branching trees by clustering simulation
trajectories at key time steps using Ward's hierarchical clustering.
Supports Dupacova backward reduction for scenario tree simplification.

Usage:
    from simulation.scenario_tree import build_scenario_tree, reduce_tree

    tree = build_scenario_tree(all_score_logs, max_depth=3, max_branches=4)
    reduced = reduce_tree(tree, target_scenarios=5)
    print(tree_to_dict(tree))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


@dataclass
class ScenarioNode:
    """A node in the scenario probability tree."""

    id: str                                     # "root", "1", "1.2", etc.
    step: int                                   # tick index (0 = root)
    probability: float                          # fraction of runs in this branch
    metrics: dict = field(default_factory=dict)  # mean, std, p5, p25, median, p75, p95
    n_supporting_runs: int = 0
    centroid: Optional[np.ndarray] = None       # mean score vector at this step
    children: list["ScenarioNode"] = field(default_factory=list)
    run_indices: list[int] = field(default_factory=list)  # which runs belong here

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def n_leaves(self) -> int:
        if not self.children:
            return 1
        return sum(c.n_leaves() for c in self.children)


def _compute_metrics(values: np.ndarray) -> dict:
    """Compute summary statistics for a set of values."""
    if len(values) == 0:
        return {}
    return {
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values)), 6),
        "p5": round(float(np.percentile(values, 5)), 6),
        "p25": round(float(np.percentile(values, 25)), 6),
        "median": round(float(np.percentile(values, 50)), 6),
        "p75": round(float(np.percentile(values, 75)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
    }


def _extract_state_vectors(
    all_score_logs: list[list[list[float]]],
    step: int,
    run_indices: list[int],
) -> np.ndarray:
    """Extract per-run mean agent scores at a given step.

    Returns (n_runs,) array of mean scores at the step.
    """
    values = []
    for idx in run_indices:
        agent_scores = []
        for agent_log in all_score_logs[idx]:
            if step < len(agent_log):
                agent_scores.append(agent_log[step])
        values.append(float(np.mean(agent_scores)) if agent_scores else 0.0)
    return np.array(values)


def _cluster_at_step(
    state_vectors: np.ndarray,
    max_branches: int,
    min_branch_prob: float,
) -> list[list[int]]:
    """Cluster state vectors using Ward's method.

    Returns list of index lists (one per cluster), pruned by min_branch_prob.
    """
    n = len(state_vectors)
    if n <= 1:
        return [list(range(n))]

    if n == 2:
        return [[0], [1]]

    # Ward's hierarchical clustering
    distances = pdist(state_vectors.reshape(-1, 1), metric="euclidean")
    Z = linkage(distances, method="ward")
    n_clusters = min(max_branches, n)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Group indices by cluster
    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    # Prune small clusters
    result = []
    overflow = []
    for indices in clusters.values():
        if len(indices) / n >= min_branch_prob:
            result.append(indices)
        else:
            overflow.extend(indices)

    # Merge overflow into nearest cluster
    if overflow and result:
        for idx in overflow:
            # Find nearest cluster by centroid distance
            val = state_vectors[idx]
            best_dist = float("inf")
            best_cluster = 0
            for ci, cluster_indices in enumerate(result):
                centroid = np.mean(state_vectors[cluster_indices])
                dist = abs(val - centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = ci
            result[best_cluster].append(idx)
    elif overflow:
        result.append(overflow)

    return result


def build_scenario_tree(
    all_score_logs: list[list[list[float]]],
    max_depth: int = 3,
    max_branches: int = 4,
    min_branch_prob: float = 0.05,
    branching_steps: list[int] | None = None,
) -> ScenarioNode:
    """
    Build a scenario probability tree from ensemble trajectory data.

    At each branching step, extracts per-run mean scores, clusters them
    using Ward's hierarchical clustering, and recurses on each cluster.

    Args:
        all_score_logs: Per-run, per-agent, per-tick score logs.
            Shape conceptually: (n_runs, n_agents, n_ticks).
        max_depth: Maximum tree depth (number of branching levels).
        max_branches: Maximum children per node.
        min_branch_prob: Minimum probability for a branch (smaller merged).
        branching_steps: Explicit tick indices to branch at.
            Default: evenly spaced at 25%, 50%, 75% of total ticks.

    Returns:
        Root ScenarioNode of the probability tree.
    """
    n_runs = len(all_score_logs)
    if n_runs == 0:
        return ScenarioNode(id="root", step=0, probability=1.0)

    # Determine tick count from first run's first agent
    n_ticks = len(all_score_logs[0][0]) if all_score_logs[0] else 0
    if n_ticks == 0:
        return ScenarioNode(id="root", step=0, probability=1.0, n_supporting_runs=n_runs)

    # Default branching steps: evenly spaced
    if branching_steps is None:
        n_branch = min(max_depth, n_ticks)
        branching_steps = [
            int(n_ticks * (i + 1) / (n_branch + 1))
            for i in range(n_branch)
        ]
        branching_steps = [s for s in branching_steps if 0 < s < n_ticks]

    all_indices = list(range(n_runs))

    def _build(node_id: str, run_indices: list[int], depth: int) -> ScenarioNode:
        n = len(run_indices)
        step = branching_steps[depth] if depth < len(branching_steps) else n_ticks - 1

        # Compute state vectors at this step
        state_vectors = _extract_state_vectors(all_score_logs, step, run_indices)
        centroid = np.array([float(np.mean(state_vectors))]) if len(state_vectors) > 0 else None
        metrics = _compute_metrics(state_vectors)

        node = ScenarioNode(
            id=node_id,
            step=step,
            probability=n / n_runs,
            metrics=metrics,
            n_supporting_runs=n,
            centroid=centroid,
            run_indices=run_indices,
        )

        # Stop recursion
        if depth >= len(branching_steps) - 1 or n <= 1:
            return node

        # Cluster and recurse
        clusters = _cluster_at_step(state_vectors, max_branches, min_branch_prob)
        if len(clusters) <= 1:
            return node  # no meaningful split

        for ci, local_indices in enumerate(clusters):
            # Map local indices back to global run indices
            global_indices = [run_indices[li] for li in local_indices]
            child = _build(f"{node_id}.{ci + 1}", global_indices, depth + 1)
            node.children.append(child)

        return node

    root = _build("root", all_indices, 0)
    return root


def reduce_tree(
    root: ScenarioNode,
    target_scenarios: int,
) -> ScenarioNode:
    """
    Dupacova backward reduction: iteratively remove least-important leaves
    until target_scenarios remain. Redistributes probability to nearest neighbor.

    Uses Wasserstein distance between leaf centroid distributions.
    """
    # Collect all leaves
    leaves: list[ScenarioNode] = []

    def _collect_leaves(node: ScenarioNode):
        if node.is_leaf():
            leaves.append(node)
        else:
            for c in node.children:
                _collect_leaves(c)

    _collect_leaves(root)

    if len(leaves) <= target_scenarios:
        return root  # already at or below target

    while len(leaves) > target_scenarios:
        # Find the leaf whose removal costs least
        # Cost = probability × min_distance_to_other_leaf
        best_remove = None
        best_cost = float("inf")
        best_merge_into = None

        for i, leaf_i in enumerate(leaves):
            ci = leaf_i.centroid[0] if leaf_i.centroid is not None else 0.0
            min_dist = float("inf")
            merge_idx = None
            for j, leaf_j in enumerate(leaves):
                if i == j:
                    continue
                cj = leaf_j.centroid[0] if leaf_j.centroid is not None else 0.0
                dist = abs(ci - cj)
                if dist < min_dist:
                    min_dist = dist
                    merge_idx = j

            cost = leaf_i.probability * min_dist
            if cost < best_cost:
                best_cost = cost
                best_remove = i
                best_merge_into = merge_idx

        if best_remove is not None and best_merge_into is not None:
            removed = leaves[best_remove]
            target = leaves[best_merge_into]
            # Redistribute probability
            target.probability += removed.probability
            target.n_supporting_runs += removed.n_supporting_runs
            target.run_indices.extend(removed.run_indices)
            # Remove from parent
            _remove_leaf(root, removed.id)
            leaves.pop(best_remove)
        else:
            break

    return root


def _remove_leaf(node: ScenarioNode, leaf_id: str) -> bool:
    """Remove a leaf by id from the tree. Returns True if found and removed."""
    for i, child in enumerate(node.children):
        if child.id == leaf_id and child.is_leaf():
            node.children.pop(i)
            return True
        if _remove_leaf(child, leaf_id):
            # If child now has no children and was a branch, collapse it
            return True
    return False


def tree_to_dict(node: ScenarioNode) -> dict:
    """Serialize a scenario tree to a JSON-compatible dict."""
    d = {
        "id": node.id,
        "step": node.step,
        "probability": round(node.probability, 6),
        "metrics": node.metrics,
        "n_supporting_runs": node.n_supporting_runs,
        "n_children": len(node.children),
    }
    if node.centroid is not None:
        d["centroid"] = [round(float(v), 6) for v in node.centroid]
    if node.children:
        d["children"] = [tree_to_dict(c) for c in node.children]
    return d


def tree_to_flat_scenarios(node: ScenarioNode) -> list[dict]:
    """Extract leaf-to-root paths as flat scenario descriptions."""
    scenarios = []

    def _walk(n: ScenarioNode, path: list[dict]):
        entry = {"id": n.id, "step": n.step, "probability": n.probability, "metrics": n.metrics}
        current_path = path + [entry]
        if n.is_leaf():
            scenarios.append({
                "leaf_id": n.id,
                "probability": n.probability,
                "n_runs": n.n_supporting_runs,
                "path": current_path,
                "final_metrics": n.metrics,
            })
        else:
            for c in n.children:
                _walk(c, current_path)

    _walk(node, [])
    return scenarios
