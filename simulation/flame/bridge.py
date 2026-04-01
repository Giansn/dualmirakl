"""
Bridge between dualmirakl WorldState/participants and FLAME GPU 2 agent populations.

Responsibilities:
    - Map LLM participant scores → influencer agent variables
    - Map influencer positions in social space (score-based 2D embedding)
    - Extract population statistics from FLAME GPU 2 → dualmirakl WorldState
    - Serialize/deserialize FLAME population snapshots for data export
"""

import math
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class PopulationSnapshot:
    """Snapshot of FLAME GPU 2 population state at a given tick."""
    tick: int
    n_population: int
    n_influencers: int
    sub_steps: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    influencer_scores: list[float] = field(default_factory=list)
    # Score distribution (10 bins from 0 to 1)
    histogram: list[int] = field(default_factory=list)
    # Spatial clustering metric (mean distance to nearest influencer)
    mean_influencer_distance: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class FlameBridge:
    """
    Bidirectional data bridge between dualmirakl sim_loop and FLAME GPU 2 engine.

    Usage:
        bridge = FlameBridge(n_influencers=4, space_size=100.0)

        # Each dualmirakl tick:
        bridge.push_influencer_scores([p.behavioral_score for p in participants])
        engine.step(sub_steps)
        snapshot = bridge.pull_population_stats(engine, tick)

        # After run:
        bridge.export_snapshots("data/run_id/flame_population.json")
    """

    def __init__(self, n_influencers: int = 4, space_size: float = 100.0):
        self.n_influencers = n_influencers
        self.space_size = space_size
        self.snapshots: list[PopulationSnapshot] = []
        self._influencer_positions = self._compute_influencer_positions()

    def _compute_influencer_positions(self) -> list[tuple[float, float]]:
        """
        Place influencers evenly around a circle in social space.
        This gives each influencer a distinct "zone of influence".
        """
        cx, cy = self.space_size / 2, self.space_size / 2
        r = self.space_size * 0.3  # 30% of space radius
        positions = []
        for i in range(self.n_influencers):
            angle = 2 * math.pi * i / self.n_influencers
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            positions.append((x, y))
        return positions

    def push_influencer_scores(self, engine, scores: list[float]):
        """
        Update influencer agent scores in the FLAME GPU 2 simulation.
        Called at the start of each dualmirakl tick (after Phase C).

        Args:
            engine: FlameEngine instance
            scores: list of behavioral_scores from LLM participants
        """
        if len(scores) != self.n_influencers:
            raise ValueError(
                f"Expected {self.n_influencers} scores, got {len(scores)}"
            )
        engine.set_influencer_scores(scores)

    def push_influencer_positions(self, engine):
        """
        Set influencer positions in social space (called once at init).
        """
        engine.set_influencer_positions(self._influencer_positions)

    def pull_population_stats(
        self, engine, tick: int, sub_steps: int
    ) -> PopulationSnapshot:
        """
        Extract population statistics after FLAME GPU 2 sub-steps.
        Called after engine.step() completes.

        Returns:
            PopulationSnapshot with aggregate statistics
        """
        stats = engine.get_population_stats()
        snapshot = PopulationSnapshot(
            tick=tick,
            n_population=stats["count"],
            n_influencers=self.n_influencers,
            sub_steps=sub_steps,
            mean_score=stats["mean_score"],
            std_score=stats["std_score"],
            min_score=stats["min_score"],
            max_score=stats["max_score"],
            influencer_scores=stats.get("influencer_scores", []),
            histogram=stats.get("histogram", []),
            mean_influencer_distance=stats.get("mean_influencer_distance", 0.0),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_population_coupling_feedback(self, engine) -> dict:
        """
        Extract population-level feedback for dualmirakl observer agents.
        Returns aggregate metrics that observers can use to inform interventions.

        Polarization is estimated from std/mean without iterating agents:
        std of uniform [0,1] = 0.289. Ratio > 1.0 indicates bimodal spread.

        Returns:
            dict with keys: mean_score, std_score, polarization, n_population
        """
        stats = engine.get_population_stats()
        std = stats["std_score"]
        mean = stats["mean_score"]

        # Polarization proxy: std normalized by uniform std (0.289)
        # Values > 1.0 suggest bimodal distribution, < 0.5 suggests consensus
        uniform_std = 0.289
        polarization = std / uniform_std if uniform_std > 0 else 0.0

        # Distance from center: how far the mean is from 0.5
        center_offset = abs(mean - 0.5)

        return {
            "mean_score": mean,
            "std_score": std,
            "polarization": round(polarization, 3),
            "center_offset": round(center_offset, 3),
            "n_population": stats["count"],
        }

    def export_snapshots(self, path: str):
        """Export all tick snapshots to JSON for post-hoc analysis."""
        data = {
            "n_influencers": self.n_influencers,
            "space_size": self.space_size,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def reset(self):
        """Clear snapshot history for a new run."""
        self.snapshots.clear()
