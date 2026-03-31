"""
Adaptive GPU Load Balancer — feedback-driven work distribution.

Reads real-time GPU telemetry from GPUMonitor and adjusts:
  1. Participant split ratio (how many agents on GPU 0 vs GPU 1)
  2. Phase scheduling (which GPU handles environment/observers)
  3. Token budget per GPU (max_tokens allocation)

Uses a proportional controller (not full PID — derivative/integral
are noisy with LLM workloads where latency varies per generation).

The balancer runs between ticks: sample → compute error → adjust split.
Convergence target: both GPUs at target_power_w (default 195W) ±5%.

Timeline:
  tick N done → sample GPUs → compute imbalance → adjust split → tick N+1 starts
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BalancerState:
    """Current balancer decision state."""
    # How many participants on authority (GPU 0) vs swarm (GPU 1)
    auth_count: int = 0
    swarm_count: int = 0
    # Which GPU handles environment stimulus
    env_backend: str = "swarm"
    # Imbalance history for trend detection
    imbalance_history: list[float] = field(default_factory=list)
    # Adjustments made
    adjustments: int = 0


class AdaptiveBalancer:
    """
    Proportional load balancer driven by GPU power telemetry.

    Every tick:
      1. GPUMonitor.sample() reads power draw on both GPUs
      2. Compute imbalance = (gpu0_power - gpu1_power) / target
      3. If |imbalance| > deadband → shift one participant

    The deadband prevents oscillation (hunting) around the target.
    """

    def __init__(
        self,
        monitor,  # GPUMonitor instance
        n_participants: int,
        *,
        target_power_w: float = 195.0,
        deadband: float = 0.10,       # 10% deadband before adjustment
        min_per_gpu: int = 1,          # never leave a GPU completely idle
        adjust_interval: int = 1,      # rebalance every N ticks
    ):
        self.monitor = monitor
        self.n_participants = n_participants
        self.target_power_w = target_power_w
        self.deadband = deadband
        self.min_per_gpu = min_per_gpu
        self.adjust_interval = adjust_interval

        # Start with even split (or close to it)
        half = n_participants // 2
        self.state = BalancerState(
            auth_count=half,
            swarm_count=n_participants - half,
            env_backend="swarm",
        )

        # Per-participant backend assignment (mutable between ticks)
        self._assignments: list[str] = []
        self._recompute_assignments()

        logger.info(
            "AdaptiveBalancer: %d participants, split %d/%d (auth/swarm), "
            "target %.0fW, deadband %.0f%%",
            n_participants, self.state.auth_count, self.state.swarm_count,
            target_power_w, deadband * 100,
        )

    def _recompute_assignments(self):
        """Rebuild the assignment list from current split counts."""
        self._assignments = (
            ["authority"] * self.state.auth_count +
            ["swarm"] * self.state.swarm_count
        )

    def get_backend(self, participant_idx: int) -> str:
        """Get the current GPU assignment for a participant by index."""
        if participant_idx < len(self._assignments):
            return self._assignments[participant_idx]
        return "swarm"  # fallback

    def rebalance(self, tick: int) -> bool:
        """
        Sample GPUs and adjust split if needed.
        Call this BETWEEN ticks (after tick N completes, before N+1 starts).

        Returns True if an adjustment was made.
        """
        if tick % self.adjust_interval != 0:
            return False

        # Sample current state
        snap = self.monitor.sample()
        if not snap:
            return False

        imbalance = self.monitor.imbalance()
        self.state.imbalance_history.append(imbalance)

        # Log telemetry
        gpu_ids = sorted(snap.keys())
        if len(gpu_ids) >= 2:
            s0, s1 = snap[gpu_ids[0]], snap[gpu_ids[1]]
            logger.info(
                "[Tick %d] GPU balance: %.0fW / %.0fW (imbalance=%.2f, split=%d/%d)",
                tick, s0.power_w, s1.power_w, imbalance,
                self.state.auth_count, self.state.swarm_count,
            )

        # Check deadband
        if abs(imbalance) < self.deadband:
            logger.debug("[Tick %d] Within deadband (%.2f < %.2f) — no adjustment",
                         tick, abs(imbalance), self.deadband)
            return False

        # Proportional adjustment: shift ONE participant per tick
        adjusted = False
        if imbalance > 0:
            # GPU 0 (authority) is hotter → move one participant FROM authority TO swarm
            if self.state.auth_count > self.min_per_gpu:
                self.state.auth_count -= 1
                self.state.swarm_count += 1
                adjusted = True
                logger.info("[Tick %d] Shifted 1 participant: authority→swarm (now %d/%d)",
                            tick, self.state.auth_count, self.state.swarm_count)
        else:
            # GPU 1 (swarm) is hotter → move one participant FROM swarm TO authority
            if self.state.swarm_count > self.min_per_gpu:
                self.state.swarm_count -= 1
                self.state.auth_count += 1
                adjusted = True
                logger.info("[Tick %d] Shifted 1 participant: swarm→authority (now %d/%d)",
                            tick, self.state.auth_count, self.state.swarm_count)

        if adjusted:
            self._recompute_assignments()
            self.state.adjustments += 1

        return adjusted

    def apply_to_participants(self, participants: list):
        """
        Hot-swap backend assignments on participant agents.
        Called after rebalance() to take effect on the next tick.
        """
        for idx, p in enumerate(participants):
            new_backend = self.get_backend(idx)
            old_backend = p.cfg.get("backend", "swarm")
            if old_backend != new_backend:
                p.cfg["backend"] = new_backend
                logger.debug("Participant %s: %s → %s", p.agent_id, old_backend, new_backend)

    def should_move_env(self, tick: int) -> str | None:
        """
        Suggest moving environment to the cooler GPU if sustained imbalance.
        Returns new backend string or None if no change needed.
        Only triggers after 3+ ticks of consistent direction.
        """
        recent = self.state.imbalance_history[-3:]
        if len(recent) < 3:
            return None

        # All recent imbalances pointing same direction and outside deadband
        if all(r > self.deadband for r in recent):
            # Authority consistently hotter — move env to swarm (already default)
            if self.state.env_backend != "swarm":
                self.state.env_backend = "swarm"
                logger.info("[Tick %d] Moving environment to swarm (authority overloaded)", tick)
                return "swarm"
        elif all(r < -self.deadband for r in recent):
            # Swarm consistently hotter — move env to authority
            if self.state.env_backend != "authority":
                self.state.env_backend = "authority"
                logger.info("[Tick %d] Moving environment to authority (swarm overloaded)", tick)
                return "authority"

        return None

    def tick_report(self) -> dict:
        """Export current state for event stream / DuckDB."""
        stats = self.monitor.stats
        gpu_ids = sorted(stats.keys())
        return {
            "split": f"{self.state.auth_count}/{self.state.swarm_count}",
            "env_backend": self.state.env_backend,
            "adjustments": self.state.adjustments,
            "imbalance": round(self.state.imbalance_history[-1], 3) if self.state.imbalance_history else 0,
            **{
                f"gpu{gid}_ema_power_w": round(stats[gid].ema_power, 1)
                for gid in gpu_ids if gid in stats
            },
            **{
                f"gpu{gid}_ema_util_pct": round(stats[gid].ema_util, 1)
                for gid in gpu_ids if gid in stats
            },
        }
