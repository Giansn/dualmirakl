"""Tests for GPU monitor + adaptive balancer (no GPU required)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from unittest.mock import MagicMock, patch
from simulation.gpu_monitor import GPUSnapshot, GPUStats, GPUMonitor
from simulation.adaptive_balancer import AdaptiveBalancer, BalancerState


# ── GPUSnapshot / GPUStats ──────────────────────────────────────────

def test_gpu_stats_ema():
    stats = GPUStats(gpu_id=0)
    now = time.monotonic()
    for power in [100, 150, 180, 195, 200]:
        snap = GPUSnapshot(
            gpu_id=0, power_w=power, power_limit_w=300,
            util_pct=80, mem_used_mb=16000, mem_total_mb=32000,
            temperature_c=65, timestamp=now,
        )
        stats.update(snap)
    # EMA should track upward
    assert stats.ema_power > 150
    assert stats.ema_power < 200
    assert stats.power_efficiency > 0


def test_gpu_stats_idle_ratio():
    stats = GPUStats(gpu_id=0)
    now = time.monotonic()
    # 8 idle + 2 active = 80% idle
    for i in range(10):
        util = 5 if i < 8 else 90
        snap = GPUSnapshot(
            gpu_id=0, power_w=100, power_limit_w=300,
            util_pct=util, mem_used_mb=16000, mem_total_mb=32000,
            temperature_c=65, timestamp=now,
        )
        stats.update(snap)
    assert stats.idle_ratio == 0.8


# ── GPUMonitor (dummy mode) ────────────────────────────────────────

def test_monitor_dummy_mode():
    """Without pynvml, monitor should return neutral snapshots."""
    monitor = GPUMonitor(gpu_ids=[0, 1])
    # Don't call start() — force dummy mode
    snap = monitor.sample()
    assert 0 in snap
    assert 1 in snap
    assert snap[0].util_pct == 50  # neutral dummy value


def test_monitor_imbalance_neutral():
    monitor = GPUMonitor(gpu_ids=[0, 1])
    monitor.sample()  # populate stats with identical dummies
    assert monitor.imbalance() == 0.0


# ── AdaptiveBalancer ───────────────────────────────────────────────

def _make_monitor_with_power(gpu0_power: float, gpu1_power: float):
    """Create a mock monitor that returns specific power values."""
    monitor = MagicMock()
    monitor.target_power_w = 195.0

    snap0 = GPUSnapshot(
        gpu_id=0, power_w=gpu0_power, power_limit_w=300,
        util_pct=80, mem_used_mb=16000, mem_total_mb=32000,
        temperature_c=65, timestamp=time.monotonic(),
    )
    snap1 = GPUSnapshot(
        gpu_id=1, power_w=gpu1_power, power_limit_w=300,
        util_pct=80, mem_used_mb=16000, mem_total_mb=32000,
        temperature_c=65, timestamp=time.monotonic(),
    )
    monitor.sample.return_value = {0: snap0, 1: snap1}

    stats0 = GPUStats(gpu_id=0)
    stats0.update(snap0)
    stats1 = GPUStats(gpu_id=1)
    stats1.update(snap1)
    monitor.stats = {0: stats0, 1: stats1}

    def imbalance():
        return (stats0.ema_power - stats1.ema_power) / monitor.target_power_w
    monitor.imbalance = imbalance

    return monitor


def test_balancer_even_split():
    monitor = _make_monitor_with_power(195, 195)
    balancer = AdaptiveBalancer(monitor, n_participants=4)
    assert balancer.state.auth_count == 2
    assert balancer.state.swarm_count == 2


def test_balancer_no_adjust_in_deadband():
    """Balanced GPUs should not trigger rebalance."""
    monitor = _make_monitor_with_power(195, 190)
    balancer = AdaptiveBalancer(monitor, n_participants=4, deadband=0.10)
    adjusted = balancer.rebalance(tick=1)
    assert not adjusted
    assert balancer.state.adjustments == 0


def test_balancer_shifts_on_imbalance():
    """GPU 0 much hotter → should shift one participant to swarm."""
    monitor = _make_monitor_with_power(250, 140)
    balancer = AdaptiveBalancer(monitor, n_participants=4, deadband=0.10)
    adjusted = balancer.rebalance(tick=1)
    assert adjusted
    assert balancer.state.auth_count == 1
    assert balancer.state.swarm_count == 3


def test_balancer_respects_min_per_gpu():
    """Should not shift below min_per_gpu."""
    monitor = _make_monitor_with_power(250, 140)
    balancer = AdaptiveBalancer(monitor, n_participants=2, deadband=0.10, min_per_gpu=1)
    # Already at 1/1 split
    adjusted = balancer.rebalance(tick=1)
    assert not adjusted


def test_balancer_apply_to_participants():
    """Should hot-swap backend on participant agents."""
    monitor = _make_monitor_with_power(250, 140)
    balancer = AdaptiveBalancer(monitor, n_participants=4, deadband=0.10)

    # Mock participants
    participants = []
    for i in range(4):
        p = MagicMock()
        p.agent_id = f"p{i}"
        p.cfg = {"backend": "authority" if i < 2 else "swarm"}
        participants.append(p)

    # Trigger rebalance (gpu0 hot → shift one to swarm)
    balancer.rebalance(tick=1)
    balancer.apply_to_participants(participants)

    auth_count = sum(1 for p in participants if p.cfg["backend"] == "authority")
    swarm_count = sum(1 for p in participants if p.cfg["backend"] == "swarm")
    assert auth_count == 1
    assert swarm_count == 3


def test_balancer_env_move_after_sustained():
    """Environment should move after 3 ticks of sustained imbalance."""
    monitor = _make_monitor_with_power(250, 140)
    balancer = AdaptiveBalancer(monitor, n_participants=6, deadband=0.10)

    # Simulate 3 ticks of GPU 0 being hotter
    for tick in range(1, 4):
        balancer.rebalance(tick)

    result = balancer.should_move_env(tick=3)
    # env already on swarm (default), and authority is hotter → no change needed
    assert result is None

    # Now simulate swarm being hotter
    monitor2 = _make_monitor_with_power(140, 250)
    balancer2 = AdaptiveBalancer(monitor2, n_participants=6, deadband=0.10)
    for tick in range(1, 4):
        balancer2.rebalance(tick)
    result2 = balancer2.should_move_env(tick=3)
    assert result2 == "authority"


def test_balancer_tick_report():
    monitor = _make_monitor_with_power(195, 190)
    balancer = AdaptiveBalancer(monitor, n_participants=4)
    balancer.rebalance(tick=1)
    report = balancer.tick_report()
    assert "split" in report
    assert "imbalance" in report
    assert report["split"] == "2/2"
