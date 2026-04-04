"""
GPU Monitor — real-time power/utilization telemetry via NVML.

Provides a lightweight async interface for GPUHarmony to read GPU state
and make adaptive load-balancing decisions every tick.

Falls back gracefully to no-op when pynvml is unavailable (e.g., CPU-only dev).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False
    logger.info("pynvml not installed — GPU monitor disabled (pip install pynvml)")


@dataclass(frozen=True)
class GPUSnapshot:
    """Point-in-time GPU telemetry for one device."""
    gpu_id: int
    power_w: float          # current draw in watts
    power_limit_w: float    # TDP / power limit in watts
    util_pct: float         # SM utilization 0-100
    mem_used_mb: float
    mem_total_mb: float
    temperature_c: int
    timestamp: float         # monotonic time


@dataclass
class GPUStats:
    """Rolling statistics for one GPU across a simulation run."""
    gpu_id: int
    snapshots: list[GPUSnapshot] = field(default_factory=list)
    _ema_power: float = 0.0
    _ema_util: float = 0.0
    _ema_alpha: float = 0.3  # fast-tracking EMA

    def update(self, snap: GPUSnapshot):
        self.snapshots.append(snap)
        if len(self.snapshots) == 1:
            self._ema_power = snap.power_w
            self._ema_util = snap.util_pct
        else:
            self._ema_power = self._ema_alpha * snap.power_w + (1 - self._ema_alpha) * self._ema_power
            self._ema_util = self._ema_alpha * snap.util_pct + (1 - self._ema_alpha) * self._ema_util

    @property
    def ema_power(self) -> float:
        return self._ema_power

    @property
    def ema_util(self) -> float:
        return self._ema_util

    @property
    def power_efficiency(self) -> float:
        """Ratio of current power to limit (0-1). 1.0 = fully loaded."""
        if not self.snapshots:
            return 0.0
        return self._ema_power / max(self.snapshots[-1].power_limit_w, 1.0)

    @property
    def idle_ratio(self) -> float:
        """Fraction of recent snapshots where util < 10%."""
        recent = self.snapshots[-10:]
        if not recent:
            return 1.0
        return sum(1 for s in recent if s.util_pct < 10) / len(recent)


class GPUMonitor:
    """
    Async-compatible GPU telemetry provider.

    Usage:
        monitor = GPUMonitor(gpu_ids=[0, 1])
        monitor.start()
        snap = monitor.sample()  # {0: GPUSnapshot, 1: GPUSnapshot}
        stats = monitor.stats    # {0: GPUStats, 1: GPUStats}
        monitor.stop()
    """

    def __init__(self, gpu_ids: list[int] | None = None, target_power_w: float = 195.0):
        self._gpu_ids = gpu_ids or [0, 1]
        self.target_power_w = target_power_w
        self._handles: dict[int, object] = {}
        self._stats: dict[int, GPUStats] = {gid: GPUStats(gpu_id=gid) for gid in self._gpu_ids}
        self._initialized = False

    def start(self) -> bool:
        """Initialize NVML and get device handles. Returns True if successful."""
        if not _NVML_AVAILABLE:
            logger.warning("GPU monitor: pynvml unavailable, running in dummy mode")
            return False
        try:
            pynvml.nvmlInit()
            for gid in self._gpu_ids:
                self._handles[gid] = pynvml.nvmlDeviceGetHandleByIndex(gid)
            self._initialized = True
            # Log initial state and capture min power limit for TDP-based targeting
            snap = self.sample()
            self.min_power_limit = min(
                (s.power_limit_w for s in snap.values()), default=300.0
            )
            for gid, s in snap.items():
                logger.info(
                    "GPU %d: %s, %.0fW / %.0fW limit, %dMB / %dMB, %d°C",
                    gid, f"{s.util_pct:.0f}% util", s.power_w, s.power_limit_w,
                    s.mem_used_mb, s.mem_total_mb, s.temperature_c,
                )
            return True
        except Exception as e:
            logger.warning("GPU monitor init failed: %s", e)
            self._initialized = False
            return False

    def stop(self):
        """Shutdown NVML."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False

    def sample(self) -> dict[int, GPUSnapshot]:
        """Take a point-in-time snapshot of all monitored GPUs."""
        if not self._initialized:
            return self._dummy_sample()

        now = time.monotonic()
        result = {}
        for gid, handle in self._handles.items():
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)           # milliwatts
                power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                snap = GPUSnapshot(
                    gpu_id=gid,
                    power_w=power_mw / 1000.0,
                    power_limit_w=power_limit_mw / 1000.0,
                    util_pct=float(util.gpu),
                    mem_used_mb=mem.used / (1024 * 1024),
                    mem_total_mb=mem.total / (1024 * 1024),
                    temperature_c=temp,
                    timestamp=now,
                )
                result[gid] = snap
                self._stats[gid].update(snap)
            except Exception as e:
                logger.debug("GPU %d sample failed: %s", gid, e)
        return result

    def _dummy_sample(self) -> dict[int, GPUSnapshot]:
        """Return neutral dummy snapshots when NVML is unavailable."""
        now = time.monotonic()
        result = {}
        for gid in self._gpu_ids:
            snap = GPUSnapshot(
                gpu_id=gid, power_w=0, power_limit_w=300,
                util_pct=50, mem_used_mb=0, mem_total_mb=32000,
                temperature_c=0, timestamp=now,
            )
            result[gid] = snap
            self._stats[gid].update(snap)
        return result

    @property
    def stats(self) -> dict[int, GPUStats]:
        return self._stats

    def imbalance(self, metric: str = "utilization") -> float:
        """
        Imbalance ratio between GPUs.

        Args:
            metric: "utilization" (default, 0-100% based) or "power" (wattage based)

        Returns: (gpu0 - gpu1) / normalizer.
        Positive = GPU 0 is busier. Negative = GPU 1 is busier.
        Range roughly [-1, 1].
        """
        if len(self._gpu_ids) < 2:
            return 0.0
        s0 = self._stats[self._gpu_ids[0]]
        s1 = self._stats[self._gpu_ids[1]]
        if metric == "utilization":
            # Normalize by 100% (max utilization)
            return (s0.ema_util - s1.ema_util) / 100.0
        else:
            return (s0.ema_power - s1.ema_power) / max(self.target_power_w, 1.0)

    def summary(self) -> dict:
        """Human-readable summary for logging/export."""
        snap = self.sample()
        return {
            f"gpu{gid}": {
                "power_w": round(s.power_w, 1),
                "power_limit_w": round(s.power_limit_w, 1),
                "util_pct": round(s.util_pct, 1),
                "ema_power_w": round(self._stats[gid].ema_power, 1),
                "ema_util_pct": round(self._stats[gid].ema_util, 1),
                "efficiency": round(self._stats[gid].power_efficiency, 3),
                "idle_ratio": round(self._stats[gid].idle_ratio, 3),
            }
            for gid, s in snap.items()
        }
