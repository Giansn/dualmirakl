"""Proof-run: GPU Telemetry & System

Verifies the live /gpu/telemetry endpoint returns correct GPU and CPU
metrics for the 3x RTX PRO 4500 Blackwell pod on RunPod.
"""

import json
import ssl
import sys
import time
import urllib.request

BASE = "https://48acx0kqem74jt-9000.proxy.runpod.net"
TIMEOUT = 10


def _get(path):
    """GET a path, return (status, body_bytes, elapsed_seconds)."""
    ctx = ssl._create_unverified_context()
    url = f"{BASE}{path}"
    req = urllib.request.Request(url)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as resp:
            body = resp.read()
            elapsed = time.monotonic() - t0
            return resp.status, body, elapsed
    except urllib.error.HTTPError as exc:
        elapsed = time.monotonic() - t0
        return exc.code, exc.read(), elapsed


def _report(label, ok, detail=""):
    tag = "[PASS]" if ok else "[FAIL]"
    msg = f"{tag} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return ok


def main():
    passed = 0
    failed = 0

    def check(label, ok, detail=""):
        nonlocal passed, failed
        if _report(label, ok, detail):
            passed += 1
        else:
            failed += 1

    # -- 1. Fetch /gpu/telemetry -------------------------------------------------
    status, body, elapsed = _get("/gpu/telemetry")
    check("GET /gpu/telemetry returns 200", status == 200, f"status={status}")

    data = {}
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        pass

    gpus = data.get("gpus", [])
    check("Response has 'gpus' array", isinstance(gpus, list), f"type={type(gpus).__name__}")
    check("Exactly 3 GPUs reported", len(gpus) == 3, f"count={len(gpus)}")

    # -- 2-6. Per-GPU checks (single pass) ----------------------------------------
    required_fields = ["index", "name", "utilization", "memory_used", "memory_total", "temp", "power_draw"]
    for gpu in gpus:
        idx = gpu.get("index", "?")
        name = gpu.get("name", "")
        name_lower = name.lower()
        mem_total = gpu.get("memory_total", 0)
        mem_used = gpu.get("memory_used", 0)

        missing = [f for f in required_fields if f not in gpu]
        check(f"GPU {idx} has all required fields", len(missing) == 0, f"missing={missing}")

        name_ok = "blackwell" in name_lower or "4500" in name_lower
        check(f"GPU {idx} name matches Blackwell/RTX PRO 4500", name_ok, f"name={name!r}")

        check(f"GPU {idx} memory_total ~32GB", 31000 <= mem_total <= 33500, f"memory_total={mem_total} MB")

        if idx in (0, 1):
            check(f"GPU {idx} memory_used > 20GB (vLLM loaded)", mem_used > 20000, f"memory_used={mem_used} MB")
        elif idx == 2:
            check(f"GPU 2 memory_used < 5GB (FLAME not active)", mem_used < 5000, f"memory_used={mem_used} MB")

    # -- 7. CPU metrics ----------------------------------------------------------
    cpu = data.get("cpu")
    check("CPU metrics present", cpu is not None, f"cpu={cpu!r}")
    if cpu is not None:
        check("cpu.percent exists", "percent" in cpu, f"keys={list(cpu.keys())}")
        check("cpu.memory_used exists", "memory_used" in cpu, f"keys={list(cpu.keys())}")
        check("cpu.memory_total exists", "memory_total" in cpu, f"keys={list(cpu.keys())}")

    # -- 8. Timestamp is recent (within 60 seconds) ------------------------------
    ts = data.get("ts", 0)
    now = time.time()
    age = abs(now - ts)
    check("Timestamp 'ts' is recent (within 60s)", age < 60, f"age={age:.1f}s")

    # -- Summary -----------------------------------------------------------------
    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"  {passed}/{total} checks passed, {failed} failed")
    print(f"{'=' * 50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
