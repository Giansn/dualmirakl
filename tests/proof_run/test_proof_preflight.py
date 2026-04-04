#!/usr/bin/env python3
"""Proof-run: preflight and system readiness checks against a live RunPod pod."""

import json
import os
import ssl
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

BASE_URL = os.environ.get(
    "PROOF_BASE_URL",
    "https://48acx0kqem74jt-9000.proxy.runpod.net",
)
TIMEOUT = 30

_ctx = ssl._create_unverified_context()

def _get(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "dualmirakl-proof-run"})
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=_ctx) as resp:
        return json.loads(resp.read().decode())


def main():
    failed = 0
    passed = 0

    def check(label: str, ok: bool, detail: str = ""):
        nonlocal failed, passed
        tag = "[PASS]" if ok else "[FAIL]"
        msg = f"{tag} {label}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        if ok:
            passed += 1
        else:
            failed += 1

    # ── 1. Preflight ────────────────────────────────────────────────────────────

    print("=== /simulation/preflight ===")
    preflight = _get("/simulation/preflight")

    check("preflight ready", preflight.get("ready") is True, f"ready={preflight.get('ready')}")

    checks_list = preflight.get("checks", [])
    checks_by_name = {c["name"]: c for c in checks_list}

    # All checks must be ok or info (no fail)
    for c in checks_list:
        ok = c["status"] in ("ok", "info", "warn")
        check(f"preflight check '{c['name']}' not fail", ok, f"status={c['status']}")

    # Required check names must be present
    REQUIRED_CHECKS = ["vllm_authority", "vllm_swarm", "embedding_model", "output_dir", "world_context"]
    for name in REQUIRED_CHECKS:
        check(f"preflight has '{name}'", name in checks_by_name)

    # ── 2. System status ────────────────────────────────────────────────────────

    print("\n=== /system/status ===")
    status = _get("/system/status")

    check("system ready", status.get("ready") is True, f"ready={status.get('ready')}")

    services = status.get("services", {})
    for svc in ("authority", "swarm", "gateway"):
        svc_status = services.get(svc, {}).get("status")
        check(f"service '{svc}' up", svc_status == "up", f"status={svc_status}")

    check("setup_done", status.get("setup_done") is True, f"setup_done={status.get('setup_done')}")

    gpus = status.get("gpus", [])
    check("3 GPUs detected", len(gpus) == 3, f"count={len(gpus)}")

    # ── Summary ─────────────────────────────────────────────────────────────────

    print(f"\n{'=' * 40}")
    print(f"Passed: {passed}  Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
