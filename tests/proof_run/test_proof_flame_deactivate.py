#!/usr/bin/env python3
"""Proof-run: FLAME deactivation and cleanup on a live RunPod pod.

Verifies that /simulation/flame/deactivate is safe to call when FLAME is
inactive (pyflamegpu not installed), returns a clean state, and is idempotent.
Also checks that GPU 2 memory stays low after deactivation.
"""

import json
import ssl
import sys
import urllib.request

BASE_URL = "https://48acx0kqem74jt-9000.proxy.runpod.net"
CTX = ssl._create_unverified_context()

def request(method, path, body=None):
    """Send an HTTP request and return (status, parsed_json)."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json", "User-Agent": "dualmirakl-proof-run"} if data else {"User-Agent": "dualmirakl-proof-run"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_bytes = e.read() if e.fp else b""
        try:
            return e.code, json.loads(body_bytes.decode())
        except Exception:
            return e.code, {"raw": body_bytes.decode()}
    except Exception as e:
        return 0, {"error": str(e)}


def main():
    failures = 0

    def check(label, condition, detail=""):
        nonlocal failures
        if condition:
            print(f"[PASS] {label}")
        else:
            print(f"[FAIL] {label}  {detail}")
            failures += 1

    # ── 1. Initial FLAME status ─────────────────────────────────────────────────
    status, data = request("GET", "/simulation/flame")
    check("GET /simulation/flame returns 200", status == 200, f"status={status}")
    check("Initial FLAME is not active", data.get("active") is False,
          f"active={data.get('active')}")

    # ── 2. Deactivate (first call) ──────────────────────────────────────────────
    status, deact1 = request("POST", "/simulation/flame/deactivate")
    check("POST /simulation/flame/deactivate returns 200", status == 200,
          f"status={status}")
    check("Deactivate response has active==false", deact1.get("active") is False,
          f"active={deact1.get('active')}")
    check("Deactivate is a no-op (no error key in response)",
          "error" not in deact1,
          f"unexpected error key: {deact1.get('error')}")

    # ── 3. Verify FLAME still inactive after deactivate ─────────────────────────
    status, data = request("GET", "/simulation/flame")
    check("GET /simulation/flame still 200 after deactivate", status == 200,
          f"status={status}")
    check("FLAME still inactive after deactivate", data.get("active") is False,
          f"active={data.get('active')}")

    # ── 4. GPU 2 memory check ──────────────────────────────────────────────────
    status, data = request("GET", "/gpu/telemetry")
    check("GET /gpu/telemetry returns 200", status == 200, f"status={status}")

    gpu2 = None
    for gpu in data.get("gpus", []):
        if gpu.get("index") == 2:
            gpu2 = gpu
            break

    if gpu2 is not None:
        mem_gb = gpu2["memory_used"] / 1024  # MiB -> GiB
        check(f"GPU 2 memory_used < 5 GB ({mem_gb:.1f} GB)", mem_gb < 5,
              f"memory_used={gpu2['memory_used']} MiB")
    else:
        # Pod may have only 2 GPUs -- not a failure, just informational
        print("[SKIP] GPU 2 not present in telemetry (pod may have <3 GPUs)")

    # ── 5. Idempotency: deactivate again ───────────────────────────────────────
    status2, deact2 = request("POST", "/simulation/flame/deactivate")
    check("Idempotent deactivate returns 200", status2 == 200, f"status={status2}")
    check("Idempotent deactivate has active==false", deact2.get("active") is False,
          f"active={deact2.get('active')}")
    check("Idempotent deactivate matches first response",
          deact2.get("active") == deact1.get("active"),
          f"first={deact1.get('active')} second={deact2.get('active')}")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FLAME deactivation proof-run: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
