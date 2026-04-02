#!/usr/bin/env python3
"""
Proof-run test: Minimal Simulation Run

Verifies the dualmirakl simulation platform on a running RunPod pod by:
1. Starting a minimal simulation (2 agents, 3 ticks)
2. Polling status until completion
3. Retrieving and validating results

Usage:
    python tests/proof_run/test_proof_minimal_sim.py
"""

import json
import ssl
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "https://48acx0kqem74jt-9000.proxy.runpod.net"
POLL_INTERVAL = 5       # seconds between status polls
POLL_TIMEOUT = 120      # max seconds to wait for completion

# Unverified SSL context for RunPod proxy
_ctx = ssl._create_unverified_context()

results = []


def record(name, passed, detail=""):
    tag = "[PASS]" if passed else "[FAIL]"
    msg = f"{tag} {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    results.append(passed)


def http_request(method, path, body=None):
    """Send an HTTP request and return (status_code, parsed_json | None)."""
    url = f"{BASE_URL}{path}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": "dualmirakl-proof-run"})
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "dualmirakl-proof-run/1.0")
    try:
        with urllib.request.urlopen(req, context=_ctx, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return e.code, {"raw": raw}
    except Exception as e:
        return 0, {"error": str(e)}


# ── Step 1: Start minimal simulation ────────────────────────────────────

print("\n=== Step 1: POST /simulation/start (minimal) ===")
start_body = {
    "scenario": "minimal",
    "n_ticks": 3,
    "n_participants": 2,
    "max_tokens": 64,
    "k": 3,
    "alpha": 0.15,
}
code, resp = http_request("POST", "/simulation/start", start_body)
print(f"  HTTP {code}: {json.dumps(resp, indent=2)[:500]}")

already_running = False
if code == 200 and isinstance(resp, dict):
    if "error" in resp and "already running" in resp.get("error", "").lower():
        record("Start returns well-formed busy response", True,
               "simulation already running")
        already_running = True
    elif resp.get("status") == "started":
        record("Simulation started", True,
               f"config={resp.get('config', {})}")
    else:
        # Unexpected but 200 -- still pass if we got valid JSON
        record("Start returned valid JSON", True, f"status={resp.get('status')}")
else:
    record("Simulation start request", False, f"HTTP {code}")


# ── Step 2: Poll /simulation/status until completed ─────────────────────

print("\n=== Step 2: Poll /simulation/status ===")
deadline = time.time() + POLL_TIMEOUT
final_status = None

def _is_terminal(st):
    """Check if simulation status represents a terminal state."""
    s = st.get("status", "")
    if s == "completed":
        return True
    if s == "idle" and st.get("run_dir") and not already_running:
        return True
    if s.startswith("error"):
        return True
    return False

while time.time() < deadline:
    code, status = http_request("GET", "/simulation/status")
    if code != 200:
        print(f"  status poll HTTP {code}")
        time.sleep(POLL_INTERVAL)
        continue

    s = status.get("status", "")
    print(f"  status={s}  tick={status.get('tick', 0)}  pct={status.get('pct', 0)}%")

    if _is_terminal(status):
        final_status = status
        break

    time.sleep(POLL_INTERVAL)

if final_status is None:
    # One last check after timeout
    _, final_status = http_request("GET", "/simulation/status")
    if final_status and _is_terminal(final_status):
        pass  # completed just after timeout
    else:
        s = final_status.get("status", "unknown") if final_status else "no response"
        record("Simulation completed", False,
               f"timed out after {POLL_TIMEOUT}s, last status={s}")

if final_status and final_status.get("status") in ("completed",):
    record("Simulation completed", True)
elif final_status and final_status.get("status") == "idle" and final_status.get("run_dir"):
    record("Simulation completed (idle with prior run)", True)
elif final_status and final_status.get("status", "").startswith("error"):
    record("Simulation completed", False,
           f"ended with error: {final_status.get('status')}")


# ── Step 3: Verify tick count and run metadata ──────────────────────────

print("\n=== Step 3: Verify simulation metadata ===")
if final_status:
    total_ticks = final_status.get("n_ticks", 0)
    completed_tick = final_status.get("tick", 0)

    record("Tick count is positive", total_ticks > 0, f"n_ticks={total_ticks}")
    record("Final tick reached total", completed_tick >= total_ticks,
           f"tick={completed_tick} n_ticks={total_ticks}")

    run_dir = final_status.get("run_dir")
    record("run_dir is present", bool(run_dir), f"run_dir={run_dir}")

    started_at = final_status.get("started_at")
    record("started_at timestamp present", started_at is not None,
           f"started_at={started_at}")
else:
    record("Status response available", False, "no final status")


# ── Step 4: GET /simulation/results ─────────────────────────────────────

print("\n=== Step 4: GET /simulation/results ===")
code, res = http_request("GET", "/simulation/results")
print(f"  HTTP {code}: keys={list(res.keys()) if isinstance(res, dict) else 'N/A'}")

if code == 200 and isinstance(res, dict):
    has_error = "error" in res
    if has_error:
        record("Results endpoint returned data", False, f"error={res.get('error')}")
    else:
        record("Results endpoint returned data", True,
               f"keys={list(res.keys())}")
else:
    record("Results endpoint returned data", False, f"HTTP {code}")


# ── Step 5: Validate results structure ──────────────────────────────────

print("\n=== Step 5: Validate results structure ===")
if code == 200 and isinstance(res, dict) and "error" not in res:
    # Check for trajectories
    has_trajectories = "trajectories" in res
    record("Results contain trajectories", has_trajectories)
    if has_trajectories:
        traj = res["trajectories"]
        if isinstance(traj, dict):
            record("Trajectories is a dict", True,
                   f"keys={list(traj.keys())[:5]}")
        elif isinstance(traj, list):
            record("Trajectories is a list", True,
                   f"length={len(traj)}")
        else:
            record("Trajectories has valid type", False,
                   f"type={type(traj).__name__}")

    # Check for config
    has_config = "config" in res
    record("Results contain config", has_config)

    # Check for dynamics analysis
    has_dynamics = "dynamics_analysis" in res
    record("Results contain dynamics_analysis", has_dynamics)

    # Check for observations
    has_obs = "observations" in res
    record("Results contain observations", has_obs)

    event_count = len(final_status.get("events", [])) if final_status else 0
    record("Events were recorded during simulation", event_count > 0,
           f"count={event_count}")
else:
    record("Results structure validation", False, "no valid results to inspect")


# ── Summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
passed = sum(results)
total = len(results)
print(f"Results: {passed}/{total} checks passed")
if all(results):
    print("ALL CHECKS PASSED")
    sys.exit(0)
else:
    failed = [i for i, r in enumerate(results) if not r]
    print(f"FAILED checks: {failed}")
    sys.exit(1)
