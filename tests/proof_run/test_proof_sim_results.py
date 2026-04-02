"""
Proof-run test: Simulation Results & Export endpoints.

Verifies /simulation/status, /simulation/results, and /v1/memories
respond correctly regardless of whether a simulation has completed.
"""

import json
import ssl
import sys
import urllib.request

BASE = "https://48acx0kqem74jt-9000.proxy.runpod.net"
CTX = ssl._create_unverified_context()
TIMEOUT = 30

failures = 0
def fetch(path):
    """GET a JSON endpoint, return (status_code, parsed_body | None)."""
    url = f"{BASE}{path}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dualmirakl-proof-run"})
        resp = urllib.request.urlopen(req, context=CTX, timeout=TIMEOUT)
        body = json.loads(resp.read().decode())
        return resp.status, body
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode())
        except Exception:
            body = None
        return exc.code, body
    except Exception as exc:
        print(f"  [ERROR] {url}: {exc}")
        return None, None
def check(name, passed, detail=""):
    global failures
    tag = "[PASS]" if passed else "[FAIL]"
    msg = f"{tag} {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    if not passed:
        failures += 1
# ── 1. /simulation/status ────────────────────────────────────────────────────

print("=== /simulation/status ===")
code, body = fetch("/simulation/status")

check("status: reachable", code is not None, f"HTTP {code}")
check("status: valid JSON", body is not None)
check(
    "status: has 'status' field",
    isinstance(body, dict) and "status" in body,
    f"status={body.get('status')!r}" if isinstance(body, dict) else "",
)

VALID_STATUSES = {"idle", "running", "completed", "error", "starting"}
if isinstance(body, dict) and "status" in body:
    check(
        "status: recognised value",
        body["status"] in VALID_STATUSES,
        f"got {body['status']!r}",
    )
    sim_status = body["status"]
else:
    sim_status = None

# ── 2. /simulation/results ───────────────────────────────────────────────────

print("\n=== /simulation/results ===")
code_r, body_r = fetch("/simulation/results")

check("results: reachable", code_r is not None, f"HTTP {code_r}")
check("results: not a 500", code_r != 500 if code_r is not None else False)
check("results: valid JSON", body_r is not None)

if isinstance(body_r, dict):
    if sim_status == "completed":
        # Simulation finished -- expect real data
        has_data_key = any(
            k in body_r
            for k in ("config", "run_id", "trajectories", "observations")
        )
        check(
            "results: has data keys (sim completed)",
            has_data_key,
            f"keys={list(body_r.keys())[:8]}",
        )
        if "trajectories" in body_r:
            check(
                "results: trajectories is list/dict",
                isinstance(body_r["trajectories"], (list, dict)),
            )
    else:
        # No completed sim -- expect a clean error/empty envelope
        check(
            "results: clean no-data response",
            "error" in body_r or "status" in body_r or body_r == {},
            f"keys={list(body_r.keys())}",
        )

# ── 3. /v1/memories ──────────────────────────────────────────────────────────

print("\n=== /v1/memories ===")
code_m, body_m = fetch("/v1/memories")

check("memories: reachable", code_m is not None, f"HTTP {code_m}")
check("memories: not a 500", code_m != 500 if code_m is not None else False)
check("memories: valid JSON", body_m is not None)

if isinstance(body_m, dict):
    # Either has stats/runs (normal) or an error key (DuckDB not initialised)
    check(
        "memories: well-formed response",
        "stats" in body_m or "runs" in body_m or "error" in body_m,
        f"keys={list(body_m.keys())}",
    )

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"Failures: {failures}")
sys.exit(1 if failures else 0)
