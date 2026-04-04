"""
Proof-run test: FLAME activation attempt (graceful degradation).

Verifies that POST /simulation/flame/activate fails gracefully when
pyflamegpu is not installed, returning an informative error without
crashing the server.
"""

import json
import ssl
import sys
import urllib.request
import urllib.error

BASE_URL = "https://48acx0kqem74jt-9000.proxy.runpod.net"
CTX = ssl._create_unverified_context()

def _request(method, path, body=None):
    """Issue an HTTP request and return (status, parsed_json)."""
    url = f"{BASE_URL}{path}"
    data = None
    headers = {"User-Agent": "dualmirakl-proof-run"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=CTX) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read() if exc.fp else b"{}"
        try:
            parsed = json.loads(body_bytes.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed = {"raw": body_bytes.decode(errors="replace")}
        return exc.code, parsed


def main() -> int:
    failures = 0
    def check(label, condition, detail=""):
        nonlocal failures
        if condition:
            print(f"[PASS] {label}")
        else:
            print(f"[FAIL] {label}  {detail}")
            failures += 1

    # ── 1. GET /simulation/flame — baseline status (should be inactive) ──────────
    status1, data1 = _request("GET", "/simulation/flame")
    check("1  GET /simulation/flame returns 200", status1 == 200,
          f"got {status1}")
    check("1  active == false before activation attempt",
          data1.get("active") is False,
          f"got active={data1.get('active')}")

    # ── 2. POST /simulation/flame/activate — attempt activation ──────────────────
    status2, data2 = _request("POST", "/simulation/flame/activate", {})
    check("2  Activation handled gracefully (no 5xx server error)",
          status2 < 500,
          f"got {status2}")

    # ── 3. Verify activation failed ─────────────────────────────────────────────
    check("3  active == false in activate response",
          data2.get("active") is False,
          f"got active={data2.get('active')}")

    # ── 4. Verify error message is informative ───────────────────────────────────
    error_msg = data2.get("error", "") or data2.get("detail", "") or ""
    error_lower = error_msg.lower()
    informative = "pyflamegpu" in error_lower or "import" in error_lower or "flame" in error_lower
    check("4  Error message mentions pyflamegpu/import/flame",
          informative,
          f"error={error_msg!r}")

    # ── 5. Verify response contains an error field ──────────────────────────────
    has_error_field = "error" in data2 or "detail" in data2
    check("5  Response contains error or detail field",
          has_error_field,
          f"keys={list(data2.keys())}")

    # ── 6. GET /simulation/flame again — state unchanged ────────────────────────
    status3, data3 = _request("GET", "/simulation/flame")
    check("6  GET /simulation/flame still returns 200",
          status3 == 200,
          f"got {status3}")
    check("6  active still false after failed activation",
          data3.get("active") is False,
          f"got active={data3.get('active')}")

    # ── Summary ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
