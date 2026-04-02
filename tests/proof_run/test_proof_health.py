"""Proof-run: Health & Model Discovery

Verifies the live dualmirakl gateway on RunPod responds correctly
for /health, /v1/models, and the root UI endpoint.
"""

import json
import ssl
import sys
import time
import urllib.request

BASE = "https://48acx0kqem74jt-9000.proxy.runpod.net"
TIMEOUT = 5  # seconds — also used as the response-time ceiling
_SSL_CTX = ssl._create_unverified_context()


def _get(path: str):
    """GET a path, return (status, body_bytes, elapsed_seconds, headers)."""
    url = f"{BASE}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "dualmirakl-proof-run"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=_SSL_CTX) as resp:
            body = resp.read()
            elapsed = time.monotonic() - t0
            return resp.status, body, elapsed, resp.headers
    except urllib.error.HTTPError as exc:
        elapsed = time.monotonic() - t0
        return exc.code, exc.read(), elapsed, exc.headers


def _report(label: str, ok: bool, detail: str = ""):
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

    # ── GET /health ─────────────────────────────────────────────────────────
    status, body, elapsed, _ = _get("/health")
    check("GET /health returns 200", status == 200, f"status={status}")

    health = {}
    try:
        health = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        pass

    expected_keys = {"authority", "swarm", "e5-small-v2"}
    check(
        "/health JSON has required keys",
        expected_keys.issubset(health.keys()),
        f"keys={list(health.keys())}",
    )

    for svc in ("authority", "swarm", "e5-small-v2"):
        val = health.get(svc, "<missing>")
        check(f"/health {svc} == 'up'", val == "up", f"value={val!r}")

    check("/health response < 5s", elapsed < TIMEOUT, f"elapsed={elapsed:.2f}s")

    # ── GET /v1/models ─────────────────────────────────────────────────────
    status, body, elapsed, _ = _get("/v1/models")
    check("GET /v1/models returns 200", status == 200, f"status={status}")

    models = {}
    try:
        models = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        pass

    check(
        '/v1/models object == "list"',
        models.get("object") == "list",
        f"object={models.get('object')!r}",
    )

    data = models.get("data", None)
    check("/v1/models data is a list", isinstance(data, list), f"type={type(data).__name__}")

    model_ids = {m.get("id") for m in (data or [])}
    for mid in ("authority", "swarm", "e5-small-v2"):
        check(
            f"/v1/models includes '{mid}'",
            mid in model_ids,
            f"found={sorted(model_ids)}",
        )

    check("/v1/models response < 5s", elapsed < TIMEOUT, f"elapsed={elapsed:.2f}s")

    # ── GET / (root UI) ────────────────────────────────────────────────────
    status, body, elapsed, _ = _get("/")
    check("GET / returns 200", status == 200, f"status={status}")

    body_text = body.decode("utf-8", errors="replace").lower()
    has_marker = "dualmirakl" in body_text or "<html" in body_text or "<!doctype" in body_text
    check("GET / returns HTML content", has_marker, f"len={len(body)}")

    # ── Summary ────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"  {passed}/{total} checks passed, {failed} failed")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
