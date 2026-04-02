"""
Proof-run test: FLAME GPU 2 status endpoint.

Verifies GET /simulation/flame returns correct config and matches local .env.
Runs against a live RunPod pod — no mocks.
"""

import json
import os
import ssl
import sys
import urllib.request

BASE_URL = os.environ.get(
    "PROOF_BASE_URL", "https://48acx0kqem74jt-9000.proxy.runpod.net"
)


def _find_env():
    """Locate .env relative to this file, walking up if needed."""
    candidate = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.isfile(candidate):
        return candidate
    # Worktree: walk up from cwd looking for .env next to gateway.py
    d = os.path.abspath(".")
    for _ in range(6):
        if os.path.isfile(os.path.join(d, ".env")):
            return os.path.join(d, ".env")
        d = os.path.dirname(d)
    return candidate  # fall back to original (will fail gracefully)

ENV_PATH = _find_env()

_SSL_CTX = ssl._create_unverified_context()


def _get(path):
    """GET a path, return parsed JSON."""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "dualmirakl-proof-run"})
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=30) as resp:
        return json.loads(resp.read())


def _parse_env(path):
    """Parse a .env file into a dict, ignoring comments and blank lines."""
    env = {}
    if not os.path.isfile(path):
        return env
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            # Strip inline comments preceded by whitespace
            for i, ch in enumerate(value):
                if ch == "#" and i > 0 and value[i - 1] == " ":
                    value = value[:i].rstrip()
                    break
            env[key.strip()] = value
    return env


def main():
    passed = 0
    failed = 0

    def check(label, ok, detail=""):
        nonlocal passed, failed
        tag = "[PASS]" if ok else "[FAIL]"
        msg = f"{tag} {label}"
        if not ok and detail:
            msg += f"  ({detail})"
        print(msg)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"--- FLAME Status & Config  (pod: {BASE_URL}) ---\n")

    try:
        data = _get("/simulation/flame")
    except Exception as e:
        print(f"[FAIL] GET /simulation/flame failed: {e}")
        return 1

    check("Response is valid JSON dict", isinstance(data, dict))

    check(
        "flame_enabled == false",
        data.get("flame_enabled") is False,
        f"got {data.get('flame_enabled')!r}",
    )
    check(
        "flame_gpu == 2",
        data.get("flame_gpu") == 2,
        f"got {data.get('flame_gpu')!r}",
    )
    check(
        "flame_n_population == 10000000",
        data.get("flame_n_population") == 10_000_000,
        f"got {data.get('flame_n_population')!r}",
    )
    check(
        'pyflamegpu == "not installed"',
        data.get("pyflamegpu") == "not installed",
        f"got {data.get('pyflamegpu')!r}",
    )

    wandb_val = data.get("wandb")
    check(
        "wandb status present",
        wandb_val in ("installed", "not installed"),
        f"got {wandb_val!r}",
    )
    print(f"       (wandb = {wandb_val!r})")

    optuna_val = data.get("optuna")
    check(
        "optuna status present",
        optuna_val in ("installed", "not installed"),
        f"got {optuna_val!r}",
    )
    print(f"       (optuna = {optuna_val!r})")

    check(
        "active == false",
        data.get("active") is False,
        f"got {data.get('active')!r}",
    )

    # -- Cross-check with local .env --
    print()
    env = _parse_env(ENV_PATH)
    if not env:
        print(f"[FAIL] Could not parse .env at {ENV_PATH}")
        failed += 1
    else:
        env_enabled = env.get("FLAME_ENABLED", "0")
        check(
            ".env FLAME_ENABLED matches pod",
            (env_enabled == "1") == data.get("flame_enabled", False),
            f".env={env_enabled!r}, pod={data.get('flame_enabled')!r}",
        )
        env_gpu = env.get("FLAME_GPU", "2")
        check(
            ".env FLAME_GPU matches pod",
            int(env_gpu) == data.get("flame_gpu", -1),
            f".env={env_gpu!r}, pod={data.get('flame_gpu')!r}",
        )
        env_pop = env.get("FLAME_N_POPULATION", "10000")
        check(
            ".env FLAME_N_POPULATION matches pod",
            int(env_pop) == data.get("flame_n_population", -1),
            f".env={env_pop!r}, pod={data.get('flame_n_population')!r}",
        )

    # -- Summary --
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"  {passed}/{total} checks passed, {failed} failed")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
