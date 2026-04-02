#!/usr/bin/env python3
"""Proof-run: verify _flame_config_from_env builds correct FLAME config."""

import os
import sys
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

BASE_URL = "https://48acx0kqem74jt-9000.proxy.runpod.net"
ENV_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".env")

EXPECTED_KEYS = {
    "n_population", "n_influencers", "space_size", "interaction_radius",
    "alpha", "kappa", "dampening", "influencer_weight", "score_mode",
    "logistic_k", "drift_sigma", "mobility", "sub_steps", "gpu_id", "seed",
}

ENV_DEFAULTS = {
    "FLAME_GPU": "2",
    "FLAME_N_POPULATION": "10000000",
    "FLAME_SPACE_SIZE": "100000",
    "FLAME_INTERACTION_RADIUS": "500",
    "FLAME_MOBILITY": "0.1",
    "FLAME_KAPPA": "0.1",
    "FLAME_INFLUENCER_WEIGHT": "5.0",
    "FLAME_DRIFT_SIGMA": "0.01",
    "FLAME_SUB_STEPS": "10",
}

passes = 0
fails = 0
def check(label, condition, detail=""):
    global passes, fails
    if condition:
        print(f"[PASS] {label}")
        passes += 1
    else:
        msg = f"[FAIL] {label}"
        if detail:
            msg += f"  -- {detail}"
        print(msg)
        fails += 1
def save_env(keys):
    """Snapshot current env vars so we can restore later."""
    return {k: os.environ.get(k) for k in keys}
def restore_env(snapshot):
    """Restore env vars from snapshot."""
    for k, v in snapshot.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
def parse_env_file(path):
    """Parse a .env file into a dict (ignoring comments and blank lines)."""
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            # strip inline comments
            val = val.split("#")[0].strip()
            result[key] = val
    return result
def main():
    from simulation.preflight import _flame_config_from_env

    test_keys = list(ENV_DEFAULTS.keys()) + [
        "SIM_N_PARTICIPANTS", "SIM_ALPHA", "SIM_SCORE_MODE",
        "SIM_LOGISTIC_K", "SIM_SEED",
    ]
    snapshot = save_env(test_keys)

    try:
        # ── 1. Basic import & call ────────────────────────────────────────
        cfg = _flame_config_from_env()
        check("import and call _flame_config_from_env", cfg is not None)

        # ── 2. Set env vars and verify returned values ────────────────────
        os.environ["FLAME_N_POPULATION"] = "5000"
        os.environ["FLAME_KAPPA"] = "0.05"
        os.environ["FLAME_INFLUENCER_WEIGHT"] = "3.0"
        os.environ["FLAME_GPU"] = "1"
        os.environ["FLAME_DRIFT_SIGMA"] = "0.02"
        os.environ["FLAME_SUB_STEPS"] = "20"
        os.environ["FLAME_SPACE_SIZE"] = "50000"
        os.environ["FLAME_INTERACTION_RADIUS"] = "250"
        os.environ["FLAME_MOBILITY"] = "0.2"

        cfg = _flame_config_from_env()
        check("n_population from env", cfg["n_population"] == 5000,
              f"got {cfg['n_population']}")
        check("kappa from env", cfg["kappa"] == 0.05,
              f"got {cfg['kappa']}")
        check("influencer_weight from env", cfg["influencer_weight"] == 3.0,
              f"got {cfg['influencer_weight']}")
        check("gpu_id from env", cfg["gpu_id"] == 1,
              f"got {cfg['gpu_id']}")
        check("drift_sigma from env", cfg["drift_sigma"] == 0.02,
              f"got {cfg['drift_sigma']}")
        check("sub_steps from env", cfg["sub_steps"] == 20,
              f"got {cfg['sub_steps']}")

        # ── 3. Override mechanism ─────────────────────────────────────────
        cfg_ov = _flame_config_from_env(overrides={"n_population": 999})
        check("override n_population", cfg_ov["n_population"] == 999,
              f"got {cfg_ov['n_population']}")
        check("override preserves other keys", cfg_ov["kappa"] == 0.05,
              f"kappa={cfg_ov['kappa']}")

        cfg_multi = _flame_config_from_env(overrides={
            "kappa": 0.77, "gpu_id": 5, "drift_sigma": 0.99
        })
        check("multi-override kappa", cfg_multi["kappa"] == 0.77,
              f"got {cfg_multi['kappa']}")
        check("multi-override gpu_id", cfg_multi["gpu_id"] == 5,
              f"got {cfg_multi['gpu_id']}")
        check("multi-override drift_sigma", cfg_multi["drift_sigma"] == 0.99,
              f"got {cfg_multi['drift_sigma']}")

    finally:
        restore_env(snapshot)

    # ── 4. Parse .env file and verify FLAME_* defaults ────────────────
    env_vars = {}
    if os.path.isfile(ENV_FILE):
        env_vars = parse_env_file(ENV_FILE)
        for key, expected in ENV_DEFAULTS.items():
            actual = env_vars.get(key, "")
            check(f".env {key}={expected}", actual == expected,
                  f"got '{actual}'")
    else:
        check(".env file exists", False, f"not found at {ENV_FILE}")

    # ── 5. All expected keys present ──────────────────────────────────
    snapshot2 = save_env(test_keys)
    try:
        # Restore clean env for a defaults-only call
        for k in test_keys:
            os.environ.pop(k, None)
        cfg_clean = _flame_config_from_env()
        missing = EXPECTED_KEYS - set(cfg_clean.keys())
        check("all expected keys present", len(missing) == 0,
              f"missing: {missing}")
        for key in sorted(EXPECTED_KEYS):
            check(f"key '{key}' has non-None value",
                  cfg_clean.get(key) is not None,
                  f"value={cfg_clean.get(key)}")
    finally:
        restore_env(snapshot2)

    # ── 6. Cross-check with pod /simulation/flame ─────────────────────
    url = f"{BASE_URL}/simulation/flame"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "dualmirakl-proof-run"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        print(f"  pod /simulation/flame -> {json.dumps(data, indent=2)}")

        expected_pop = int(env_vars.get("FLAME_N_POPULATION", "10000000"))
        expected_gpu = int(env_vars.get("FLAME_GPU", "2"))

        pod_pop = data.get("flame_n_population")
        pod_gpu = data.get("flame_gpu")

        check("pod n_population matches .env",
              pod_pop == expected_pop,
              f"pod={pod_pop}, env={expected_pop}")
        check("pod gpu matches .env",
              pod_gpu == expected_gpu,
              f"pod={pod_gpu}, env={expected_gpu}")
    except urllib.error.URLError as e:
        check("pod /simulation/flame reachable", False, str(e))
    except Exception as e:
        check("pod /simulation/flame parse", False, str(e))

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  {passes} passed, {fails} failed")
    print(f"{'='*50}")
    return 0 if fails == 0 else 1
if __name__ == "__main__":
    raise SystemExit(main())
