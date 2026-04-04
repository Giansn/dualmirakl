"""
Proof-run: Scenario Listing & Validation

Verifies the /simulation/scenarios endpoint on a live RunPod gateway,
then validates local scenario YAML files and ScenarioConfig loading.

Usage:
    python tests/proof_run/test_proof_scenario_listing.py
"""

import json
import os
import ssl
import sys
import urllib.request

# ── project root on sys.path for local imports ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _ROOT)

BASE_URL = os.environ.get(
    "PROOF_BASE_URL",
    "https://48acx0kqem74jt-9000.proxy.runpod.net",
)

# Tolerant SSL context for RunPod proxy
try:
    _ssl_ctx = ssl.create_default_context()
except Exception:
    _ssl_ctx = ssl._create_unverified_context()

REQUIRED_SCENARIOS = ["minimal", "social_dynamics", "market_ecosystem"]

_HEADERS = {
    "User-Agent": "dualmirakl-proof-run/1.0",
    "Accept": "application/json",
}

def main():
    results: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str = ""):
        tag = "[PASS]" if passed else "[FAIL]"
        msg = f"{tag} {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        results.append((name, passed, detail))

    # ── 1. GET /simulation/scenarios ─────────────────────────────────────────────

    scenarios = None
    try:
        url = f"{BASE_URL}/simulation/scenarios"
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, context=_ssl_ctx, timeout=30) as resp:
            raw = resp.read().decode()
            scenarios = json.loads(raw)
        record(
            "GET /simulation/scenarios returns JSON list",
            isinstance(scenarios, list),
            f"type={type(scenarios).__name__}, len={len(scenarios) if isinstance(scenarios, list) else 'N/A'}",
        )
    except Exception as exc:
        record("GET /simulation/scenarios returns JSON list", False, str(exc))

    # ── 2. At least 5 scenarios ─────────────────────────────────────────────────

    if scenarios is not None and isinstance(scenarios, list):
        record(
            "At least 5 scenarios returned",
            len(scenarios) >= 5,
            f"count={len(scenarios)}",
        )
    else:
        record("At least 5 scenarios returned", False, "no scenario data")

    # ── 3. Known scenarios exist ─────────────────────────────────────────────────

    if scenarios is not None and isinstance(scenarios, list):
        names = {s.get("name") for s in scenarios if isinstance(s, dict)}
        for req_name in REQUIRED_SCENARIOS:
            record(
                f"Known scenario present: {req_name}",
                req_name in names,
                f"found={sorted(names)}",
            )
    else:
        for req_name in REQUIRED_SCENARIOS:
            record(f"Known scenario present: {req_name}", False, "no scenario data")

    # ── 4. Each scenario has name + path fields ──────────────────────────────────

    if scenarios is not None and isinstance(scenarios, list):
        all_have_fields = True
        missing_detail = []
        for s in scenarios:
            if not isinstance(s, dict):
                all_have_fields = False
                missing_detail.append(f"non-dict entry: {s!r}")
                continue
            if "name" not in s:
                all_have_fields = False
                missing_detail.append(f"missing 'name' in {s}")
            if "path" not in s:
                all_have_fields = False
                missing_detail.append(f"missing 'path' in {s}")
        record(
            "Every scenario has 'name' and 'path' fields",
            all_have_fields,
            "; ".join(missing_detail) if missing_detail else f"checked {len(scenarios)} entries",
        )
    else:
        record("Every scenario has 'name' and 'path' fields", False, "no scenario data")

    # ── 5. Local YAML files parse correctly ──────────────────────────────────────

    try:
        import yaml
    except ImportError:
        yaml = None

    scenarios_dir = os.path.join(_ROOT, "scenarios")
    yaml_files = sorted(
        f for f in os.listdir(scenarios_dir)
        if f.endswith(".yaml") and os.path.isfile(os.path.join(scenarios_dir, f))
    ) if os.path.isdir(scenarios_dir) else []

    if yaml is not None:
        all_parse = True
        parse_errors = []
        for fname in yaml_files:
            fpath = os.path.join(scenarios_dir, fname)
            try:
                with open(fpath) as fh:
                    data = yaml.safe_load(fh)
                if not isinstance(data, dict):
                    all_parse = False
                    parse_errors.append(f"{fname}: not a mapping")
            except Exception as exc:
                all_parse = False
                parse_errors.append(f"{fname}: {exc}")
        record(
            "Local scenario YAMLs parse successfully",
            all_parse and len(yaml_files) > 0,
            "; ".join(parse_errors) if parse_errors else f"parsed {len(yaml_files)} files",
        )
    else:
        record("Local scenario YAMLs parse successfully", False, "pyyaml not installed")

    # ── 6. ScenarioConfig.load() works for minimal.yaml ─────────────────────────

    minimal_path = os.path.join(scenarios_dir, "minimal.yaml")
    try:
        from simulation.scenario import ScenarioConfig

        config = ScenarioConfig.load(minimal_path)
        ok = (
            config.meta.name == "hello-world"
            and len(config.agents.roles) > 0
            and config.environment.tick_count == 3
        )
        record(
            "ScenarioConfig.load() works for minimal.yaml",
            ok,
            f"meta.name={config.meta.name!r}, roles={len(config.agents.roles)}, ticks={config.environment.tick_count}",
        )
    except Exception as exc:
        record("ScenarioConfig.load() works for minimal.yaml", False, str(exc))

    # ── Summary ──────────────────────────────────────────────────────────────────

    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = total - passed

    print()
    print(f"{'=' * 60}")
    print(f"  Scenario Listing & Validation: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  (all green)")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
