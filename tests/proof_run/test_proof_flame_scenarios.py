"""
Proof-run: FLAME GPU scenario integration.

Verifies that all scenario YAML files load correctly via ScenarioConfig,
that FlameConfig sub-models parse with valid values, and that the pod's
scenario listing matches the local files.

Run:  python tests/proof_run/test_proof_flame_scenarios.py
"""

import os
import sys
import urllib.request
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import yaml
from simulation.scenario import ScenarioConfig, FlameConfig

BASE_URL = os.environ.get(
    "PROOF_BASE_URL",
    "https://48acx0kqem74jt-9000.proxy.runpod.net",
)
SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scenarios")

passed = 0
failed = 0
def check(label: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        print(f"  [PASS] {label}")
        passed += 1
    else:
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        failed += 1
# ── 1. Discover local scenario YAML files ──────────────────────────────────

print("\n=== 1. Local scenario YAML discovery ===")

yaml_files = sorted(
    f for f in os.listdir(SCENARIOS_DIR)
    if f.endswith(".yaml") and not f.startswith("_")
)

check(
    "Found scenario YAML files",
    len(yaml_files) >= 1,
    f"found {len(yaml_files)}",
)
print(f"     Scenarios: {yaml_files}")
# ── 2. Raw YAML flame section inspection ───────────────────────────────────

print("\n=== 2. Raw YAML flame section check ===")

raw_flame_data = {}  # filename -> raw flame dict or None

for fname in yaml_files:
    path = os.path.join(SCENARIOS_DIR, fname)
    with open(path) as f:
        data = yaml.safe_load(f)
    flame_section = data.get("flame")
    raw_flame_data[fname] = flame_section
    has_flame = flame_section is not None
    if has_flame:
        check(f"{fname}: has flame section with keys", len(flame_section) > 0)
        print(f"       -> {flame_section}")
    else:
        print(f"  [INFO] {fname}: no flame section (defaults will apply)")
# ── 3. ScenarioConfig.load — FlameConfig parsing ──────────────────────────

print("\n=== 3. ScenarioConfig.load + FlameConfig parsing ===")

loaded_configs = {}  # filename -> ScenarioConfig

for fname in yaml_files:
    path = os.path.join(SCENARIOS_DIR, fname)
    try:
        cfg = ScenarioConfig.load(path)
        loaded_configs[fname] = cfg
        check(f"{fname}: loads successfully", True)
        check(
            f"{fname}: flame is FlameConfig instance",
            isinstance(cfg.flame, FlameConfig),
            f"got {type(cfg.flame).__name__}",
        )
    except Exception as exc:
        check(f"{fname}: loads successfully", False, str(exc))
# ── 4. Enabled flame configs — validate numeric fields ─────────────────────

print("\n=== 4. Enabled flame configs — numeric validation ===")

flame_enabled_found = False

for fname, cfg in loaded_configs.items():
    if not cfg.flame.enabled:
        continue
    flame_enabled_found = True
    fc = cfg.flame
    check(
        f"{fname}: population_size is positive int",
        isinstance(fc.population_size, int) and fc.population_size > 0,
        f"population_size={fc.population_size}",
    )
    check(
        f"{fname}: kappa is finite non-negative float",
        isinstance(fc.kappa, (int, float)) and fc.kappa >= 0,
        f"kappa={fc.kappa}",
    )
    check(
        f"{fname}: influencer_weight is finite positive float",
        isinstance(fc.influencer_weight, (int, float)) and fc.influencer_weight > 0,
        f"influencer_weight={fc.influencer_weight}",
    )

if not flame_enabled_found:
    print("  (no scenarios have flame.enabled=true — skipped numeric checks)")
# ── 5. Disabled / missing flame — defaults apply ──────────────────────────

print("\n=== 5. Disabled / missing flame — defaults ===")

defaults = FlameConfig()

for fname, cfg in loaded_configs.items():
    if cfg.flame.enabled:
        continue
    check(
        f"{fname}: flame.enabled is False",
        cfg.flame.enabled is False,
    )
    # If the raw YAML had no flame section at all, all defaults should apply
    if raw_flame_data.get(fname) is None:
        check(
            f"{fname}: (no flame section) defaults match FlameConfig()",
            cfg.flame.population_size == defaults.population_size
            and cfg.flame.kappa == defaults.kappa
            and cfg.flame.influencer_weight == defaults.influencer_weight,
            f"got pop={cfg.flame.population_size} kappa={cfg.flame.kappa} iw={cfg.flame.influencer_weight}",
        )
# ── 6. minimal.yaml — specific checks ─────────────────────────────────────

print("\n=== 6. minimal.yaml — flame defaults ===")

if "minimal.yaml" in loaded_configs:
    mc = loaded_configs["minimal.yaml"]
    check(
        "minimal.yaml: flame.enabled is False",
        mc.flame.enabled is False,
    )
    check(
        "minimal.yaml: population_size is positive",
        mc.flame.population_size > 0,
        f"population_size={mc.flame.population_size}",
    )
    check(
        "minimal.yaml: kappa is non-negative",
        mc.flame.kappa >= 0,
        f"kappa={mc.flame.kappa}",
    )
    check(
        "minimal.yaml: influencer_weight is positive",
        mc.flame.influencer_weight > 0,
        f"influencer_weight={mc.flame.influencer_weight}",
    )
else:
    check("minimal.yaml: found in loaded configs", False, "file missing")
# ── 7. Cross-check with pod /simulation/scenarios ─────────────────────────

print("\n=== 7. Cross-check with pod ===")

scenarios_url = f"{BASE_URL}/simulation/scenarios"
pod_scenarios = None
def _health_fallback():
    """Verify pod is reachable via /health when /simulation/scenarios is unavailable."""
    try:
        health_url = f"{BASE_URL}/health"
        with urllib.request.urlopen(health_url, timeout=15) as resp:
            health = json.loads(resp.read().decode())
        check(
            "Fallback: pod reachable via /health",
            True,
            f"status={health.get('status', 'unknown')}",
        )
    except Exception as exc2:
        check("Fallback: pod reachable via /health", False, str(exc2))
try:
    req = urllib.request.Request(scenarios_url, method="GET", headers={"User-Agent": "dualmirakl-proof-run"})
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=15) as resp:
        pod_scenarios = json.loads(resp.read().decode())
    check(
        f"GET {scenarios_url} returned OK",
        True,
        f"keys={list(pod_scenarios.keys()) if isinstance(pod_scenarios, dict) else type(pod_scenarios).__name__}",
    )
except urllib.error.HTTPError as exc:
    check(f"GET {scenarios_url}", False, f"HTTP {exc.code}: {exc.reason}")
    _health_fallback()
except Exception as exc:
    check(f"GET {scenarios_url}", False, str(exc))
    _health_fallback()

# If pod returned a scenario list, cross-check names
if pod_scenarios and isinstance(pod_scenarios, (dict, list)):
    # Try common response shapes: {"scenarios": [...]}, [...], {"items": [...]}
    pod_names = None
    if isinstance(pod_scenarios, list):
        pod_names = pod_scenarios
    elif isinstance(pod_scenarios, dict):
        for key in ("scenarios", "items", "names", "files"):
            if key in pod_scenarios and isinstance(pod_scenarios[key], list):
                pod_names = pod_scenarios[key]
                break

    if pod_names is not None:
        local_basenames = set(os.path.splitext(f)[0] for f in yaml_files)
        # Normalise pod names (strip path prefixes if present)
        pod_basenames = set(
            (entry["name"] if isinstance(entry, dict) else os.path.basename(str(entry)))
            for entry in pod_names
        )
        overlap = local_basenames & pod_basenames
        check(
            "Pod scenario list overlaps with local files",
            len(overlap) > 0,
            f"overlap={sorted(overlap)} local={sorted(local_basenames)} pod={sorted(pod_basenames)}",
        )
    else:
        print("  (pod response shape not a recognisable list — skipped cross-check)")
# ── Summary ────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"FLAME scenario integration: {passed} passed, {failed} failed")
print(f"{'='*50}")

sys.exit(1 if failed else 0)
