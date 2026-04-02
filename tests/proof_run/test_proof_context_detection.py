"""
Proof-run: Context Detection
Verifies /simulation/detect and /v1/documents against a live RunPod gateway.
"""

import json
import os
import ssl
import sys
import urllib.request

BASE = os.getenv("PROOF_BASE_URL", "https://48acx0kqem74jt-9000.proxy.runpod.net")
CTX = ssl._create_unverified_context()
FAIL = False
def get(path: str) -> dict:
    url = f"{BASE}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "dualmirakl-proof-run"})
    with urllib.request.urlopen(req, context=CTX, timeout=30) as resp:
        return json.loads(resp.read().decode())
def check(label: str, ok: bool, detail: str = ""):
    global FAIL
    tag = "[PASS]" if ok else "[FAIL]"
    msg = f"{tag} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not ok:
        FAIL = True
# ── /simulation/detect ──────────────────────────────────────────────────────

detect = get("/simulation/detect")

check("detect: has_context == true",
      detect.get("has_context") is True,
      f"got {detect.get('has_context')}")

check("detect: n_documents >= 10",
      detect.get("n_documents", 0) >= 10,
      f"got {detect.get('n_documents')}")

check("detect: can_proceed == true",
      detect.get("can_proceed") is True,
      f"got {detect.get('can_proceed')}")

present_cats = [p["category"] for p in detect.get("present", [])]
check("detect: >= 3 present categories",
      len(present_cats) >= 3,
      f"got {len(present_cats)}: {present_cats}")

missing = detect.get("missing", [])
check("detect: missing list is empty",
      len(missing) == 0,
      f"got {len(missing)}: {[m['category'] for m in missing]}" if missing else "none missing")

warnings = detect.get("warnings", [])
check("detect: warnings list is empty",
      len(warnings) == 0,
      f"got {len(warnings)}: {warnings}" if warnings else "no warnings")
# ── /v1/documents ───────────────────────────────────────────────────────────

docs = get("/v1/documents")

check("docs: n_documents >= 10",
      docs.get("n_documents", 0) >= 10,
      f"got {docs.get('n_documents')}")

check("docs: n_chunks >= 500",
      docs.get("n_chunks", 0) >= 500,
      f"got {docs.get('n_chunks')}")

doc_list = docs.get("documents", [])
required_fields = ("name", "role", "n_chunks", "uploaded_at")
all_ok = True
bad = []
for d in doc_list:
    for f in required_fields:
        if f not in d or d[f] is None:
            all_ok = False
            bad.append(f"{d.get('name', '?')} missing {f}")
check("docs: every document has name/role/n_chunks/uploaded_at",
      all_ok,
      ", ".join(bad) if bad else f"checked {len(doc_list)} documents")
# ── Summary ─────────────────────────────────────────────────────────────────

print()
if FAIL:
    print("RESULT: some checks failed")
    sys.exit(1)
else:
    print("RESULT: all checks passed")
