#!/usr/bin/env python3
"""Proof-run: Document Store CRUD (read-only) against a live RunPod gateway."""

import json
import ssl
import sys
import urllib.request

BASE = "https://48acx0kqem74jt-9000.proxy.runpod.net"
CTX = ssl._create_unverified_context()
FAIL = False
def _get(path):
    req = urllib.request.Request(f"{BASE}{path}", headers={"User-Agent": "dualmirakl-proof-run"})
    with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
        return json.loads(r.read())
def _post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json", "User-Agent": "dualmirakl-proof-run"},
    )
    with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
        return json.loads(r.read())
def check(label, ok, detail=""):
    global FAIL
    tag = "[PASS]" if ok else "[FAIL]"
    msg = f"{tag} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not ok:
        FAIL = True
# ── 1. List documents ────────────────────────────────────────────────────────

docs_resp = _get("/v1/documents")

n_docs = docs_resp.get("n_documents", 0)
check("GET /v1/documents returns at least 10 documents",
      n_docs >= 10, f"n_documents={n_docs}")

n_chunks = docs_resp.get("n_chunks", 0)
check("Total chunks >= 500",
      n_chunks >= 500, f"n_chunks={n_chunks}")

# ── 2. Document fields ───────────────────────────────────────────────────────

required_fields = {"name", "role", "n_chunks", "uploaded_at"}
docs = docs_resp.get("documents", [])
all_have_fields = all(required_fields <= set(d.keys()) for d in docs)
check("Each document has name, role, n_chunks, uploaded_at",
      all_have_fields)

# ── 3. Semantic search ───────────────────────────────────────────────────────

query_resp = _post("/v1/documents/query",
                   {"query": "population dynamics", "top_k": 5})
results = query_resp.get("results", [])
check("Semantic search returns results",
      len(results) > 0, f"n_results={len(results)}")

# ── 4. Query result fields ────────────────────────────────────────────────────

chunk_fields = {"text", "score", "document"}
all_chunks_ok = all(chunk_fields <= set(r.keys()) for r in results)
check("Query results have text, score, document fields",
      all_chunks_ok)

# ── 5. Similarity scores in [0,1] and sorted descending ──────────────────────

scores = [r.get("score", -1) for r in results]
scores_valid = all(0 <= s <= 1 for s in scores)
check("Similarity scores between 0 and 1",
      scores_valid, f"scores={scores}")

scores_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
check("Scores sorted descending",
      scores_sorted)

# ── 6. Graph endpoint responds ────────────────────────────────────────────────

graph_resp = _get("/v1/graph")
check("GET /v1/graph responds",
      isinstance(graph_resp, dict))

# ── 7. Graph response has n_entities and n_relations ──────────────────────────

has_n_entities = "n_entities" in graph_resp
has_n_relations = "n_relations" in graph_resp
check("Graph response has n_entities field",
      has_n_entities, f"n_entities={graph_resp.get('n_entities')}")
check("Graph response has n_relations field",
      has_n_relations, f"n_relations={graph_resp.get('n_relations')}")

# ── Summary ───────────────────────────────────────────────────────────────────

print()
sys.exit(1 if FAIL else 0)
