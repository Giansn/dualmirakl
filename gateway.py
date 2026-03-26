import os
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from sentence_transformers import SentenceTransformer
import httpx

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_proj_dir = Path(__file__).parent

AUTHORITY = os.getenv("AUTHORITY_URL", "http://localhost:8000/v1")
SWARM     = os.getenv("SWARM_URL",     "http://localhost:8001/v1")

# e5-small-v2 loaded once at startup — CPU inference ~2-5ms per call
_embed_path = os.path.join(os.getenv("HF_HOME", "/per.volume/huggingface"), "hub", "e5-small-v2")
_embed = SentenceTransformer(_embed_path)

client = httpx.AsyncClient(
    http2=True,
    timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
)

def route(model: str) -> str:
    return AUTHORITY if "authority" in model else SWARM

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    target = route(body.get("model", "authority"))
    if body.get("stream", False):
        async def gen():
            async with client.stream("POST", f"{target}/chat/completions", json=body) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")
    r = await client.post(f"{target}/chat/completions", json=body)
    return r.json()

@app.post("/v1/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    inp = body.get("input", [])
    texts = inp if isinstance(inp, list) else [inp]
    loop = asyncio.get_event_loop()
    vectors = await loop.run_in_executor(None, _embed.encode, texts)
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vectors)],
        "model": "e5-small-v2",
    }

@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [
        {"id": "authority", "object": "model"},
        {"id": "swarm", "object": "model"},
        {"id": "e5-small-v2", "object": "model"},
    ]}

@app.get("/", response_class=HTMLResponse)
async def ui():
    html_file = _proj_dir / "interface.html"
    html = html_file.read_text(encoding="utf-8")
    # Inject live health status server-side (browser can't reach localhost through proxies)
    s = await health()
    for dot_id, key in [("stAuth", "authority"), ("stSwarm", "swarm"), ("stEmbed", "e5-small-v2")]:
        cls = "dot on" if s.get(key) == "up" else "dot off"
        html = html.replace(f'class="dot chk" id="{dot_id}"', f'class="{cls}" id="{dot_id}"')
    return HTMLResponse(html)


@app.get("/health")
async def health():
    s = {"e5-small-v2": "up"}  # CPU-local, always available
    for name, url in [("authority", AUTHORITY), ("swarm", SWARM)]:
        try:
            r = await client.get(url.replace("/v1", "") + "/health", timeout=3.0)
            s[name] = "up" if r.status_code == 200 else f"error {r.status_code}"
        except Exception as e:
            s[name] = f"down ({e})"
    return s


# ── Document Context Store ────────────────────────────────────────────────────
#
# Uploaded documents are chunked, embedded via e5-small-v2, and stored as a
# temporary JSON file that the simulation can read as world context.
#
# Flow:
#   interface.html → POST /v1/documents → gateway chunks + embeds → world_context.json
#   sim_loop.py → reads world_context.json → injects into agent prompts
#
# The context file persists until cleared or overwritten by a new upload.
# ──────────────────────────────────────────────────────────────────────────────

import json
from datetime import datetime, timezone

_context_dir = _proj_dir / "context"
_context_file = _context_dir / "world_context.json"


def _chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Split text into sentence-boundary-aware chunks."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > chunk_size and current:
            chunks.append(current.strip())
            current = s
        else:
            current = f"{current} {s}" if current else s
    if current.strip():
        chunks.append(current.strip())
    return chunks


@app.post("/v1/documents")
async def upload_document(req: Request):
    """
    Receive document text, chunk it, embed via e5-small-v2, and store
    as world_context.json for the simulation to use.

    Body: {"text": "...", "name": "file.pdf", "role": "world_context"}

    The context file contains:
    {
        "documents": [
            {"name": "...", "uploaded_at": "...", "chunks": [
                {"text": "...", "embedding": [...], "index": 0}, ...
            ]}
        ],
        "summary": "concatenated top-level text for agent injection",
        "n_chunks": int,
        "n_documents": int,
    }
    """
    body = await req.json()
    text = body.get("text", "")
    name = body.get("name", "unnamed")
    role = body.get("role", "world_context")
    append = body.get("append", True)

    if not text.strip():
        return {"error": "No text provided"}

    _context_dir.mkdir(parents=True, exist_ok=True)

    # Load existing context if appending
    existing = {"documents": [], "summary": "", "n_chunks": 0, "n_documents": 0}
    if append and _context_file.exists():
        try:
            existing = json.loads(_context_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Chunk and embed
    chunks = _chunk_text(text, chunk_size=400)
    loop = asyncio.get_event_loop()
    vectors = await loop.run_in_executor(None, _embed.encode, chunks)

    doc_entry = {
        "name": name,
        "role": role,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "n_chunks": len(chunks),
        "chunks": [
            {
                "text": chunk,
                "embedding": vec.tolist(),
                "index": i,
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ],
    }

    existing["documents"].append(doc_entry)
    existing["n_documents"] = len(existing["documents"])
    existing["n_chunks"] = sum(d["n_chunks"] for d in existing["documents"])

    # Build summary: first 3000 chars of all documents (for direct agent injection)
    all_text = "\n\n".join(
        "\n".join(c["text"] for c in d["chunks"])
        for d in existing["documents"]
    )
    existing["summary"] = all_text[:3000]

    _context_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    result = {
        "status": "stored",
        "document": name,
        "n_chunks": len(chunks),
        "total_chunks": existing["n_chunks"],
        "total_documents": existing["n_documents"],
        "context_file": str(_context_file),
    }

    # Optional GraphRAG extraction
    extract_graph = body.get("extract_graph", False)
    if extract_graph:
        import threading

        def _extract():
            import asyncio as _aio

            async def _do():
                from simulation.graph_rag import extract_graph as _extract_graph
                await _extract_graph(chunks=chunks, embed_fn=_embed.encode, doc_name=name)

            _aio.run(_do())

        threading.Thread(target=_extract, daemon=True).start()
        result["graph_extraction"] = "started"

    return result


@app.get("/v1/documents")
async def list_documents():
    """List all uploaded documents in the context store."""
    if not _context_file.exists():
        return {"documents": [], "n_chunks": 0, "n_documents": 0}
    ctx = json.loads(_context_file.read_text(encoding="utf-8"))
    return {
        "documents": [
            {"name": d["name"], "role": d.get("role", "world_context"),
             "n_chunks": d["n_chunks"], "uploaded_at": d.get("uploaded_at")}
            for d in ctx.get("documents", [])
        ],
        "n_chunks": ctx.get("n_chunks", 0),
        "n_documents": ctx.get("n_documents", 0),
    }


@app.delete("/v1/documents")
async def clear_documents():
    """Clear all uploaded documents."""
    if _context_file.exists():
        _context_file.unlink()
    return {"status": "cleared"}


@app.post("/v1/documents/query")
async def query_documents(req: Request):
    """
    Semantic search over uploaded documents using e5-small-v2.

    Body: {"query": "...", "top_k": 5, "threshold": 0.5}
    Returns: most relevant chunks with similarity scores.
    """
    body = await req.json()
    query = body.get("query", "")
    top_k = body.get("top_k", 5)
    threshold = body.get("threshold", 0.5)

    if not query or not _context_file.exists():
        return {"results": []}

    ctx = json.loads(_context_file.read_text(encoding="utf-8"))

    # Embed query
    loop = asyncio.get_event_loop()
    q_vec = (await loop.run_in_executor(None, _embed.encode, [query]))[0]

    # Search all chunks
    import numpy as _np
    results = []
    for doc in ctx.get("documents", []):
        for chunk in doc.get("chunks", []):
            c_vec = _np.array(chunk["embedding"])
            sim = float(_np.dot(q_vec, c_vec) / (
                _np.linalg.norm(q_vec) * _np.linalg.norm(c_vec) + 1e-8
            ))
            if sim >= threshold:
                results.append({
                    "text": chunk["text"],
                    "score": round(sim, 4),
                    "document": doc["name"],
                    "chunk_index": chunk["index"],
                })

    results.sort(key=lambda x: -x["score"])
    return {"results": results[:top_k], "query": query}


# ── Simulation API ───────────────────────────────────────────────────────────

_sim_state = {
    "status": "idle", "tick": 0, "n_ticks": 0,
    "started_at": None, "run_dir": None,
    "events": [],  # [{tick, type, detail}]
    "scores": [],  # latest scores per agent
    "pct": 0,
}

# Live simulation objects — set by _run(), used by /v1/interview
_sim_live = {
    "participants": None,   # list[ParticipantAgent] during run
    "world_state": None,    # WorldState during run
    "loop": None,           # asyncio event loop of sim thread
}


@app.get("/simulation/preflight")
async def sim_preflight():
    """
    Infrastructure check — call on instance/pod arrival.
    Verifies vLLM servers, embedding model, output dir, context file.
    """
    from simulation.sim_loop import preflight_check
    return await preflight_check()


@app.get("/simulation/flame")
async def sim_flame_status():
    """
    FLAME GPU 2 status — reports engine, W&B, and Optuna availability.
    Always returns (never fails), even when FLAME is disabled.
    """
    from simulation.flame_setup import flame_status
    return flame_status()


@app.get("/simulation/detect")
async def sim_detect():
    """
    World context detection — call before user starts a simulation.
    Reports what context is present/missing and why each matters.
    Does NOT block — user decides whether to proceed or upload more.
    """
    from simulation.sim_loop import detect_missing_context
    return detect_missing_context()


@app.post("/simulation/start")
async def sim_start(req: Request):
    """Start a simulation run. Accepts config overrides in POST body."""
    if _sim_state["status"] == "running":
        return {"error": "Simulation already running", "status": _sim_state}

    body = await req.json() if req.headers.get("content-type") == "application/json" else {}

    # Run detection (non-blocking) and include warnings in response
    from simulation.sim_loop import detect_missing_context
    detection = detect_missing_context()

    _sim_state["status"] = "running"
    _sim_state["tick"] = 0
    _sim_state["n_ticks"] = body.get("n_ticks", 12)
    _sim_state["pct"] = 0
    _sim_state["scores"] = []
    _sim_state["events"] = []
    _sim_state["started_at"] = datetime.now(timezone.utc).isoformat()
    _sim_state["run_dir"] = None

    # Run simulation in background thread (doesn't block gateway)
    import threading

    def _run():
        import asyncio as _aio
        from simulation.sim_loop import run_simulation, close_client

        async def _sim():
            try:
                _sim_live["loop"] = _aio.get_event_loop()

                def _on_tick(info):
                    _sim_state["tick"] = info["tick"]
                    _sim_state["n_ticks"] = info["n_ticks"]
                    _sim_state["pct"] = info["pct"]
                    _sim_state["scores"] = info["scores"]
                    for ev in info["events"]:
                        _sim_state["events"].append({
                            "tick": info["tick"],
                            "pct": info["pct"],
                            **ev,
                        })
                    # Store live refs for interview endpoint
                    if "participants" in info:
                        _sim_live["participants"] = info["participants"]
                    if "world_state" in info:
                        _sim_live["world_state"] = info["world_state"]

                participants, world_state = await run_simulation(
                    n_ticks=body.get("n_ticks", 12),
                    n_participants=body.get("n_participants", 4),
                    k=body.get("k", 4),
                    alpha=body.get("alpha", 0.15),
                    max_tokens=body.get("max_tokens", 192),
                    seed=body.get("seed", 42),
                    score_mode=body.get("score_mode", "ema"),
                    logistic_k=body.get("logistic_k", 6.0),
                    on_tick=_on_tick,
                    flame_enabled=body.get("flame_enabled"),
                    flame_config=body.get("flame_config"),
                    continue_from=body.get("continue_from"),
                )
                # Run dynamics analysis on results
                from simulation.dynamics import analyze_simulation
                score_logs = [p.score_log for p in participants]
                config = {
                    "alpha": body.get("alpha", 0.15),
                    "kappa": body.get("kappa", 0.0),
                    "score_mode": body.get("score_mode", "ema"),
                    "logistic_k": body.get("logistic_k", 6.0),
                }
                analysis = analyze_simulation(score_logs, run_config=config)

                # Save analysis alongside simulation output
                from simulation.sim_loop import OUTPUT_DIR
                data_dir = Path(OUTPUT_DIR)
                # Find the most recent run dir
                run_dirs = sorted(data_dir.iterdir(), reverse=True) if data_dir.exists() else []
                if run_dirs:
                    run_dir = run_dirs[0]
                    (run_dir / "dynamics_analysis.json").write_text(
                        json.dumps(analysis, indent=2, default=str)
                    )
                    _sim_state["run_dir"] = str(run_dir)

                _sim_state["status"] = "completed"
                _sim_state["tick"] = body.get("n_ticks", 12)
            except Exception as e:
                _sim_state["status"] = f"error: {e}"
            finally:
                _sim_live["participants"] = None
                _sim_live["world_state"] = None
                _sim_live["loop"] = None
                try:
                    await close_client()
                except Exception:
                    pass

        _aio.run(_sim())

    threading.Thread(target=_run, daemon=True).start()

    return {
        "status": "started",
        "config": body,
        "context_detection": detection,
    }


@app.get("/simulation/status")
async def sim_status():
    return _sim_state


@app.get("/simulation/results")
async def sim_results():
    if _sim_state["status"] != "completed" or not _sim_state["run_dir"]:
        return {"error": "No completed simulation", "status": _sim_state["status"]}

    run_dir = Path(_sim_state["run_dir"])
    results = {}

    for fname in ["config.json", "trajectories.json", "observations.json",
                   "compliance.json", "interventions.json", "dynamics_analysis.json",
                   "flame_population.json"]:
        fpath = run_dir / fname
        if fpath.exists():
            results[fname.replace(".json", "")] = json.loads(fpath.read_text())

    return results


@app.post("/v1/interview")
async def interview_agent(req: Request):
    """
    Interview a live participant agent mid-simulation.

    Body: {"agent_id": "participant_0", "question": "How are you feeling?"}
    Returns the agent's in-character response via the swarm GPU.

    Only works while a simulation is running — agents must be live.
    """
    body = await req.json()
    agent_id = body.get("agent_id", "")
    question = body.get("question", "")

    if not agent_id or not question:
        return {"error": "agent_id and question are required"}

    if _sim_state["status"] != "running":
        return {"error": "No simulation running", "status": _sim_state["status"]}

    participants = _sim_live.get("participants")
    if not participants:
        return {"error": "Simulation participants not yet initialized"}

    # Find the target agent
    target = None
    for p in participants:
        if p.agent_id == agent_id:
            target = p
            break

    if target is None:
        available = [p.agent_id for p in participants]
        return {"error": f"Agent '{agent_id}' not found", "available": available}

    # Send interview via swarm GPU
    tick = _sim_state.get("tick", 0)
    try:
        interview_body = {
            "model": "swarm",
            "messages": [
                {"role": "system", "content": target._build_system_prompt()},
                *target.history[-4:],
                {"role": "user", "content": (
                    f"[Interview at tick {tick}] An external observer asks: {question}\n"
                    f"Answer honestly and in character."
                )},
            ],
            "max_tokens": 192,
            "temperature": 0.7,
        }
        r = await client.post(f"{SWARM}/chat/completions", json=interview_body)
        r.raise_for_status()
        response_text = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return {"error": f"Interview failed: {e}"}

    return {
        "agent_id": agent_id,
        "question": question,
        "response": response_text,
        "tick": tick,
        "score": round(target.behavioral_score, 3),
    }


@app.post("/simulation/optimize")
async def sim_optimize(req: Request):
    """
    Run Optuna Bayesian optimization over simulation parameters.
    Non-blocking — runs in background thread.

    Body: {
        "mode": "fast"|"full",
        "n_trials": 100,
        "include_flame": false,
        "target_mean": 0.5
    }
    """
    body = await req.json() if req.headers.get("content-type") == "application/json" else {}

    import threading

    def _run():
        try:
            from simulation.optimize import run_optimization
            run_optimization(
                mode=body.get("mode", "fast"),
                n_trials=body.get("n_trials", 100),
                include_flame=body.get("include_flame", False),
                target_mean=body.get("target_mean", 0.5),
            )
        except ImportError as e:
            _sim_state["status"] = f"error: {e}"

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "optimization_started", "config": body}


# ── GraphRAG endpoints ───────────────────────────────────────────────────────

@app.post("/v1/documents/extract_graph")
async def extract_graph_from_documents(req: Request):
    """
    Trigger GraphRAG entity/relation extraction on uploaded documents.

    Reads chunks from world_context.json, runs authority-slot extraction,
    stores results in DuckDB. Async — returns immediately.
    """
    if not _context_file.exists():
        return {"error": "No documents uploaded. Upload documents first via POST /v1/documents."}

    ctx = json.loads(_context_file.read_text(encoding="utf-8"))
    all_chunks = []
    for doc in ctx.get("documents", []):
        for chunk in doc.get("chunks", []):
            all_chunks.append(chunk["text"])

    if not all_chunks:
        return {"error": "No document chunks found."}

    import threading

    def _run():
        import asyncio as _aio

        async def _extract():
            from simulation.graph_rag import extract_graph
            entities, relations = await extract_graph(
                chunks=all_chunks,
                embed_fn=_embed.encode,
                doc_name="batch_extraction",
            )
            return len(entities), len(relations)

        return _aio.run(_extract())

    # Run in background thread (authority slot call)
    threading.Thread(target=_run, daemon=True).start()

    return {
        "status": "extraction_started",
        "n_chunks": len(all_chunks),
        "message": "GraphRAG extraction running in background. Use GET /v1/graph to check results.",
    }


@app.get("/v1/graph")
async def get_graph():
    """Get all entities and relations from the GraphRAG knowledge graph."""
    try:
        from simulation.graph_rag import get_graph_entities, get_graph_relations
        entities = get_graph_entities()
        relations = get_graph_relations()
        return {
            "entities": entities,
            "relations": relations,
            "n_entities": len(entities),
            "n_relations": len(relations),
        }
    except Exception as e:
        return {"error": str(e), "entities": [], "relations": []}


@app.delete("/v1/graph")
async def clear_graph():
    """Clear all entities and relations from the GraphRAG knowledge graph."""
    try:
        from simulation.graph_rag import clear_graph
        clear_graph()
        return {"status": "cleared"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/v1/graph/query")
async def query_graph(req: Request):
    """
    Semantic search over the knowledge graph.

    Body: {"query": "...", "top_k": 20, "threshold": 0.4}
    Returns relevant graph triples as context text.
    """
    body = await req.json()
    query = body.get("query", "")
    top_k = body.get("top_k", 20)
    threshold = body.get("threshold", 0.4)

    if not query:
        return {"error": "query is required", "context": ""}

    from simulation.graph_rag import query_graph_context
    context = query_graph_context(
        scenario_context=query,
        embed_fn=_embed.encode,
        top_k=top_k,
        threshold=threshold,
    )
    return {"context": context, "query": query}


# ── Post-Sim Analysis endpoints ──────────────────────────────────────────────

_analyse_state = {
    "status": "idle",
    "run_dir": None,
    "report": None,
    "steps": 0,
}


@app.post("/simulation/analyse")
async def sim_analyse(req: Request):
    """
    Start a post-simulation ReACT analysis.

    Body: {"run_id": "data/run_...", "questions": ["What drove divergence?"]}
    """
    body = await req.json()
    run_dir = body.get("run_id") or body.get("run_dir", "")
    questions = body.get("questions", ["What are the key findings from this simulation?"])

    if not run_dir:
        # Use most recent run
        from pathlib import Path as _P
        data_dir = _P("data")
        if data_dir.exists():
            run_dirs = sorted(data_dir.iterdir(), reverse=True)
            if run_dirs:
                run_dir = str(run_dirs[0])

    if not run_dir:
        return {"error": "No run_dir specified and no completed runs found."}

    if _analyse_state["status"] == "running":
        return {"error": "Analysis already running.", "status": _analyse_state}

    _analyse_state["status"] = "running"
    _analyse_state["run_dir"] = run_dir
    _analyse_state["report"] = None
    _analyse_state["steps"] = 0

    import threading

    def _run():
        import asyncio as _aio

        async def _do_analysis():
            from simulation.react_observer import PostSimAnalyser

            def _on_step(step, tool, preview):
                _analyse_state["steps"] = step + 1

            analyser = PostSimAnalyser(run_dir=run_dir)
            report = await analyser.analyse(
                questions=questions,
                on_step=_on_step,
            )
            _analyse_state["report"] = report
            _analyse_state["status"] = "completed"

        try:
            _aio.run(_do_analysis())
        except Exception as e:
            _analyse_state["status"] = f"error: {e}"
            _analyse_state["report"] = {"error": str(e)}

    threading.Thread(target=_run, daemon=True).start()

    return {
        "status": "analysis_started",
        "run_dir": run_dir,
        "questions": questions,
    }


@app.get("/simulation/analyse/status")
async def sim_analyse_status():
    """Check post-sim analysis progress."""
    return _analyse_state


@app.get("/simulation/analyse/report")
async def sim_analyse_report():
    """Get the completed analysis report."""
    if _analyse_state["status"] != "completed" or not _analyse_state["report"]:
        return {"error": "No completed analysis.", "status": _analyse_state["status"]}
    return _analyse_state["report"]


# ── Memory persistence endpoints ─────────────────────────────────────────────

@app.get("/v1/memories")
async def list_memories():
    """List persisted memory statistics across runs."""
    try:
        from simulation.agent_memory import DuckDBMemoryBackend
        backend = DuckDBMemoryBackend(run_id="query")
        return {
            "stats": backend.memory_stats(),
            "runs": backend.get_run_ids(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/v1/memories/{run_id}")
async def get_run_memories(run_id: str):
    """Get all persisted memories for a specific run."""
    try:
        from simulation.storage import get_db
        db = get_db()
        rows = db.execute(
            "SELECT agent_id, title, content, tags, tick, memory_type, importance "
            "FROM agent_memories WHERE run_id = ? ORDER BY agent_id, tick",
            [run_id],
        ).fetchall()
        return {
            "run_id": run_id,
            "n_memories": len(rows),
            "memories": [
                {"agent_id": r[0], "title": r[1], "content": r[2], "tags": r[3],
                 "tick": r[4], "type": r[5], "importance": r[6]}
                for r in rows
            ],
        }
    except Exception as e:
        return {"error": str(e)}
