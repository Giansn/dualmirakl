import os
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
import httpx

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_proj_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=_proj_dir / "static"), name="static")

AUTHORITY = os.getenv("AUTHORITY_URL", "http://localhost:8000/v1")
SWARM     = os.getenv("SWARM_URL",     "http://localhost:8001/v1")

# GPU Harmony defaults — enable dual-GPU pipelining for all simulation runs
os.environ.setdefault("SIM_GPU_SPLIT", "1")
os.environ.setdefault("SIM_PIPELINE", "1")

# e5-small-v2 loaded once at startup — CPU inference ~2-5ms per call
_embed_path = os.path.join(os.getenv("HF_HOME", "/workspace/huggingface"), "hub", "e5-small-v2")
if not os.path.exists(_embed_path):
    _embed_path = "intfloat/e5-small-v2"  # fallback: download from HuggingFace Hub
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

@app.get("/gpu", response_class=HTMLResponse)
async def gpu_monitor():
    html_file = _proj_dir / "static" / "gpu_monitor.html"
    html = html_file.read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_viz():
    html_file = _proj_dir / "static" / "pipeline.html"
    html = html_file.read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/gpu/telemetry")
async def gpu_telemetry():
    import subprocess, shutil
    if not shutil.which("nvidia-smi"):
        return {"gpus": [], "ts": __import__("time").time(), "error": "nvidia-smi not found"}
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization": int(parts[2]),
                    "memory_used": int(parts[3]),
                    "memory_total": int(parts[4]),
                    "temp": int(parts[5]),
                    "power_draw": float(parts[6]),
                })
        resp = {"gpus": gpus, "ts": __import__("time").time()}
        # CPU metrics (lightweight, no extra dependency)
        try:
            import psutil
            mem = psutil.virtual_memory()
            resp["cpu"] = {
                "percent": psutil.cpu_percent(interval=None),
                "memory_used": mem.used // (1024 * 1024),
                "memory_total": mem.total // (1024 * 1024),
            }
        except ImportError:
            resp["cpu"] = None
        return resp
    except Exception as e:
        return {"gpus": [], "ts": __import__("time").time(), "error": str(e), "cpu": None}


@app.get("/", response_class=HTMLResponse)
async def ui():
    html_file = _proj_dir / "static" / "interface.html"
    html = html_file.read_text(encoding="utf-8")
    # Inject live health status server-side (browser can't reach localhost through proxies)
    s = await health()
    for dot_id, key in [("stAuth", "authority"), ("stSwarm", "swarm"), ("stEmbed", "e5-small-v2")]:
        cls = "dot on" if s.get(key) == "up" else "dot off"
        html = html.replace(f'class="dot chk" id="{dot_id}"', f'class="{cls}" id="{dot_id}"')
    import hashlib
    etag = hashlib.md5(html.encode()).hexdigest()[:12]
    return HTMLResponse(html, headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "ETag": f'"{etag}"',
    })


@app.get("/health")
async def health():
    s = {"e5-small-v2": "up"}  # CPU-local, always available
    for name, url in [("authority", AUTHORITY), ("swarm", SWARM)]:
        try:
            r = await client.get(url + "/models", timeout=3.0)
            if r.status_code == 200:
                models = [m["id"] for m in r.json().get("data", [])]
                s[name] = "up" if name in models else f"down (serving {models}, not {name})"
            else:
                s[name] = f"error {r.status_code}"
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

    # Build summary: concatenated text of all documents (for direct agent injection)
    all_text = "\n\n".join(
        "\n".join(c["text"] for c in d["chunks"])
        for d in existing["documents"]
    )
    existing["summary"] = all_text[:30000]

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

    n_seeds = body.get("n_seeds", 1)
    base_seed = body.get("seed", 42)
    n_ticks_req = body.get("n_ticks", 12)
    total_ticks = n_ticks_req * n_seeds

    _sim_state["status"] = "running"
    _sim_state["tick"] = 0
    _sim_state["n_ticks"] = total_ticks
    _sim_state["pct"] = 0
    _sim_state["scores"] = []
    _sim_state["events"] = []
    _sim_state["started_at"] = datetime.now(timezone.utc).isoformat()
    _sim_state["run_dir"] = None
    _sim_state["current_seed"] = 1
    _sim_state["n_seeds"] = n_seeds

    # Run simulation in background thread (doesn't block gateway)
    import threading

    def _run():
        import asyncio as _aio
        from simulation.sim_loop import run_simulation, close_client

        async def _sim():
            try:
                _sim_live["loop"] = _aio.get_event_loop()
                multi_run_logs: list[list[list[float]]] = []
                all_interventions: list[dict] = []
                last_participants = None
                last_world_state = None

                for seed_idx in range(n_seeds):
                    current_seed = base_seed + seed_idx
                    _sim_state["current_seed"] = seed_idx + 1
                    tick_offset = seed_idx * n_ticks_req

                    def _on_tick(info, _offset=tick_offset):
                        _sim_state["tick"] = _offset + info["tick"]
                        _sim_state["n_ticks"] = total_ticks
                        _sim_state["pct"] = int((_offset + info["tick"]) / total_ticks * 100)
                        _sim_state["scores"] = info["scores"]
                        for ev in info["events"]:
                            _sim_state["events"].append({
                                "tick": info["tick"],
                                "pct": _sim_state["pct"],
                                "seed": current_seed,
                                **ev,
                            })
                        if "participants" in info:
                            _sim_live["participants"] = info["participants"]
                        if "world_state" in info:
                            _sim_live["world_state"] = info["world_state"]

                    participants, world_state = await run_simulation(
                        n_ticks=n_ticks_req,
                        n_participants=body.get("n_participants", 4),
                        k=body.get("k", 4),
                        alpha=body.get("alpha", 0.15),
                        max_tokens=body.get("max_tokens", 192),
                        seed=current_seed,
                        score_mode=body.get("score_mode", "ema"),
                        logistic_k=body.get("logistic_k", 6.0),
                        on_tick=_on_tick,
                        flame_enabled=body.get("flame_enabled"),
                        flame_config=body.get("flame_config"),
                        continue_from=body.get("continue_from"),
                    )
                    multi_run_logs.append([p.score_log for p in participants])
                    last_participants = participants
                    last_world_state = world_state

                # Use last run's participants for final state
                participants = last_participants

                # Run dynamics analysis on primary run
                from simulation.dynamics import analyze_simulation
                score_logs = multi_run_logs[0]  # primary run for dynamics
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
                run_dirs = sorted(data_dir.iterdir(), reverse=True) if data_dir.exists() else []
                if run_dirs:
                    run_dir = run_dirs[0]
                    (run_dir / "dynamics_analysis.json").write_text(
                        json.dumps(analysis, indent=2, default=str)
                    )
                    _sim_state["run_dir"] = str(run_dir)

                    # Load interventions from run
                    _ivs = []
                    try:
                        _ivf = run_dir / "interventions.json"
                        if _ivf.exists():
                            _ivs = json.loads(_ivf.read_text())
                    except Exception:
                        pass

                    # Possibility branches report
                    try:
                        import numpy as _np
                        from simulation.possibility_report import compute_possibility_report
                        _report_config = {
                            **config,
                            "dampening": 1.0,
                            "susceptibility": float(_np.mean([p.susceptibility for p in participants])),
                            "resilience": float(_np.mean([p.resilience for p in participants])),
                        }
                        _wc = ""
                        try:
                            _wcf = _proj_dir / "context" / "world_context.json"
                            if _wcf.exists():
                                _wcd = json.loads(_wcf.read_text())
                                _wc = " ".join(c.get("text","") for c in _wcd.get("chunks",[]))
                        except Exception:
                            pass
                        _poss = compute_possibility_report(
                            score_logs=score_logs,
                            config=_report_config,
                            run_id=run_dir.name,
                            multi_run_logs=multi_run_logs if n_seeds > 1 else None,
                            world_context=_wc,
                            interventions=_ivs,
                        )
                        (run_dir / "possibility_branches.json").write_text(_poss.to_json())
                    except Exception as _pe:
                        import traceback as _tb2
                        _tb2.print_exc()

                _sim_state["status"] = "completed"
                _sim_state["tick"] = total_ticks
            except Exception as e:
                import traceback as _tb
                _sim_state["status"] = f"error: {e}"
                _tb.print_exc()
            finally:
                _sim_live["participants"] = None
                _sim_live["world_state"] = None
                _sim_live["loop"] = None

        _aio.run(_sim())

    threading.Thread(target=_run, daemon=True).start()

    return {
        "status": "started",
        "config": {**body, "n_seeds": n_seeds},
        "context_detection": detection,
    }


@app.post("/simulation/ensemble")
async def sim_ensemble(req: Request):
    """Start an ensemble of simulation runs with convergence-based stopping."""
    if _sim_state["status"] == "running":
        return {"error": "Simulation already running", "status": _sim_state}

    body = await req.json() if req.headers.get("content-type") == "application/json" else {}

    _sim_state["status"] = "running"
    _sim_state["started_at"] = datetime.now(timezone.utc).isoformat()
    _sim_state["tick"] = 0
    _sim_state["pct"] = 0
    _sim_state["events"] = []
    _sim_state["scores"] = []
    _sim_state["run_dir"] = None

    def _run():
        import asyncio as _aio
        from simulation.ensemble import run_ensemble

        async def _ensemble():
            try:
                def _on_run(info):
                    _sim_state["tick"] = info.get("completed_runs", 0)
                    _sim_state["n_ticks"] = info.get("total_runs", 0)
                    _sim_state["pct"] = info.get("pct", 0)

                result = await run_ensemble(
                    n_runs=body.get("n_runs", 10),
                    convergence_metric=body.get("convergence_metric", "mean_score"),
                    cv_threshold=body.get("cv_threshold", 0.05),
                    max_runs=body.get("max_runs"),
                    base_seed=body.get("base_seed", 42),
                    on_run=_on_run,
                    experiment_name=body.get("experiment_name"),
                    n_ticks=body.get("n_ticks", 12),
                    n_participants=body.get("n_participants", 4),
                )
                _sim_state["status"] = "completed"
                _sim_state["ensemble_result"] = result.to_dict()
            except Exception as e:
                import traceback as _tb
                _sim_state["status"] = f"error: {e}"
                _tb.print_exc()

        _aio.run(_ensemble())

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "ensemble_started", "config": body}


@app.post("/simulation/nested_ensemble")
async def sim_nested_ensemble(req: Request):
    """Start a nested ensemble (outer epistemic x inner aleatory+LLM)."""
    if _sim_state["status"] == "running":
        return {"error": "Simulation already running", "status": _sim_state}

    body = await req.json() if req.headers.get("content-type") == "application/json" else {}

    _sim_state["status"] = "running"
    _sim_state["started_at"] = datetime.now(timezone.utc).isoformat()
    _sim_state["tick"] = 0
    _sim_state["pct"] = 0
    _sim_state["events"] = []
    _sim_state["scores"] = []
    _sim_state["run_dir"] = None

    def _run():
        import asyncio as _aio
        from simulation.ensemble import run_nested_ensemble

        async def _nested():
            try:
                def _on_run(info):
                    _sim_state["tick"] = info.get("completed_runs", 0)
                    _sim_state["n_ticks"] = info.get("total_runs", 0)
                    _sim_state["pct"] = info.get("pct", 0)

                result = await run_nested_ensemble(
                    parameter_grid=body.get("parameter_grid", [{}]),
                    n_inner=body.get("n_inner", 5),
                    base_seed=body.get("base_seed", 42),
                    on_run=_on_run,
                    experiment_name=body.get("experiment_name"),
                    n_ticks=body.get("n_ticks", 12),
                    n_participants=body.get("n_participants", 4),
                )
                _sim_state["status"] = "completed"
                _sim_state["nested_result"] = result.to_dict()
            except Exception as e:
                import traceback as _tb
                _sim_state["status"] = f"error: {e}"
                _tb.print_exc()

        _aio.run(_nested())

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "nested_ensemble_started", "config": body}


@app.post("/simulation/scenario_tree")
async def sim_scenario_tree(req: Request):
    """Build a scenario tree from the most recent ensemble result."""
    body = await req.json() if req.headers.get("content-type") == "application/json" else {}

    # Find all_score_logs from the latest ensemble/nested result
    logs = None
    for key in ("ensemble_result", "nested_result"):
        r = _sim_state.get(key)
        if r and "all_score_logs" in r:
            logs = r["all_score_logs"]
            break

    if not logs:
        return {"error": "No ensemble data available. Run an ensemble first."}

    from simulation.scenario_tree import build_scenario_tree, reduce_tree, tree_to_dict
    tree = build_scenario_tree(
        logs,
        max_depth=body.get("max_depth", 3),
        max_branches=body.get("max_branches", 4),
        min_branch_prob=body.get("min_branch_prob", 0.05),
    )
    target = body.get("target_scenarios")
    if target and isinstance(target, int):
        tree = reduce_tree(tree, target)

    return {"scenario_tree": tree_to_dict(tree)}


@app.post("/simulation/calibrate")
async def sim_calibrate(req: Request):
    """Run ABC-SMC calibration in background."""
    if _sim_state.get("calibration_status") == "running":
        return {"error": "Calibration already running"}

    body = await req.json() if req.headers.get("content-type") == "application/json" else {}
    _sim_state["calibration_status"] = "running"

    def _run():
        import asyncio as _aio
        from simulation.abc_calibration import abc_smc, make_abc_sim_func, ABCPrior

        try:
            param_names = body.get("param_names", ["alpha", "dampening", "susceptibility", "resilience"])
            bounds = body.get("bounds", [(0.05, 0.4), (0.3, 1.0), (0.1, 0.9), (0.0, 0.5)])
            priors = [ABCPrior(name=n, type="uniform", low=b[0], high=b[1])
                      for n, b in zip(param_names, bounds)]
            observed = body.get("observed", {"mean_score": 0.5, "score_std": 0.1})

            sim_func = make_abc_sim_func(param_names, bounds)
            result = abc_smc(
                sim_func=sim_func,
                priors=priors,
                observed=observed,
                n_particles=body.get("n_particles", 100),
                n_populations=body.get("n_populations", 5),
            )
            _sim_state["calibration_status"] = "completed"
            _sim_state["calibration_result"] = result.to_dict()
        except Exception as e:
            import traceback
            _sim_state["calibration_status"] = f"error: {e}"
            traceback.print_exc()

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "calibration_started", "config": body}


@app.get("/simulation/calibrate/status")
async def sim_calibrate_status():
    """Return calibration progress."""
    return {
        "status": _sim_state.get("calibration_status", "idle"),
        "result": _sim_state.get("calibration_result"),
    }


@app.get("/simulation/report")
async def sim_report(req: Request):
    """Generate self-contained HTML report from latest simulation results."""
    from simulation.report import generate_report

    ensemble = _sim_state.get("ensemble_result") or _sim_state.get("nested_result")
    dynamics = None
    tree = None
    calibration = _sim_state.get("calibration_result")

    # Try to load dynamics analysis from latest run dir
    if _sim_state.get("run_dir"):
        try:
            da_path = Path(_sim_state["run_dir"]) / "dynamics_analysis.json"
            if da_path.exists():
                dynamics = json.loads(da_path.read_text())
        except Exception:
            pass

    html = generate_report(
        ensemble_result=ensemble,
        dynamics_analysis=dynamics,
        scenario_tree=tree,
        calibration_result=calibration,
    )

    download = req.query_params.get("download")
    if download:
        return Response(
            content=html,
            media_type="text/html",
            headers={"Content-Disposition": "attachment; filename=dualmirakl_report.html"},
        )
    return HTMLResponse(html)


@app.post("/simulation/pipeline")
async def sim_pipeline(req: Request):
    """Run the full output pipeline (ensemble + analysis + validation + report)."""
    if _sim_state["status"] == "running":
        return {"error": "Simulation already running", "status": _sim_state}

    body = await req.json() if req.headers.get("content-type") == "application/json" else {}

    _sim_state["status"] = "running"
    _sim_state["started_at"] = datetime.now(timezone.utc).isoformat()
    _sim_state["tick"] = 0
    _sim_state["pct"] = 0
    _sim_state["events"] = []
    _sim_state["scores"] = []
    _sim_state["run_dir"] = None

    def _run():
        import asyncio as _aio
        from simulation.output_pipeline import OutputPipeline, PipelineStageConfig
        from simulation.scenario import ScenarioConfig

        async def _pipeline():
            try:
                scenario_path = body.get("scenario", "scenarios/social_dynamics.yaml")
                config = ScenarioConfig.load(scenario_path)
                config.validate_scenario(strict=True)

                stage_cfg = PipelineStageConfig()
                for stage_name in body.get("skip_stages", []):
                    if hasattr(stage_cfg, stage_name):
                        object.__setattr__(stage_cfg, stage_name, False)

                def _on_stage(name, status):
                    _sim_state["events"].append({
                        "type": "pipeline_stage", "stage": name, "status": status,
                    })

                pipeline = OutputPipeline(
                    scenario_config=config,
                    stage_config=stage_cfg,
                    on_stage=_on_stage,
                )
                result = await pipeline.run(
                    n_runs=body.get("n_runs", 10),
                    base_seed=body.get("base_seed", 42),
                )
                _sim_state["status"] = "completed"
                _sim_state["pipeline_result"] = result.to_dict()
                _sim_state["run_dir"] = str(pipeline.output_dir)
            except Exception as e:
                import traceback as _tb
                _sim_state["status"] = f"error: {e}"
                _tb.print_exc()

        _aio.run(_pipeline())

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "pipeline_started", "config": body}


@app.get("/simulation/pipeline/result")
async def sim_pipeline_result():
    """Return pipeline result JSON."""
    pr = _sim_state.get("pipeline_result")
    if not pr:
        return {"error": "No pipeline result available", "status": _sim_state.get("status", "idle")}
    return pr


@app.get("/simulation/pipeline/report")
async def sim_pipeline_report(req: Request):
    """Return the HTML report from the latest pipeline run."""
    run_dir = _sim_state.get("run_dir")
    if not run_dir:
        return {"error": "No pipeline run completed"}
    report_path = Path(run_dir) / "report.html"
    if not report_path.exists():
        return {"error": "Report not found in pipeline output"}
    html = report_path.read_text()
    if req.query_params.get("download"):
        return Response(
            content=html, media_type="text/html",
            headers={"Content-Disposition": "attachment; filename=pipeline_report.html"},
        )
    return HTMLResponse(html)


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
                   "possibility_branches.json", "flame_population.json"]:
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
