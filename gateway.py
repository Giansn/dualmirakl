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
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


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


# ── Simulation API ───────────────────────────────────────────────────────────

import json
from datetime import datetime, timezone

_sim_state = {"status": "idle", "tick": 0, "n_ticks": 0, "started_at": None, "run_dir": None}


@app.post("/simulation/start")
async def sim_start(req: Request):
    """Start a simulation run. Accepts config overrides in POST body."""
    if _sim_state["status"] == "running":
        return {"error": "Simulation already running", "status": _sim_state}

    body = await req.json() if req.headers.get("content-type") == "application/json" else {}

    _sim_state["status"] = "running"
    _sim_state["tick"] = 0
    _sim_state["started_at"] = datetime.now(timezone.utc).isoformat()
    _sim_state["run_dir"] = None

    # Run simulation in background thread (doesn't block gateway)
    import threading

    def _run():
        import asyncio as _aio
        from simulation.sim_loop import run_simulation, close_client

        async def _sim():
            try:
                participants, world_state = await run_simulation(
                    n_ticks=body.get("n_ticks", 12),
                    n_participants=body.get("n_participants", 4),
                    k=body.get("k", 4),
                    alpha=body.get("alpha", 0.15),
                    max_tokens=body.get("max_tokens", 192),
                    seed=body.get("seed", 42),
                    score_mode=body.get("score_mode", "ema"),
                    logistic_k=body.get("logistic_k", 6.0),
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
                try:
                    await close_client()
                except Exception:
                    pass

        _aio.run(_sim())

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "started", "config": body}


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
                   "compliance.json", "interventions.json", "dynamics_analysis.json"]:
        fpath = run_dir / fname
        if fpath.exists():
            results[fname.replace(".json", "")] = json.loads(fpath.read_text())

    return results
