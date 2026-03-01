import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer
import httpx

app = FastAPI()

AUTHORITY = os.getenv("AUTHORITY_URL", "http://localhost:8000/v1")
SWARM     = os.getenv("SWARM_URL",     "http://localhost:8001/v1")

# e5-small-v2 loaded once at startup — CPU inference ~2-5ms per call
_embed = SentenceTransformer("/per.volume/huggingface/hub/e5-small-v2")

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
