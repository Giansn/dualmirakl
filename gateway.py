from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx, asyncio

app = FastAPI()

GPU0 = "http://localhost:8000/v1"  # Command-R
GPU1 = "http://localhost:8001/v1"  # Qwen

client = httpx.AsyncClient(
    http2=True,
    timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
)

def route(model: str) -> str:
    return GPU0 if "command-r" in model else GPU1

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    target = route(body.get("model", "command-r"))
    stream = body.get("stream", False)
    if stream:
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
    r = await client.post(f"{GPU1}/embeddings", json=body)
    return r.json()

@app.get("/v1/models")
async def models():
    return {"object":"list","data":[
        {"id":"command-r","object":"model"},
        {"id":"qwen","object":"model"},
    ]}

@app.get("/health")
async def health():
    s = {}
    for name, url in [("command-r", GPU0), ("qwen", GPU1)]:
        try:
            r = await client.get(url.replace("/v1","") + "/health", timeout=3.0)
            s[name] = "up" if r.status_code == 200 else f"error {r.status_code}"
        except Exception as e:
            s[name] = f"down ({e})"
    return s
