import httpx
import asyncio
from typing import Literal

GPU0_URL = "http://localhost:8000/v1"
GPU1_URL = "http://localhost:8001/v1"

MODELS = {
    "command-r": {"url": GPU0_URL, "name": "command-r-7b"},
    "qwen":      {"url": GPU1_URL, "name": "qwen-7b"},
}

# Persistent client — reuses connections (keep-alive pool) across all calls.
# HTTP/2 multiplexes concurrent requests over a single TCP connection per host.
_client = httpx.AsyncClient(
    http2=True,
    timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
)


async def close_client() -> None:
    """Call once at shutdown to drain the connection pool."""
    await _client.aclose()


async def chat(
    backend: Literal["command-r", "qwen"],
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    cfg = MODELS[backend]
    payload = {
        "model": cfg["name"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = await _client.post(f"{cfg['url']}/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def dual_query(prompt: str) -> dict:
    """Send the same prompt to both backends concurrently."""
    msgs = [{"role": "user", "content": prompt}]
    cmd_r, qwen = await asyncio.gather(
        chat("command-r", msgs),
        chat("qwen", msgs),
    )
    return {"command-r": cmd_r, "qwen": qwen}


async def agent_turn(
    agent_id: str,
    backend: Literal["command-r", "qwen"],
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
    max_tokens: int = 256,
) -> str:
    """Single agent turn. max_tokens=256: action descriptions are short."""
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    response = await chat(backend, messages, max_tokens=max_tokens)
    print(f"[{agent_id}] {response[:120]}...")
    return response


async def health_check() -> dict:
    """Check both vLLM servers are up."""
    status = {}
    for name, cfg in MODELS.items():
        try:
            r = await _client.get(
                f"{cfg['url'].replace('/v1', '')}/health",
                timeout=5.0,
            )
            status[name] = "up" if r.status_code == 200 else f"error {r.status_code}"
        except Exception as e:
            status[name] = f"down ({e})"
    return status


if __name__ == "__main__":
    async def main():
        print("=== dualmirakl health check ===")
        s = await health_check()
        for k, v in s.items():
            print(f"  {k}: {v}")

        if all(v == "up" for v in s.values()):
            print("\n=== dual query test ===")
            results = await dual_query(
                "In one sentence: what is the core mechanism of media addiction?"
            )
            for model, resp in results.items():
                print(f"\n[{model}]\n{resp}")

        await close_client()

    asyncio.run(main())
