import os
import httpx
import asyncio
from typing import Literal

AUTHORITY_URL = os.getenv("AUTHORITY_URL", "http://localhost:8000/v1")
SWARM_URL     = os.getenv("SWARM_URL",     "http://localhost:8001/v1")

MODELS = {
    "authority": {"url": AUTHORITY_URL, "name": "authority"},
    "swarm":     {"url": SWARM_URL,     "name": "swarm"},
}

# Persistent client — reuses connections (keep-alive pool) across all calls.
# HTTP/2 multiplexes concurrent requests over a single TCP connection per host.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _client


async def close_client() -> None:
    """Call once at shutdown to drain the connection pool."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()


def _messages_to_cache_key(messages: list[dict]) -> str:
    """Deterministic string representation of messages for cache hashing."""
    import json as _json
    return _json.dumps(messages, sort_keys=True, ensure_ascii=True)


async def chat(
    backend: Literal["authority", "swarm"],
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    # Response cache: check for cached response
    try:
        from simulation.response_cache import cache as _cache
        if _cache._enabled:
            _prompt_key = _messages_to_cache_key(messages)
            _model_id = MODELS[backend]["name"]
            _cached = _cache.lookup(_prompt_key, _model_id, temperature)
            if _cached is not None:
                return _cached
    except Exception:
        _cache = None
        _prompt_key = None
        _model_id = None

    cfg = MODELS[backend]
    payload = {
        "model": cfg["name"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = await _get_client().post(f"{cfg['url']}/chat/completions", json=payload)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    # Nemotron reasoning parser may put output in reasoning_content with content=null
    response_text = msg.get("content") or msg.get("reasoning_content") or ""

    # Response cache: store new response
    try:
        if _cache is not None and _cache._enabled and _prompt_key is not None:
            _cache.store(_prompt_key, _model_id, temperature, None, response_text)
    except Exception:
        pass

    return response_text


async def dual_query(prompt: str) -> dict:
    """Send the same prompt to both backends concurrently."""
    msgs = [{"role": "user", "content": prompt}]
    authority, swarm = await asyncio.gather(
        chat("authority", msgs),
        chat("swarm", msgs),
    )
    return {"authority": authority, "swarm": swarm}


async def agent_turn(
    agent_id: str,
    backend: Literal["authority", "swarm"],
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
    max_tokens: int = 256,
) -> str:
    """Single agent turn. max_tokens=256: action descriptions are short."""
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        # Enforce strict role alternation (Mistral Nemo rejects non-alternating)
        for msg in history:
            if messages and msg["role"] == messages[-1]["role"]:
                # Merge consecutive same-role messages
                messages[-1]["content"] += "\n" + msg["content"]
            elif msg["role"] == "assistant" and messages[-1]["role"] == "system":
                # History starts with assistant — inject a synthetic user turn
                messages.append({"role": "user", "content": "(continuing)"})
                messages.append(msg)
            else:
                messages.append(msg)
    messages.append({"role": "user", "content": user_message})
    response = await chat(backend, messages, max_tokens=max_tokens)
    return response


async def health_check() -> dict:
    """Check both vLLM servers are up."""
    status = {}
    for name, cfg in MODELS.items():
        try:
            r = await _get_client().get(
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
                "In one sentence: what is an agent-based simulation?"
            )
            for model, resp in results.items():
                print(f"\n[{model}]\n{resp}")

        await close_client()

    asyncio.run(main())
