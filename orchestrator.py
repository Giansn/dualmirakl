"""
dualmirakl orchestrator
-----------------------
Claude Code talks to two local vLLM endpoints:
  - GPU0 port 8000 : Command-R 7B AWQ  (reasoning / synthesis)
  - GPU1 port 8001 : Qwen 2.5 7B AWQ  (generation / roleplay)

Used as the LLM backbone for the MA thesis multi-agent simulation
(JAX/NumPyro + FLAME GPU 2 + SimPy) on media addiction dynamics.
"""

import httpx
import asyncio
from typing import Literal

GPU0_URL = "http://localhost:8000/v1"
GPU1_URL = "http://localhost:8001/v1"

MODELS = {
    "command-r": {"url": GPU0_URL, "name": "command-r-7b"},
    "qwen":      {"url": GPU1_URL, "name": "qwen-7b"},
}


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
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{cfg['url']}/chat/completions", json=payload)
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
) -> str:
    """Single agent turn with system prompt and optional history."""
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    response = await chat(backend, messages)
    print(f"[{agent_id}] {response[:120]}...")
    return response


async def health_check() -> dict:
    """Check both vLLM servers are up."""
    status = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, cfg in MODELS.items():
            try:
                r = await client.get(f"{cfg['url'].replace('/v1', '')}/health")
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

    asyncio.run(main())
