"""
Batched vLLM Inference.

Instead of N sequential requests per tick: collect all agent prompts,
send as concurrent batch to vLLM, assign responses.

vLLM batches internally via continuous batching — we need to send
requests SIMULTANEOUSLY so they land in the same batch.
Routes by backend (authority vs swarm) for true dual-GPU parallelism.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from orchestrator import chat


@dataclass
class InferenceRequest:
    """A single agent inference request."""
    agent_id: str
    backend: str  # "authority" | "swarm"
    messages: list[dict]
    max_tokens: int = 256
    temperature: float = 0.7


@dataclass
class InferenceResult:
    agent_id: str
    response: str
    error: str | None = None


async def _single_infer(req: InferenceRequest) -> str:
    """Single async inference call via orchestrator."""
    return await chat(
        backend=req.backend,
        messages=req.messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )


async def batch_infer(requests: list[InferenceRequest]) -> list[InferenceResult]:
    """
    Send all requests in parallel.

    Requests to authority (GPU 0) and swarm (GPU 1) run
    truly parallel on different GPUs.
    """
    tasks = [_single_infer(req) for req in requests]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        InferenceResult(
            agent_id=req.agent_id,
            response=raw if isinstance(raw, str) else "",
            error=None if isinstance(raw, str) else str(raw),
        )
        for req, raw in zip(requests, raw_results)
    ]


async def batch_infer_chunked(
    requests: list[InferenceRequest],
    max_concurrent: int = 16,
) -> list[InferenceResult]:
    """
    Like batch_infer, but with backpressure.

    max_concurrent limits simultaneous requests per GPU to avoid
    vLLM queue overflow and OOM.

    On 2x RTX PRO 4500 (32GB): ~8 concurrent per GPU is safe
    with 30B models at 4-bit quantization.
    """
    semaphores = {
        "authority": asyncio.Semaphore(max(1, max_concurrent // 2)),
        "swarm": asyncio.Semaphore(max(1, max_concurrent // 2)),
    }

    async def _throttled(req: InferenceRequest) -> str:
        sem = semaphores.get(req.backend, semaphores["swarm"])
        async with sem:
            return await _single_infer(req)

    tasks = [_throttled(req) for req in requests]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        InferenceResult(
            agent_id=req.agent_id,
            response=raw if isinstance(raw, str) else "",
            error=None if isinstance(raw, str) else str(raw),
        )
        for req, raw in zip(requests, raw_results)
    ]
