# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

RunPod pod — two RTX PRO 4500 Blackwell GPUs (32GB VRAM each, sm_12.0), CUDA 12.8, PyTorch 2.9.1.
Models pre-downloaded to `/per.volume/huggingface/hub/`. Copy `.env.example` → `.env`, set `ANTHROPIC_API_KEY`.
Working directory for all commands: `/per.volume/dualmirakl/`.

## Commands

```bash
# Start both vLLM servers (staggered; waits up to 3 min per health check)
bash start_all.sh

# Stop servers
bash stop_all.sh

# Orchestrator health check + test dual query
python orchestrator.py

# Run the simulation (12 h, 4 agents by default)
python simulation/sim_loop.py

# Start the FastAPI gateway (unified /v1 endpoint over both backends)
uvicorn gateway:app --host 0.0.0.0 --port 9000

# Environment audit (disk, GPU, libs, model files)
bash audit_env.sh

# Install Python deps (vLLM already system-installed on RunPod)
pip install -r requirements.txt
```

Logs: `logs/gpu0.log`, `logs/gpu1.log`. PIDs: `logs/pids.txt`.

## Architecture

```
GPU 0 — Command-R 7B AWQ  :8000   (reasoning: researcher, policy_analyst)
GPU 1 — Qwen 2.5 7B AWQ   :8001   (generation: media_user, platform_ai)
         │                │
         └──── gateway.py ──── unified /v1 endpoint :9000
                   │
              orchestrator.py  (async dual-backend client)
                   │
         simulation/sim_loop.py  (SimPy + asyncio.gather ticks)
```

**`orchestrator.py`** — single module-level `httpx.AsyncClient` (HTTP/2, keep-alive). Key functions:
- `chat(backend, messages)` — POST to one backend
- `dual_query(prompt)` — same prompt to both backends via `asyncio.gather`
- `agent_turn(agent_id, backend, system_prompt, user_message, history)` — one simulation step
- `health_check()` — checks both `/health` endpoints
- `close_client()` — drain pool at shutdown; always call in a `finally` block

**`gateway.py`** — FastAPI proxy that exposes a single `/v1` surface over both backends. Routes by model name: `"command-r"` → port 8000, anything else → port 8001. Supports streaming (`stream: true`). Embeddings (`/v1/embeddings`) are handled locally via `sentence-transformers` on CPU — gte-small (33MB BERT, 384-dim) is loaded once at startup, no third vLLM server needed. Start with `uvicorn gateway:app`.

**`simulation/agent_roles.py`** — agent personas (`backend`, `system`), plus `ENGAGEMENT_ANCHORS` (gte-small vocab for scoring) and `INTERVENTION_CODEBOOK` (gte-small vocab for observer output extraction).

**`simulation/sim_loop.py`** — stratified tick architecture with closed inter-agent feedback loop:
- **Phase A** `PlatformAgent.decide()` — sequential per user; sees `addiction_score` + user's response to last content (closes feedback loop)
- **Phase B** `MediaUserAgent.step()` — reacts to platform content; concurrent via `asyncio.gather`
- **Phase C** `embed_score()` + `update_score()` — gte-small cosine similarity against `ENGAGEMENT_ANCHORS`; EMA dynamics with `alpha=0.2` (high inertia, calibrated to PSMU longitudinal autocorrelation r~0.75–0.85); thesis sensitivity parameter, sweep `[0.1, 0.4]`
- **Phase D** `ObserverAgent.observe()` — researcher + policy_analyst fire concurrently every `K=3` ticks on GPU0; interventions extracted via `INTERVENTION_CODEBOOK` similarity (threshold 0.72), not tag parsing
- `WorldState` holds typed interventions (`platform_constraint`, `user_nudge`, `score_modifier`), granular `full_log()` for thesis data export, and token-capped aggregate stats for LLM observer prompts
- `run_simulation(n_hours, n_users, k, alpha)` — `k` and `alpha` are configurable for sensitivity analysis

## Key constraints

- Run `orchestrator.health_check()` (or `start_all.sh`) before any simulation — both servers must be up.
- `max-model-len 8192` on both servers. Keep `history[-4:]` in `agent_turn` calls to stay within budget.
- `CUDA_VISIBLE_DEVICES` is set inside `start_gpu0.sh` / `start_gpu1.sh` — do not override globally.
- Never instantiate a new `httpx.AsyncClient` per request. The module-level client in `orchestrator.py` and `gateway.py` is intentional.
- Keep `max_tokens=256` for routine simulation steps; output length directly drives inference time.

## Efficiency design

- `--enable-prefix-caching`: agents of the same role share a system prompt → KV cache reuse across every turn (largest single latency saving).
- `--enable-chunked-prefill`: interleaves prefill/decode across concurrent requests.
- `--gpu-memory-utilization 0.93` / `--dtype auto` (bfloat16 on Blackwell): ~28.3 GB KV cache reserved per GPU.
- Phase A (platform decisions) is intentionally sequential — both models load from the same NFS volume; parallel reads cause I/O contention.
- Phase B (user reactions) uses `asyncio.gather` — do not reintroduce serial loops here.
- `PlatformAgent.batch_decide()` stub exists for N>10; implement cohort-level strategy when scaling.

## Planned upgrades (not yet implemented)

- Replace EMA score dynamics with a NumPyro Beta-distribution model (signal extraction via gte-small is already in place).
- FLAME GPU 2 for population-scale runs (current default is 4 users).
