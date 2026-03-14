# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
bash start_authority.sh        # GPU 0, port 8000
bash start_swarm.sh            # GPU 1, port 8001
bash start_gateway.sh          # CPU, port 9000 — embedding + proxy + web UI
bash start_all.sh              # all three with health polling
bash stop_all.sh               # kill all vLLM processes
bash pull_models.sh            # download models to $HF_HOME via huggingface-cli
bash chmod.sh                  # set +x on all scripts
bash audit_env.sh              # check GPU, CUDA, model files, ports
python orchestrator.py         # dual health check + test query
python -m simulation.sim_loop  # run simulation (interactive CLI)
python -m pytest tests/ -v     # run test suite (29 tests, no vLLM needed)
```

Logs: `logs/authority.log`, `logs/swarm.log`, `logs/gateway.log`
Sim output: `data/{run_id}/` (config, trajectories, observations, compliance, interventions)

## Architecture

```
GPU 0 — authority slot  :8000   (environment + observer_a + observer_b)
GPU 1 — swarm slot      :8001   (participant agents)
CPU   — gateway         :9000   (e5-small-v2 embedding + proxy + web UI)
```

### GPU Load Balance

Environment routes to authority (not swarm) to balance GPU utilization.
Per 12-tick run (N=4, K=4): authority ~56 calls, swarm ~48 calls (~1.2:1 ratio).

Three backend keys in `BACKEND_CONFIG` (agent_rolesv3.py):
- `analyst` → authority (observer_a, observer_b)
- `environment` → authority (environment agent — DDA-aware stimulus generation)
- `persona` → swarm (participant agents)

Override at runtime: `BACKEND_ENVIRONMENT=swarm python -m simulation.sim_loop`

### Model Slots

Model configs live in `models/authority.env` and `models/swarm.env`.
To swap a model: edit `MODEL` and `QUANT_FLAGS` in the relevant `.env` file.
The served name is always `authority` or `swarm` — no other code changes needed.

### Three-Layer Cooperation

- **Authority** — reasoning model; environment stimulus generation, observer analysis/intervention.
- **Swarm** — persona model; participant responses, persona summaries, compression (chat UI).
- **Embedding (e5-small-v2)** — 384-dim CPU embedding; behavioral scoring, RAG retrieval.

## Simulation (sim_loop.py)

Four-phase tick architecture:
- **Phase A** (sequential): environment generates stimuli per participant → authority
- **Phase B** (concurrent): participants respond via asyncio.gather → swarm
- **Phase C** (CPU): batch embedding → EMA score update (e5-small-v2, no GPU)
- **Phase D** (every K ticks, sequential): observer_a analyses → observer_b intervenes → authority

Key defaults (tuned for throughput): `max_tokens=192`, `alpha=0.15`, `K=4`, `history_window=4`.
All params configurable via `SIM_*` env vars in `.env` or interactive CLI prompts.

Data export: results auto-saved to `data/{run_id}/` as JSON (config, trajectories, observations, compliance, interventions).

## Key Files

- `orchestrator.py` — `agent_turn(agent_id, backend, system_prompt, user_message, history, max_tokens)`: single agent turn via httpx HTTP/2 client. `chat()`, `dual_query()`, `health_check()`.
- `gateway.py` — FastAPI; serves web UI at `/`; routes `/v1/chat/completions` by model name; `/v1/embeddings` via CPU e5-small-v2. CORS enabled.
- `interface.html` — single-file chat UI with RAG, compression, streaming, PDF+OCR, simulation panel. Served by gateway at root.
- `simulation/sim_loop.py` — complete v3 simulation loop with all fixes (sequential observers, batch embedding, persona summaries, compliance checks, data export, sensitivity analysis).
- `simulation/agent_rolesv3.py` — agent role definitions, `BACKEND_CONFIG`, engagement anchors, intervention codebook, compliance patterns. English + German variants.
- `models/authority.env` — authority slot: MODEL, QUANT_FLAGS, MAX_MODEL_LEN=8192, MAX_NUM_SEQS=12, prefix caching + chunked prefill.
- `models/swarm.env` — swarm slot: nemotron-nano-30b NVFP4, MAX_MODEL_LEN=8192, MAX_NUM_SEQS=12.
- `tests/test_sim_loop.py` — 29 tests + duration estimator. Run without vLLM servers.

## Deployment

**RunPod**: Docker image `giansn/dualmirakl:runpod-cu128`. Post-start hook clones repo to `/per.volume/dualmirakl/`, runs `autostart.sh`. Models on persistent volume.

**Docker Compose**: `docker compose up -d` starts authority/swarm/gateway services. `docker compose run sim` runs simulation. Models mounted via `models_cache` volume.

**Entrypoint modes**: Set `ENTRYPOINT_MODE` env var: `all` (default), `authority`, `swarm`, `gateway`, `sim`, `shell`.

## vLLM Notes

- `CUDA_VISIBLE_DEVICES` set per start script — do not override globally.
- Nemotron (swarm): NVFP4 auto-detected; requires `VLLM_USE_FLASHINFER_MOE_FP4=1`.
- Both slots: prefix caching enabled (KV reuse for shared system prompts), chunked prefill enabled, `gpu-memory-utilization=0.90`.
- Ports configurable via `AUTHORITY_PORT`, `SWARM_PORT`, `GATEWAY_PORT` env vars.

## Planned

- **Swarm reranker**: chunk reranking between e5-small-v2 retrieval and authority context injection.
- **FLAME GPU 2**: population-scale simulation runs.
- **NumPyro state model**: probabilistic score transitions.
