# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
bash start_authority.sh   # GPU 0, port 8000 — set MODEL in models/authority.env first
bash start_swarm.sh       # GPU 1, port 8001 — nemotron-nano-30b, ready
bash start_gateway.sh     # CPU, port 9000 — embedding + unified proxy
bash start_all.sh         # all three with health polling
bash stop_all.sh          # kill all vLLM processes
bash audit_env.sh         # check GPU, CUDA, model files, ports
python orchestrator.py    # dual health check + test query
python simulation/sim_loop.py  # run simulation
```

Logs: `logs/authority.log`, `logs/swarm.log`, `logs/gateway.log`

## Architecture

```
GPU 0 — authority slot  :8000   (analyst agents: observer_a, observer_b)
GPU 1 — swarm slot      :8001   (persona agents: environment, participant)
CPU   — embedding       :9000   (e5-small-v2, via gateway.py)
```

### Model Slots

Model configs live in `models/authority.env` and `models/swarm.env`.
To swap a model: edit `MODEL` and `QUANT_FLAGS` in the relevant `.env` file.
The served name is always `authority` or `swarm` — no other code changes needed.

### Three-Layer Cooperation

- **Authority** — primary reasoning model; receives pre-processed context; directs swarm on what it needs.
- **Swarm** — generative/persona model; chunks/labels raw data per authority's direction; reranks retrieval results; handles file organisation.
- **Embedding (e5-small-v2)** — pure embedding layer (384-dim, CPU); produces no text; controlled indirectly by what swarm embeds and queries.

Cooperation loop: `authority → direction → swarm → chunks → e5-small-v2 → top-k → swarm → filtered context → authority → output + new need → [repeat]`

**RAG pipeline** (planned, per `SETUP.md`): ingest (swarm chunks ~512 tokens → e5-small-v2 embeds → FAISS/ChromaDB) → query (authority natural-language query → e5-small-v2 similarity → top-k=5) → rerank (swarm filters to top 3, ~1000 tokens) → generate (authority produces grounded output, loops if gaps remain).

## Key Files

- `orchestrator.py` — `chat(backend, messages)`: POST to one backend; backend is `"authority"` or `"swarm"`. `dual_query(prompt)`: concurrent query to both. Module-level `_client` (httpx, HTTP/2) — never instantiate per request.
- `gateway.py` — FastAPI; routes by model name (`"authority"` → GPU0, else GPU1); `/v1/embeddings` served locally via e5-small-v2.
- `simulation/agent_rolesv3.py` — `BACKEND_CONFIG` maps `"analyst"` → `"authority"`, `"persona"` → `"swarm"`. Override at runtime: `BACKEND_ANALYST=authority BACKEND_PERSONA=swarm`.
- `models/authority.env` — authority slot: MODEL path, QUANT_FLAGS, MAX_MODEL_LEN, EXTRA_FLAGS.
- `models/swarm.env` — swarm slot: nemotron-nano-30b NVFP4 config with exact HF-recommended env vars and flags.

## vLLM Notes

- `CUDA_VISIBLE_DEVICES` is set inside each start script — do not override globally.
- Nemotron (swarm): NVFP4 quantization auto-detected from `hf_quant_config.json`; requires `VLLM_USE_FLASHINFER_MOE_FP4=1` (set in `models/swarm.env`); uses custom reasoning parser `nano_v3_reasoning_parser.py` (ships with model weights).
- Swarm served on port 8001 with `--served-model-name swarm`; authority on 8000 as `authority`.
- `max-model-len 32768` for both slots (VRAM budget); increase in `.env` files if needed.

## Planned

- **Swarm reranker**: chunk reranking between e5-small-v2 retrieval and authority context injection.
- **FLAME GPU 2**: population-scale simulation runs.
- **NumPyro state model**: probabilistic score transitions.
