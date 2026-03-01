# SETUP — Environment and Model Configuration

## Hardware

```
RunPod Pod
├── GPU 0 — RTX PRO 4500 Blackwell, 32 GB VRAM  →  authority slot :8000
├── GPU 1 — RTX PRO 4500 Blackwell, 32 GB VRAM  →  swarm slot     :8001
└── CPU                                          →  embedding      :9000
```

## Model Slots

| Slot | Current Model | Path | GPU | Port |
|------|--------------|------|-----|------|
| authority | *(not set)* | set in `models/authority.env` | 0 | 8000 |
| swarm | nemotron-nano-30b (NVFP4) | `hub/nemotron-nano-30b` | 1 | 8001 |
| embedding | e5-small-v2 | `hub/e5-small-v2` | CPU | — |

## Swapping a Model

1. Edit `models/<slot>.env` — update `MODEL` path and `QUANT_FLAGS`
2. Run `bash start_<slot>.sh` — everything else picks up automatically

No other files need changing. The served model name is always `authority` or `swarm`.

## Slot Roles

### Authority — Analyst / Reasoning
- Runs observer_a (analysis) and observer_b (interventions)
- Should be a strong reasoning model
- VRAM budget: ~16 GB (AWQ) to ~32 GB (BF16+INT4)

### Swarm — Generative / Persona
- Runs environment and participant agents
- Should be fast and high-throughput
- Current: nemotron-nano-30b NVFP4 (~15 GB, 3B active params)
- NVFP4 quantization auto-detected — no `--quantization` flag needed
- Requires: `VLLM_USE_FLASHINFER_MOE_FP4=1` (set in swarm.env)

### Embedding — e5-small-v2 (CPU)
- 384-dim sentence embeddings, cosine similarity
- Loaded at gateway startup, ~2–5 ms per call
- Used by sim_loop.py for behavioral signal scoring

## Agent-to-Slot Mapping

| Agent | Slot | Role |
|-------|------|------|
| `observer_a` | authority | Analysis only — no interventions |
| `observer_b` | authority | Proposes interventions from observer_a output |
| `environment` | swarm | Generates scenario events per tick |
| `participant` | swarm | Persona agent responding to events |

## RAG Pipeline (planned)

```
Ingest:   swarm chunks (~512 tokens) → e5-small-v2 embeds → FAISS/ChromaDB
Query:    authority formulates natural-language query
          → e5-small-v2 similarity search → top-k=5
Rerank:   swarm filters to top 3 (~1000 tokens)
Generate: authority receives clean context → produces output → loops if gaps remain
```

Large inputs must never be passed whole to authority — use RAG above a few hundred tokens.

## Planned Improvements

- **Swarm reranker**: implement chunk reranking between e5-small-v2 retrieval and authority context injection
- **FLAME GPU 2**: population-scale runs once single-agent loop is validated
- **NumPyro state model**: probabilistic score transitions calibrated on pilot data
