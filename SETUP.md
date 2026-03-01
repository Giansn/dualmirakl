# SETUP.md

Full architecture reference for the dualmirakl stack.
Covers hardware, model roles, communication design, RAG pipeline, and simulation architecture.

---

## Hardware

| | |
|---|---|
| Pod | RunPod, 2x RTX PRO 4500 Blackwell |
| VRAM | 32GB per GPU |
| CUDA | 12.8 |
| PyTorch | 2.9.1+cu128 |
| Persistent volume | `/per.volume` |

---

## Models

| Model | Location | Serves |
|---|---|---|
| GLM-5 | `/per.volume/huggingface/hub/glm-5` | GPU0 :8000 |
| DeepSeek-V3.2-Special | `/per.volume/huggingface/hub/deepseek-v3.2-special` | GPU1 :8001 |
| e5-small-v2 | `/per.volume/huggingface/hub/e5-small-v2` | CPU |

Both LLM servers run via vLLM with `--quantization awq`, `--dtype auto` (bfloat16), `--gpu-memory-utilization 0.93`, `--max-model-len 8192`, prefix caching and chunked prefill enabled.

---

## Role Architecture

Three models, three distinct roles. No overlap.

### GLM-5 — Authority / Front End
- The primary reasoning and output model
- Receives clean, pre-processed context — never raw data
- Performs RAG synthesis: formulates retrieval queries, receives filtered chunks, produces grounded output
- Directs DeepSeek on what it needs and how to prepare it

### DeepSeek-V3.2-Special — Middleware / Organiser
- Sits between raw data and GLM-5
- Chunks and labels incoming documents and simulation output for embedding
- Receives direction from GLM-5 on what structure and labelling is needed
- After e5-small-v2 retrieves, DeepSeek reranks and filters results before forwarding to GLM-5
- Manages file organisation on disk
- Handles throughput tasks — fast generation, reformatting, preprocessing

### e5-small-v2 — Index / Retrieval Layer
- Pure embedding model: input text → 384-dim vector. Produces no text output.
- Embeds all chunks prepared by DeepSeek into a vector index
- On retrieval query: finds top-k most similar chunks via cosine similarity
- Cannot be instructed — DeepSeek controls it by controlling what gets embedded and what query gets sent
- Keeps GLM-5's context window manageable: large files never enter the context whole

---

## Communication Flow

```
Raw input (documents, simulation logs, external papers)
        │
        ▼
   DeepSeek  ── chunks, labels, structures per GLM-5's direction
        │
        ▼
 e5-small-v2 ── embeds chunks → stored in vector index
        │
        │   (retrieval loop)
        │
    GLM-5    ── formulates query based on current reasoning gap
        │
        ▼
 e5-small-v2 ── cosine similarity search → top-k chunks
        │
        ▼
   DeepSeek  ── reranks, filters, compresses retrieved chunks
        │
        ▼
    GLM-5    ── receives clean relevant context → produces output
        │
        ▼
   DeepSeek  ── receives output → organises, files, prepares next batch
```

### Iterative loop
GLM-5 produces output → identifies what it still needs → directs DeepSeek → DeepSeek prepares and embeds → e5-small-v2 retrieves → DeepSeek filters → GLM-5 continues. Each cycle refines the output.

### Why embed before handing to GLM-5
Large files exceed the 8192 token context limit immediately. RAG is the solution: chunk and embed first, then retrieve only the relevant pieces at query time. GLM-5 never sees a full document — it sees the 3–5 chunks most relevant to its current question (~1000 tokens). This applies to all inputs above a few hundred tokens.

---

## RAG Pipeline Detail

```
1. INGEST
   Raw file → DeepSeek chunks it (size TBD, ~512 tokens per chunk with overlap)
            → DeepSeek labels each chunk (topic, source, type)
            → e5-small-v2 embeds each chunk → stored in FAISS/ChromaDB index

2. QUERY
   GLM-5 formulates retrieval query (natural language)
   → e5-small-v2 embeds the query → cosine similarity against index
   → top-k chunks returned (k=5 default)

3. RERANK
   DeepSeek receives top-k chunks
   → scores by relevance to GLM-5's current task
   → removes redundant or low-signal chunks
   → forwards top 3 to GLM-5

4. GENERATE
   GLM-5 receives: system prompt + retrieved chunks + query
   → produces grounded output with source references
   → identifies gaps → loop back to step 2 if needed
```

---

## Simulation Architecture (sim_loop.py)

The simulation models media engagement dynamics via a stratified tick loop.

### Agents

| Class | Model | Role |
|---|---|---|
| `PlatformAgent` | DeepSeek-V3.2-Special | Decides content per user each tick |
| `MediaUserAgent` | DeepSeek-V3.2-Special | Responds to content served |
| `ObserverAgent` (researcher) | GLM-5 | Analyses dynamics, proposes interventions |
| `ObserverAgent` (policy_analyst) | GLM-5 | Proposes policy interventions |

### Tick structure (per simulated hour)

```
Phase A — PlatformAgent.decide()       sequential, one call per user
          Input:  user addiction_score + response to last content
          Output: content decision for this tick
          Model:  DeepSeek-V3.2-Special, GPU1

Phase B — MediaUserAgent.step()        concurrent (asyncio.gather)
          Input:  platform content from Phase A
          Output: user response
          Model:  DeepSeek-V3.2-Special, GPU1

Phase C — embed_score() + update_score()   no LLM, CPU only
          e5-small-v2 embeds user response
          cosine similarity vs ENGAGEMENT_ANCHORS
          EMA update: score = alpha * signal + (1-alpha) * current_score
          alpha=0.2 (high inertia, calibrated to PSMU longitudinal r~0.75-0.85)

Phase D — ObserverAgent.observe()      concurrent (asyncio.gather), every K=3 ticks
          Input:  K-tick observation window from WorldState
          Output: analysis + intervention recommendations
          Model:  GLM-5, GPU0
          Intervention extraction: e5-small-v2 embeds output,
          cosine similarity vs INTERVENTION_CODEBOOK (threshold 0.72)
```

### Key parameters (thesis sensitivity variables)
- `alpha` — EMA responsiveness, default 0.2, sweep `[0.1, 0.4]`
- `k` — observer frequency in ticks, default 3, configurable
- `INTERVENTION_THRESHOLD` — codebook match threshold, default 0.72

### Inter-agent data flow
```
platform_ai output  →  media_user input (Phase A → B)
media_user output   →  platform_ai next tick (Phase B → A, closes feedback loop)
media_user output   →  WorldState observation log
WorldState log      →  observer context (Phase D)
observer output     →  e5-small-v2 extraction → typed interventions
interventions       →  Phase A + B prompts next tick
```

### Intervention types
| Type | Effect |
|---|---|
| `platform_constraint` | Injected into platform_ai prompt |
| `user_nudge` | Injected into media_user prompt |
| `score_modifier` | Applies dampening coefficient to score update |

---

## Startup Sequence

```bash
cd /per.volume/dualmirakl

# 1. Start LLM servers (sequential — shared NFS volume, parallel reads cause I/O contention)
bash start_all.sh
# Preflight: checks httpx, h2, simpy
# Port collision check: 8000, 8001
# Launches GPU0, polls completions endpoint (warmup probe, not just /health)
# Only after GPU0 ready: launches GPU1, same probe
# Timeout: 3 min per server. On failure: nvidia-smi dump + kill + exit 1
# PIDs saved to logs/pids.txt

# 2. Verify
python orchestrator.py

# 3. Run simulation
python simulation/sim_loop.py

# 4. Optional: start unified gateway
uvicorn gateway:app --host 0.0.0.0 --port 9000

# Stop
bash stop_all.sh
```

Logs: `logs/gpu0.log`, `logs/gpu1.log` (rotated to `.last` on each start).

---

## Key Constraints

- `max-model-len 8192` on both servers — keep injected context within budget
- `CUDA_VISIBLE_DEVICES` set per-script — do not override globally
- Never create a new `httpx.AsyncClient` per request — use the module-level client in `orchestrator.py`
- `max_tokens=256` default for simulation steps — output length drives inference time
- Phase A is sequential by design — do not parallelise (NFS I/O contention)
- Phase B uses `asyncio.gather` — do not reintroduce serial loops
- `PlatformAgent.batch_decide()` stub exists for N>10 users — cohort-level strategy, not yet implemented

---

## Planned Upgrades

- **Vector store**: FAISS or ChromaDB for persistent RAG index (e5-small-v2 embeddings)
- **DeepSeek reranker**: implement chunk reranking between e5-small-v2 retrieval and GLM-5 context injection
- **NumPyro score model**: replace EMA with Beta-distribution probabilistic dynamics
- **FLAME GPU 2**: population-scale simulation runs
- **`batch_decide()`**: cohort-level platform strategy for N>10 users
