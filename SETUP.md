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

| Model | Size | Location | Serves |
|---|---|---|---|
| Command-R 7B AWQ | ~4GB | `/per.volume/huggingface/hub/command-r7b-awq` | GPU0 :8000 |
| Qwen 2.5 7B AWQ | ~4GB | `/per.volume/huggingface/hub/qwen2.5-7b-awq` | GPU1 :8001 |
| gte-small | ~33MB | `/per.volume/huggingface/hub/gte-small` | CPU |

Both LLM servers run via vLLM with `--quantization awq`, `--dtype auto` (bfloat16), `--gpu-memory-utilization 0.93`, `--max-model-len 8192`, prefix caching and chunked prefill enabled.

---

## Role Architecture

Three models, three distinct roles. No overlap.

### Command-R — Authority / Front End
- The primary reasoning and output model
- Receives clean, pre-processed context — never raw data
- Performs RAG synthesis: formulates retrieval queries, receives filtered chunks, produces grounded output
- Directs Qwen on what it needs and how to prepare it
- Built specifically for RAG and grounded generation — citations, consistency, professional output

### Qwen — Middleware / Organiser
- Sits between raw data and Command-R
- Chunks and labels incoming documents and simulation output for embedding
- Receives direction from Command-R on what structure and labelling is needed
- After gte-small retrieves, Qwen reranks and filters results before forwarding to Command-R
- Manages file organisation on disk
- Handles throughput tasks — fast generation, reformatting, preprocessing

### gte-small — Index / Retrieval Layer
- Pure embedding model: input text → 384-dim vector. Produces no text output.
- Embeds all chunks prepared by Qwen into a vector index
- On retrieval query: finds top-k most similar chunks via cosine similarity
- Cannot be instructed — Qwen controls it by controlling what gets embedded and what query gets sent
- Keeps Command-R's context window manageable: large files never enter the context whole

---

## Communication Flow

```
Raw input (documents, simulation logs, external papers)
        │
        ▼
      Qwen  ── chunks, labels, structures per Command-R's direction
        │
        ▼
   gte-small ── embeds chunks → stored in vector index
        │
        │   (retrieval loop)
        │
Command-R ── formulates query based on current reasoning gap
        │
        ▼
   gte-small ── cosine similarity search → top-k chunks
        │
        ▼
      Qwen  ── reranks, filters, compresses retrieved chunks
        │
        ▼
Command-R ── receives clean relevant context → produces output
        │
        ▼
      Qwen  ── receives output → organises, files, prepares next batch
```

### Iterative loop
Command-R produces output → identifies what it still needs → directs Qwen → Qwen prepares and embeds → gte-small retrieves → Qwen filters → Command-R continues. Each cycle refines the output.

### Why embed before handing to Command-R
Large files exceed the 8192 token context limit immediately. RAG is the solution: chunk and embed first, then retrieve only the relevant pieces at query time. Command-R never sees a full document — it sees the 3–5 chunks most relevant to its current question (~1000 tokens). This applies to all inputs above a few hundred tokens.

---

## RAG Pipeline Detail

```
1. INGEST
   Raw file → Qwen chunks it (size TBD, ~512 tokens per chunk with overlap)
             → Qwen labels each chunk (topic, source, type)
             → gte-small embeds each chunk → stored in FAISS/ChromaDB index

2. QUERY
   Command-R formulates retrieval query (natural language)
   → gte-small embeds the query → cosine similarity against index
   → top-k chunks returned (k=5 default)

3. RERANK
   Qwen receives top-k chunks
   → scores by relevance to Command-R's current task
   → removes redundant or low-signal chunks
   → forwards top 3 to Command-R

4. GENERATE
   Command-R receives: system prompt + retrieved chunks + query
   → produces grounded output with source references
   → identifies gaps → loop back to step 2 if needed
```

---

## Simulation Architecture (sim_loop.py)

The simulation models media engagement dynamics via a stratified tick loop.

### Agents

| Class | Model | Role |
|---|---|---|
| `PlatformAgent` | Qwen | Decides content per user each tick |
| `MediaUserAgent` | Qwen | Responds to content served |
| `ObserverAgent` (researcher) | Command-R | Analyses dynamics, proposes interventions |
| `ObserverAgent` (policy_analyst) | Command-R | Proposes policy interventions |

### Tick structure (per simulated hour)

```
Phase A — PlatformAgent.decide()       sequential, one call per user
          Input:  user addiction_score + response to last content
          Output: content decision for this tick
          Model:  Qwen, GPU1

Phase B — MediaUserAgent.step()        concurrent (asyncio.gather)
          Input:  platform content from Phase A
          Output: user response
          Model:  Qwen, GPU1

Phase C — embed_score() + update_score()   no LLM, CPU only
          gte-small embeds user response
          cosine similarity vs ENGAGEMENT_ANCHORS
          EMA update: score = alpha * signal + (1-alpha) * current_score
          alpha=0.2 (high inertia, calibrated to PSMU longitudinal r~0.75-0.85)

Phase D — ObserverAgent.observe()      concurrent (asyncio.gather), every K=3 ticks
          Input:  K-tick observation window from WorldState
          Output: analysis + intervention recommendations
          Model:  Command-R, GPU0
          Intervention extraction: gte-small embeds output,
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
observer output     →  gte-small extraction → typed interventions
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

- **Vector store**: FAISS or ChromaDB for persistent RAG index (gte-small embeddings)
- **Qwen reranker**: implement chunk reranking between gte-small retrieval and Command-R context injection
- **NumPyro score model**: replace EMA with Beta-distribution probabilistic dynamics
- **FLAME GPU 2**: population-scale simulation runs
- **`batch_decide()`**: cohort-level platform strategy for N>10 users
