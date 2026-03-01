# COWORK.md
## Model Cooperation Reference

---

### GLM-5 — Authority

**Function** — Primary reasoning and output model. Formulates retrieval queries, synthesises retrieved context into coherent, grounded output. Identifies gaps and directs the next preparation cycle.

**Skills** — RAG synthesis · grounded generation · source citation · long-context consistency · professional output

**Interaction** — Directs DeepSeek on what information is needed and how it should be structured. Receives filtered, ranked context from DeepSeek. Produces output and initiates the next retrieval cycle.

---

### DeepSeek-V3.2-Special — Middleware

**Function** — Intermediary between raw information and GLM-5. Chunks and labels all incoming documents per GLM-5's direction. Filters and reranks retrieval results before forwarding. Manages file organisation.

**Skills** — Chunking · labelling · preprocessing · reranking · file management · high-throughput generation

**Interaction** — Receives direction from GLM-5. Prepares and routes material to e5-small-v2 for embedding. Receives raw retrieval results from e5-small-v2, filters them, and passes a clean shortlist to GLM-5.

---

### e5-small-v2 (384-dim) — Index

**Function** — Pure embedding and retrieval layer. Converts text to vectors for storage and similarity search. Produces no text output. Controlled indirectly through what DeepSeek chooses to embed and query.

**Skills** — Text embedding · cosine similarity · vector retrieval · context compression

**Interaction** — Receives prepared chunks from DeepSeek for indexing. Returns top-k similar chunks to DeepSeek on query. Keeps GLM-5's context window within token budget by enabling selective retrieval over large corpora.

---

### Cooperation Flow

```
GLM-5       →  direction          →  DeepSeek
DeepSeek    →  prepared chunks    →  e5-small-v2
e5-small-v2 →  top-k results      →  DeepSeek
DeepSeek    →  filtered context   →  GLM-5
GLM-5       →  output + new need  →  DeepSeek  [repeat]
```
