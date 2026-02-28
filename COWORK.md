# COWORK.md
## Model Cooperation Reference

---

### Command-R 7B — Authority

**Function** — Primary reasoning and output model. Formulates retrieval queries, synthesises retrieved context into coherent, grounded output. Identifies gaps and directs the next preparation cycle.

**Skills** — RAG synthesis · grounded generation · source citation · long-context consistency · professional output

**Interaction** — Directs Qwen on what information is needed and how it should be structured. Receives filtered, ranked context from Qwen. Produces output and initiates the next retrieval cycle.

---

### Qwen 2.5 7B — Middleware

**Function** — Intermediary between raw information and Command-R. Chunks and labels all incoming documents per Command-R's direction. Filters and reranks retrieval results before forwarding. Manages file organisation.

**Skills** — Chunking · labelling · preprocessing · reranking · file management · high-throughput generation

**Interaction** — Receives direction from Command-R. Prepares and routes material to gte-small for embedding. Receives raw retrieval results from gte-small, filters them, and passes a clean shortlist to Command-R.

---

### gte-small (BERT 384-dim) — Index

**Function** — Pure embedding and retrieval layer. Converts text to vectors for storage and similarity search. Produces no text output. Controlled indirectly through what Qwen chooses to embed and query.

**Skills** — Text embedding · cosine similarity · vector retrieval · context compression

**Interaction** — Receives prepared chunks from Qwen for indexing. Returns top-k similar chunks to Qwen on query. Keeps Command-R's context window within token budget by enabling selective retrieval over large corpora.

---

### Cooperation Flow

```
Command-R  →  direction          →  Qwen
Qwen       →  prepared chunks    →  gte-small
gte-small  →  top-k results      →  Qwen
Qwen       →  filtered context   →  Command-R
Command-R  →  output + new need  →  Qwen  [repeat]
```
