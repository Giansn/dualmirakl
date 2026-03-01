# COWORK — Agent Cooperation Model

## Slots

### Authority — GPU 0, port 8000
**Role** — Primary reasoning and output model. Runs analyst agents (observer_a, observer_b).

**Function** — Structured reasoning, analysis, synthesis. Directs the swarm slot on what information is needed and how it should be structured. Receives filtered, ranked context from the swarm. Produces output and initiates the next retrieval cycle.

**Config** — `models/authority.env`

---

### Swarm — GPU 1, port 8001
**Role** — Generative and persona model. Runs participant and environment agents.

**Function** — High-throughput generation. Intermediary between raw information and the authority slot. Chunks and labels incoming documents. Filters and reranks retrieval results before forwarding. Manages file organisation.

**Config** — `models/swarm.env`

---

### Embedding — CPU (via gateway, port 9000)
**Role** — Pure embedding and retrieval layer.

**Function** — Converts text to vectors for storage and similarity search. Produces no text output. Controlled indirectly through what the swarm slot chooses to embed and query.

**Model** — e5-small-v2 (384-dim, ~2–5 ms per call)

---

## Cooperation Loop

```
authority  →  direction          →  swarm
swarm      →  prepared chunks    →  embedding
embedding  →  top-k results      →  swarm
swarm      →  filtered context   →  authority
authority  →  output + new need  →  swarm  [repeat]
```

## Backend Keys

| Key | Slot | Agents |
|-----|------|--------|
| `authority` | GPU 0 :8000 | observer_a, observer_b |
| `swarm` | GPU 1 :8001 | environment, participant |
| `e5-small-v2` | CPU :9000 | embedding only |
