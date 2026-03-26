# ADR-001: GraphRAG, Memory Persistence, Persona Pipeline, Post-Sim ReACT

**Status:** Proposed
**Date:** 2026-03-25
**Deciders:** Giansn

## Context

dualmirakl runs structured multi-agent simulations on a dual-GPU stack (authority + swarm) with an async orchestrator, SimPy tick loop, and a FastAPI gateway. Four enhancement areas have been identified, inspired by MiroFish's architecture but adapted to dualmirakl's local-first, GPU-bound design.

**Current state of the relevant subsystems:**

| Subsystem | What exists | What's missing |
|-----------|-------------|----------------|
| Document ingestion | `gateway.py` chunks text, embeds via e5-small-v2, stores flat JSON in `world_context.json`. Summary (first 3000 chars) injected into agent prompts. Semantic search via `POST /v1/documents/query`. | No entity/relation extraction. No structured knowledge representation from documents. Retrieval is chunk-level only. |
| Graph memory | `graph_memory.py` — pure-Python directed graph (NetworkX-style). Distills simulation events (scores, responses, interventions) into nodes/edges with temporal metadata. Queryable by ReACT observer. | Graph is runtime-only (sim events). No document-derived entities. No persistence across runs. No embedding-based graph retrieval. |
| Agent memory | `agent_memory.py` — per-agent in-memory store with semantic retrieval (e5-small-v2), dedup (cosine > 0.9), LRU eviction, structured output integration. Exported as JSON. | No persistence across simulation runs. No cross-run memory retrieval. No memory consolidation/compression. |
| Ontology generator | `ontology_generator.py` — LLM-driven archetype + transition rule generation from domain documents via authority slot. Output merges into `ScenarioConfig`. | No connection to agent persona generation. Generates archetypes (K1, K2, K3) but not individual 6-component personas. |
| ReACT observer | `react_observer.py` — multi-step reasoning loop with 6 tools (query_scores, query_events, check_interventions, query_memory, interview_agent, query_graph). Drop-in for ObserverAgent in Phase D1. | Intra-simulation only. No post-simulation analysis mode. No access to dynamics analysis results or cross-run comparisons. |
| Storage | All JSON files. `data/{run_id}/*.json` for simulation output. `context/world_context.json` for documents. Optional SQLite via Optuna. | No DuckDB. No relational storage. No vector index beyond flat numpy arrays. |

**Constraints:**

- 2x RTX PRO 4500 Blackwell (32 GB VRAM each) — authority on GPU 0, swarm on GPU 1
- e5-small-v2 on CPU (~2-5ms per call, 384-dim vectors)
- All inference must stay local — no cloud LLM APIs
- Docker-deployed on RunPod — new dependencies must fit in the existing image layer strategy
- 508 existing tests — changes must not break the test suite

## Decision

Implement four enhancements in dependency order. Each is designed as an additive module with backward-compatible integration points — existing behavior is preserved when features are not enabled.

---

## Options Considered

### Enhancement 1: GraphRAG — Document → Knowledge Graph → Context Injection

Three options for extracting structured knowledge from uploaded documents and making it available to the simulation.

#### Option A: DuckDB + Authority-Slot Entity Extraction

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| New dependencies | `duckdb` (~15 MB, embedded, pip install) |
| LLM cost per doc | ~1 authority call per 5 chunks (batch extraction) |
| Storage overhead | Negligible (DuckDB file, typically < 10 MB) |
| Team familiarity | DuckDB is new to the stack |

**Approach:**
- New module `simulation/graph_rag.py` with `extract_entities(chunks: list[str]) -> list[Entity]` and `extract_relations(entities, chunks) -> list[Relation]`
- Authority slot processes chunks in batches of ~5 with JSON-mode structured output: `{entities: [{name, type, properties}], relations: [{source, target, type, context}]}`
- DuckDB tables: `entities(id, name, type, embedding FLOAT[384], properties JSON)`, `relations(id, src_entity, tgt_entity, rel_type, weight, context, embedding FLOAT[384])`
- Gateway extension: `POST /v1/documents` gains `?extract_graph=true` parameter → triggers async extraction pipeline after chunking/embedding
- Context injection: `sim_loop.py` loads graph triples at startup, filters by cosine similarity to scenario context, injects as `[GRAPH CONTEXT]\nSubject → predicate → Object\n...` alongside existing `world_context_summary`
- `graph_memory.py` gets an optional `seed_from_graphrag(entities, relations)` method to pre-populate the runtime graph before tick 0

**Pros:**
- DuckDB has native `FLOAT[384]` array type — vector operations without external index
- Embedded DB, zero infrastructure overhead, transactional
- Authority slot is underutilized during document upload (no sim running) — free compute
- Entity extraction prompt is a single, testable unit
- Existing flat-chunk retrieval stays parallel (dual retrieval: chunks + triples)

**Cons:**
- DuckDB is a new dependency (though small and self-contained)
- Entity extraction quality depends on authority model (Mistral Nemo 12B) — may need prompt iteration
- First extraction adds latency to document upload (~10-30s for a typical document)

#### Option B: Extend Existing graph_memory.py with Document Parsing (No DuckDB)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| New dependencies | None |
| LLM cost per doc | Same as A |
| Storage overhead | In-memory only (lost on restart) |
| Team familiarity | High (extends existing code) |

**Approach:**
- Add `ingest_document(chunks)` to `GraphMemory` that uses authority slot for extraction
- Store entities/relations as NetworkX-style nodes/edges in the existing in-memory graph
- Export to JSON alongside simulation data
- Re-ingest from JSON on next run if needed

**Pros:**
- Zero new dependencies
- Reuses existing graph_memory.py patterns
- Simpler implementation

**Cons:**
- No persistence — graph lost when process ends, must re-extract
- No vector indexing on entities — can't do embedding-based graph retrieval efficiently
- Mixes document-derived knowledge with sim-event-derived knowledge in the same structure (namespace collision risk)
- Can't query the graph outside of a running simulation
- Doesn't scale to multiple documents well (all in-memory)

#### Option C: KuzuDB (Embedded Graph DB)

| Dimension | Assessment |
|-----------|------------|
| Complexity | High |
| New dependencies | `kuzu` (~50 MB) |
| LLM cost per doc | Same as A |
| Storage overhead | KuzuDB directory |
| Team familiarity | Low |

**Approach:**
- Use KuzuDB (embedded graph database) as referenced in the amadad MiroFish fork
- Native graph queries (Cypher-like syntax)
- Store entities, relations, and temporal edges natively

**Pros:**
- Purpose-built for graph operations (path queries, pattern matching)
- Proper graph query language
- Persistent by default

**Cons:**
- Heavier dependency than DuckDB
- Overkill for the current graph size (typically < 1000 nodes)
- KuzuDB doesn't have native vector operations — would need DuckDB or numpy alongside it anyway for embedding retrieval
- Two storage engines to maintain

**Recommendation: Option A (DuckDB + Authority-Slot Extraction)**

DuckDB provides the best balance: single dependency that covers both relational storage (entities, relations, memories) and vector operations (native FLOAT arrays with cosine similarity). KuzuDB adds a second storage engine without sufficient benefit at current scale. Option B lacks persistence and vector indexing.

---

### Enhancement 2: Agent Memory Persistence

Three options for persisting agent memories across simulation runs.

#### Option A: DuckDB Backend (shared with GraphRAG)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| New dependencies | None (reuses DuckDB from Enhancement 1) |
| Storage overhead | ~1 KB per memory entry (content + 384-dim embedding) |
| Performance impact | Write-behind on tick boundary, async batch insert |

**Approach:**
- New class `DuckDBMemoryBackend` in `agent_memory.py` implementing the same interface
- Schema: `agent_memories(id VARCHAR, run_id VARCHAR, agent_id VARCHAR, tick INTEGER, memory_type VARCHAR, title VARCHAR, content VARCHAR, tags VARCHAR[], embedding FLOAT[384], importance FLOAT, created_at TIMESTAMP, decay_rate FLOAT DEFAULT 0.05)`
- In-memory `AgentMemoryStore` remains the hot path during simulation (no latency change)
- After each tick: batch-insert new memories to DuckDB in a background coroutine
- At simulation end: flush all memories
- `run_simulation()` gains optional `--continue-from {run_id}` parameter
- When continuing: load top-K memories per agent from prior run, filtered by cosine similarity to new scenario context, weighted by `importance × exp(-decay_rate × ticks_since_creation)`
- Memory distillation: every N ticks (configurable), authority slot summarizes episodic memories into semantic memories (compression)

**Pros:**
- Single storage layer (DuckDB) for both graph and memory
- In-memory hot path unchanged — zero latency overhead during sim
- Natural cross-run querying: "what did participant_0 remember across all runs?"
- Decay function prevents unbounded growth without deleting data

**Cons:**
- Adds complexity to AgentMemoryStore (two backends)
- Memory distillation uses authority slot compute (1 call per agent per N ticks)
- Schema needs migration strategy if structure changes

#### Option B: JSON-File Persistence (Extend Current Export)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| New dependencies | None |
| Storage overhead | ~5 KB per agent per run (JSON with base64-encoded embeddings) |
| Performance impact | None during sim, small at export |

**Approach:**
- Extend existing `export_results()` to write `agent_memories.json` with full embedding data (already partially done — embeddings are stripped in current export)
- `run_simulation(continue_from="data/run_20260325_*/")` loads the JSON and rehydrates AgentMemoryStore
- No DuckDB involvement

**Pros:**
- Minimal change
- Already partially implemented (export exists, just needs embedding serialization)
- No new dependencies

**Cons:**
- JSON files with 384-dim float arrays are large and slow to parse
- No efficient querying across runs (would need to load all JSON files)
- No decay/importance scoring without loading everything into memory
- No vector search across persisted memories — must load all into numpy first

#### Option C: SQLite with sqlite-vec Extension

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| New dependencies | `sqlite-vec` |
| Storage overhead | Similar to DuckDB |
| Performance impact | Similar to DuckDB |

**Approach:**
- Use SQLite (already present via Optuna) with the `sqlite-vec` extension for vector operations
- Similar schema to Option A but in SQLite

**Pros:**
- SQLite already in the stack (Optuna)
- sqlite-vec provides proper vector indexing

**Cons:**
- sqlite-vec is a C extension — platform-specific builds, Docker layer considerations
- Less native array support than DuckDB (FLOAT[384] vs blob encoding)
- Two storage systems if DuckDB is used for GraphRAG
- DuckDB's analytical query performance is significantly better for batch operations

**Recommendation: Option A (DuckDB Backend)**

Reusing DuckDB from Enhancement 1 avoids a second storage engine. JSON persistence (Option B) doesn't support efficient cross-run queries. SQLite-vec (Option C) adds platform-specific build complexity and would mean two storage engines alongside DuckDB.

---

### Enhancement 3: Seed → Persona Pipeline

Two options for connecting the knowledge graph to automatic persona generation.

#### Option A: Extend ontology_generator.py with generate_personas()

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| New dependencies | None |
| LLM cost | 1 authority call per batch of personas |
| Integration points | ontology_generator.py, scenario.py, sim_loop.py |

**Approach:**
- New function `generate_personas(graph_entities, archetypes, n_personas, scenario_config) -> list[PersonaSpec]` in `ontology_generator.py`
- `PersonaSpec` is a dataclass with the 6 persona components matching `agent_rolesv3.py`'s PARTICIPANT_TEMPLATE: `identity`, `behavior_rules`, `emotional_range`, `knowledge_bounds`, `consistency_rules`, `hard_limits`
- Authority slot receives: KG entities + archetype definitions → generates diversified personas
- Stochastic variation: existing Beta distributions (susceptibility, resilience) preserved; adds `knowledge_depth` as Uniform(0.3, 0.9) controlling how much KG context each persona "knows"
- Scenario YAML gets optional field: `persona_generation: {source: "graph"|"manual", count: 12}`
- When `source: "graph"`: sim_loop calls `generate_personas()` before tick 0, overrides static participant_template prompts
- When `source: "manual"` (default): existing behavior unchanged
- Generated personas cached in DuckDB `generated_personas` table for reuse

**Pros:**
- Natural extension of existing ontology_generator.py
- Persona structure matches existing 6-component template exactly
- Scenario YAML controls whether to use it — zero impact on existing scenarios
- Caching avoids re-generation on subsequent runs with same document

**Cons:**
- Quality depends on KG completeness (Enhancement 1 is a prerequisite)
- Authority slot call at sim startup adds ~5-15s latency
- Prompt engineering for diverse persona generation is non-trivial (RLHF agreeability problem applies)

#### Option B: Template-Based Persona Interpolation (No LLM)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| New dependencies | None |
| LLM cost | Zero |
| Integration points | agent_rolesv3.py only |

**Approach:**
- Define persona templates per archetype (K1, K2, K3) with fill-in slots
- Use KG entities to fill slots deterministically
- No LLM involved in persona generation

**Pros:**
- Fast, deterministic, no LLM cost
- Reproducible across runs

**Cons:**
- Limited diversity — personas are combinations of templates, not genuinely diverse
- Can't capture nuanced domain-specific personality traits from the KG
- Templates need manual authoring per scenario domain
- Misses the key value proposition: heterogeneous, document-grounded personas

**Recommendation: Option A (Extend ontology_generator.py)**

The LLM-based approach is the entire point — template interpolation produces shallow diversity that doesn't justify the infrastructure. The authority slot is available pre-simulation.

---

### Enhancement 4: Post-Sim ReACT Analysis

Two options for extending the ReACT observer to work after simulation completion.

#### Option A: Dual-Mode ReactObserver + New Gateway Endpoints

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| New dependencies | None |
| LLM cost per analysis | ~5-10 authority calls (ReACT steps) |
| Integration points | react_observer.py, gateway.py |

**Approach:**
- `ReactObserver` gets a `mode` parameter: `"live"` (existing) or `"post_sim"`
- In `post_sim` mode, tool executors load data from `data/{run_id}/` JSON files instead of live WorldState
- Extended tool palette for post-sim:
  - `query_trajectories(agent_id?, metric?)` — reads `trajectories.json`
  - `query_dynamics(module?)` — reads `dynamics_analysis.json` (Lyapunov, bifurcation, transfer entropy, etc.)
  - `compare_agents(agent_a, agent_b, metric)` — pair comparison from trajectories
  - `query_flame(tick?)` — reads `flame_population.json`
  - `interview_memory(agent_id, question)` — reconstructs agent context from persisted memories (Enhancement 2), generates response via swarm slot
  - `statistical_test(hypothesis, data)` — runs scipy tests (Mann-Whitney, KS) directly
- Gateway endpoints:
  - `POST /simulation/analyse` — body: `{run_id, questions: [str]}` → starts async ReACT loop
  - `GET /simulation/analyse/status` — progress tracking
  - `GET /simulation/analyse/report` — final structured report
- Structured output: `{executive_summary, methodology_notes, key_findings: [{finding, evidence, confidence}], agent_spotlights, recommendations, limitations}`
- Existing `POST /v1/interview` repurposed for follow-up questions using report context

**Pros:**
- Reuses ReactObserver infrastructure (tool dispatch, safety gates, ReACT prompt construction)
- Authority slot is idle post-simulation — dedicated compute
- Structured report format enables downstream consumption (dashboards, exports)
- interview_memory leverages Enhancement 2's persisted memories
- Statistical tests ground findings in quantitative evidence

**Cons:**
- Dual-mode adds conditional logic to ReactObserver
- Post-sim tool executors are a parallel code path (maintenance surface)
- Report generation quality depends heavily on dynamics_analysis.json completeness

#### Option B: Separate PostSimAnalyzer Class

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium-High |
| New dependencies | None |
| LLM cost per analysis | Same as A |
| Integration points | New file, gateway.py |

**Approach:**
- New module `simulation/post_sim_analyzer.py` — completely independent from ReactObserver
- Own tool definitions, own prompt templates, own LLM interaction loop
- Shares no code with react_observer.py

**Pros:**
- Clean separation — no risk of breaking live observer
- Can evolve independently

**Cons:**
- Duplicates tool dispatch infrastructure, safety gate logic, ReACT prompt construction
- Two systems to maintain for essentially the same pattern
- Misses reuse of tested tool executors (query_scores, query_memory already exist)

**Recommendation: Option A (Dual-Mode ReactObserver)**

The ReACT pattern is identical — reason, act, observe, repeat. Duplicating the infrastructure is unnecessary overhead. A `mode` parameter with conditional tool loading is cleaner than a parallel implementation.

---

## Trade-off Analysis

### Single Storage Engine (DuckDB) vs. Multiple

Using DuckDB for all four enhancements (graph storage, memory persistence, persona caching, analysis report storage) means a single dependency, single connection pool, and consistent query patterns. The alternative — mixing JSON files, SQLite, and possibly KuzuDB — creates operational complexity without meaningful benefit at current scale.

**Trade-off:** DuckDB is new to the stack, but it replaces ad-hoc JSON file management with structured, queryable, persistent storage. The learning curve is minimal (SQL-based, Python API similar to sqlite3).

### In-Memory Hot Path + Write-Behind vs. Direct DB Operations

All four enhancements keep the simulation's hot path in-memory (dict, list, numpy arrays). DuckDB writes happen at tick boundaries or post-simulation. This means:
- Zero latency impact during simulation ticks
- Risk: crash before write-behind = lost data (acceptable — simulation data is also exported as JSON backup)

### Authority Slot Utilization

Enhancements 1, 3, and 4 add work to the authority slot at different times:
- Enhancement 1 (GraphRAG): during document upload (pre-simulation)
- Enhancement 3 (Persona): during simulation startup (pre-tick 0)
- Enhancement 4 (Post-Sim): after simulation completion
- None compete with the authority slot's in-simulation duties (environment + observers)

This is a deliberate design: the authority GPU has idle windows that these enhancements exploit.

### Backward Compatibility

Every enhancement is opt-in:
- GraphRAG: `?extract_graph=true` on document upload
- Memory persistence: `--continue-from` on simulation start
- Persona pipeline: `persona_generation.source: "graph"` in scenario YAML
- Post-Sim ReACT: explicit `POST /simulation/analyse` call

Existing simulations run identically without any configuration change.

## Consequences

### What becomes easier
- Running follow-up simulations grounded in prior run knowledge (memory persistence)
- Generating domain-specific, heterogeneous agent populations from research documents (persona pipeline)
- Producing structured post-simulation reports without manual log analysis (post-sim ReACT)
- Querying document knowledge at the entity/relation level, not just chunk level (GraphRAG)
- Cross-run analysis: "how did participant_0's behavior evolve across 5 runs?"

### What becomes harder
- The system has a new dependency (DuckDB) that needs Docker layer management
- Memory distillation and persona generation add authority slot calls — must monitor GPU utilization during those windows
- Post-sim analysis reports need prompt quality tuning for each scenario domain
- DuckDB schema changes require migration planning

### What we'll need to revisit
- DuckDB performance under concurrent access if we ever run parallel simulations
- Memory decay rate tuning (currently hardcoded default 0.05) — needs empirical calibration
- Entity extraction prompt quality per domain — the prompt in graph_rag.py will need iteration
- Report structure may need domain-specific sections beyond the generic template
- KuzuDB may become worthwhile if graph queries become complex (Cypher vs. SQL for path queries)

## Implementation Order

```
Phase 1: DuckDB Foundation
  ├── pip install duckdb, add to requirements.txt
  ├── simulation/storage.py — DuckDB connection pool + schema creation
  └── Tests: storage init, CRUD, vector queries

Phase 2: GraphRAG (Enhancement 1)
  ├── simulation/graph_rag.py — entity/relation extraction via authority slot
  ├── gateway.py — ?extract_graph=true parameter
  ├── graph_memory.py — seed_from_graphrag() method
  ├── sim_loop.py — load graph context at startup
  └── Tests: extraction parsing, graph seeding, context injection

Phase 3: Memory Persistence (Enhancement 2)
  ├── agent_memory.py — DuckDBMemoryBackend class
  ├── sim_loop.py — --continue-from parameter, write-behind on tick boundary
  ├── Memory distillation (authority slot summarization every N ticks)
  └── Tests: persistence round-trip, cross-run retrieval, decay scoring

Phase 4: Persona Pipeline (Enhancement 3)
  ├── ontology_generator.py — generate_personas() function
  ├── scenario.py — persona_generation config section
  ├── sim_loop.py — conditional persona generation pre-tick-0
  └── Tests: persona generation, scenario integration, caching

Phase 5: Post-Sim ReACT (Enhancement 4)
  ├── react_observer.py — post_sim mode + extended tools
  ├── gateway.py — /simulation/analyse endpoints
  ├── New tools: query_trajectories, query_dynamics, compare_agents, interview_memory, statistical_test
  └── Tests: post-sim tool execution, report generation, structured output
```

## Action Items

1. [ ] Add `duckdb` to `requirements.txt` and Docker layer
2. [ ] Create `simulation/storage.py` with DuckDB connection management and schema DDL
3. [ ] Implement `simulation/graph_rag.py` — entity/relation extraction pipeline
4. [ ] Extend `gateway.py` `POST /v1/documents` with `extract_graph` parameter
5. [ ] Add `seed_from_graphrag()` to `graph_memory.py`
6. [ ] Extend `sim_loop.py` to load graph context at startup
7. [ ] Implement `DuckDBMemoryBackend` in `agent_memory.py`
8. [ ] Add `--continue-from` parameter to `run_simulation()`
9. [ ] Implement memory distillation (episodic → semantic compression)
10. [ ] Add `generate_personas()` to `ontology_generator.py`
11. [ ] Add `persona_generation` config to `scenario.py`
12. [ ] Extend `ReactObserver` with `post_sim` mode
13. [ ] Add `/simulation/analyse` endpoints to `gateway.py`
14. [ ] Implement post-sim tools (query_trajectories, query_dynamics, etc.)
15. [ ] Add scenario YAML fields for new features
16. [ ] Update CLAUDE.md with new module documentation
17. [ ] Write tests for all new modules (target: maintain 500+ test count)
18. [ ] Update Docker image and push to both main and runpod branches
