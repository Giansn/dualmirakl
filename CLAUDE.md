# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## RunPod Quick Start (generic pod, no custom Docker)

```bash
bash scripts/go.sh                # one command: setup (if needed) + start all services
```

Or step by step: `bash scripts/setup.sh` then `bash scripts/start_all.sh`.

## Commands

```bash
bash scripts/go.sh                                               # setup + start (idempotent)
bash scripts/go.sh --restart                                     # stop + start
bash scripts/go.sh --status                                      # check what's running
bash scripts/status.sh                                           # quick health dashboard
bash scripts/start_all.sh                                        # start authority(:8000) + swarm(:8001) + gateway(:9000)
bash scripts/stop_all.sh                                         # kill all vLLM processes
bash scripts/pull_models.sh                                      # download models to $HF_HOME
python -m simulation.sim_loop                                    # run simulation (interactive CLI, legacy mode)
python -m simulation.sim_loop --scenario scenarios/social_dynamics.yaml  # scenario-driven
python -m simulation.scenario validate scenarios/foo.yaml        # dry-run validation (zero GPU)
python -m pytest tests/ -v                                       # all tests, no vLLM needed
python -m pytest tests/test_scenario.py -v                       # single file
python -m pytest tests/test_action_schema.py::TestSchemaStructure -v     # single class
python -m pytest tests/test_safety.py::TestObserverMode::test_analyse_blocks_intervention -v  # single test
python simulation/dynamics.py                                    # dynamics analysis demo (A-H)
python examples/walkthrough.py                                   # 9-step interactive tutorial
bash scripts/audit_env.sh                                        # environment + model health check
```

## Architecture

```
GPU 0 -- authority :8000  (environment + observers)     MAX_MODEL_LEN=16384, seqs=16
GPU 1 -- swarm     :8001  (participants)                MAX_MODEL_LEN=16384, seqs=16
GPU 2 -- flame            (FLAME GPU 2 population dynamics, optional)
CPU   -- gateway   :9000  (e5-small-v2 + proxy + UI + sim API + doc store)
```

GPU balance ~1.2:1 (authority:swarm). Both have prefix caching + chunked prefill, gpu-mem=0.90.

### Agent-to-slot mapping

| Agent | Slot | Role |
|-------|------|------|
| `observer_a` | authority (GPU 0) | Analysis only — no interventions |
| `observer_b` | authority (GPU 0) | Proposes interventions from observer_a output |
| `environment` | swarm (GPU 1) | Generates scenario events per tick |
| `participant` | swarm (GPU 1) | Persona agents responding to events |

### Cooperation loop

```
authority  →  direction          →  swarm
swarm      →  prepared chunks    →  embedding (e5-small-v2, CPU)
embedding  →  top-k results      →  swarm
swarm      →  filtered context   →  authority
authority  →  output + new need  →  swarm  [repeat]
```

### Concurrency model

- `orchestrator.py` uses a singleton `httpx.AsyncClient` with HTTP/2 multiplexing
- Phase B (participant responses) runs concurrently via `asyncio.gather()`
- Phase C+D overlap: embedding on CPU thread pool (`run_in_executor`) while observer runs on GPU
- Observer A → B is sequential (B depends on A's analysis output)
- `MultiRunScheduler` in `parallel/tick_scheduler.py` manages concurrent runs with `asyncio.Semaphore` respecting vLLM max_seqs

### Static import cycles (refactor safety)

GitNexus may report **mutual `IMPORTS` edges** between file pairs, e.g. `optimize.py` ↔ `surrogate.py`, `react_observer.py` ↔ `sim_loop.py`, `optimize.py` ↔ `sim_loop.py`, `gpu_harmony.py` ↔ `sim_loop.py`. At runtime these are usually **broken by lazy imports** (imports inside functions / after tick 0). Avoid adding **top-level** cross-imports between those modules without checking load order.

## Scenario System (domain-agnostic framework)

dualmirakl is a general-purpose multi-agent simulation framework. All domain-specific configuration lives in `scenarios/*.yaml` files. Swap one file to run a completely different simulation domain.

```bash
# Available scenarios
scenarios/social_dynamics.yaml        # behavioral dynamics (default)
scenarios/network_resilience.yaml     # infrastructure failure cascading
scenarios/market_ecosystem.yaml       # trader agent herd dynamics
scenarios/minimal.yaml                # "Hello World" (2 agents, 3 ticks) — setup verification
scenarios/_template.yaml              # annotated reference for new scenarios
```

### scenario.yaml structure

```
meta:          name, version, description
agents:        roles (id, slot, type, system_prompt, max_tokens, count)
archetypes:    profiles (id, label, properties) + distribution
actions:       schemas (from action_schema.py) + instances
scoring:       mode (ema|logistic), distributions, parameters
transitions:   from/to profiles + registered Python function name
memory:        enabled, max_entries_per_agent, dedup_threshold
safety:        enabled, action_allowlist, fallback_action
context_categories:  domain-specific document detection categories
flame:         enabled, population_size, kappa, influencer_weight
react:         enabled, max_steps, tools (ReACT observer config)
topologies:    list of {id, type, weight, cluster_size} (dual-environment)
environment:   tick_count, tick_unit, initial_state
```

Validate before running: `python -m simulation.scenario validate scenarios/foo.yaml`

### Transition functions (simulation/transitions.py)

Registered with `@register_transition("name")` decorator. Built-ins: `escalation_sustained`, `recovery_sustained`, `threshold_cross`, `oscillation_detect`. Custom functions can be added and referenced from YAML.

## Simulation

Optimized tick: Pre(persona) -> A(batch stimuli, 1 call) -> B(concurrent responses) -> C+D(overlap: embed CPU + observer GPU) -> F(FLAME, optional)

Score modes: EMA (default) or logistic (saturation). Heterogeneous agents: susceptibility~Beta(2,3), resilience~Beta(2,5). Optional coupling kappa for peer influence. All params configurable per scenario.

Data export: `data/{run_id}/` -- config, trajectories, observations, compliance, interventions, event_stream, agent_memories.

### Event Stream (simulation/event_stream.py)

Unified chronological event log. All phases emit typed events (stimulus, response, score, observation, intervention, compliance, flame_snapshot). Queryable by agent_id, event_type, tick range. Exported as event_stream.json.

### Action Schemas (simulation/action_schema.py)

Typed JSON schemas for agent outputs. Participant: respond, disengage, escalate. Observer A: analyse (with clustering/concern enums). Observer B: intervene (with codebook enum) or no_intervention. Schema injected into prompts, parsed from output, falls back to free-text via cosine similarity matching.

### Agent Memory (simulation/agent_memory.py)

Per-agent persistent memory store across ticks. Semantic retrieval via e5-small-v2 + tag-based lookup. Deduplication by cosine similarity (>0.9 threshold). LRU eviction. DuckDB persistence backend for cross-run memory.

### Safety Gates (simulation/safety.py)

Observer mode enforcement (ANALYSE vs INTERVENE) + action safety classification (AUTO/REVIEW/APPROVE). dynamics_dampening requires APPROVE tier. Allowlist is the only override mechanism. Tool safety tiers: tool.query_* → AUTO, tool.interview_agent → REVIEW.

### ReACT Observer (simulation/react_observer.py)

Multi-step observer. Iteratively reasons → calls tools → observes results → final answer. Enabled via `react.enabled: true` in scenario YAML. Tools: query_scores, query_events, check_interventions, query_memory, interview_agent, query_graph. Bounded by max_steps (default 5). Drop-in replacement for ObserverAgent.analyse() in Phase D1.

### Graph Memory (simulation/graph_memory.py)

Real-time knowledge graph feedback loop. EventStream events distilled into a NetworkX-style shared graph after each tick. Entities: agents, score regions, intervention nodes, behavioral patterns. Temporal edges with valid_at/invalid_at. Queryable by the ReACT observer via query_graph tool.

### Dual-Environment Topologies (simulation/topology.py)

Agents experience multiple interaction topologies simultaneously (Option A: dual-context, single-response). Types: `independent` (broadcast, default) and `clustered` (agents grouped, see neighbors). Combined stimuli in Phase B, single score signal. Zero overhead when single topology.

### Ontology Generator (simulation/ontology_generator.py)

LLM-generated archetypes and transitions from domain documents. Uses authority backend to auto-generate profiles, distribution fractions, and transition rules. Output mergeable into ScenarioConfig. Also contains `generate_personas()` for automatic persona creation from KG.

### DuckDB Storage (simulation/storage.py)

Embedded DuckDB storage layer for persistent data across runs. Tables: `entities`, `relations` (GraphRAG), `agent_memories` (cross-run persistence), `generated_personas` (cache), `analysis_reports` (post-sim). File: `data/dualmirakl.duckdb`. Supports native FLOAT[384] vector operations for e5-small-v2 embeddings.

### GraphRAG (simulation/graph_rag.py)

Document → Knowledge Graph pipeline. Entity/relation extraction via authority slot (batch, JSON-mode). Persists to DuckDB. Context injection: `query_graph_context()` retrieves relevant triples by cosine similarity. Seeds `graph_memory.py` before tick 0. Triggered by `POST /v1/documents?extract_graph=true` or `POST /v1/documents/extract_graph`.

### Possibility Report (simulation/possibility_report.py)

Structured simulation output: possibility branches sorted by probability. Composes attractor basins, bifurcation analysis, sensitivity indices, and convergence checks from dynamics.py + stats/ into ranked divergent outcome trajectories with narrative and parameter levers.

### Post-Sim ReACT Analysis (simulation/react_observer.py — PostSimAnalyser)

Authority-slot post-simulation analysis using the ReACT pattern. Loads data from `data/{run_id}/`, iteratively queries trajectories, dynamics, events, memories, graph. Produces structured reports. Tools: query_trajectories, query_dynamics, compare_agents, query_events, query_graph, interview_memory, statistical_test. API: `POST /simulation/analyse`.

## Extension Modules

### stats/ — Analysis metrics (no LLM, pure math)

- `core.py` — stance_drift, polarization (Sarle's bimodality coefficient), opinion_clusters (K-Means + silhouette), influence_network (degree centrality)
- `validation.py` — validation helpers

### ml/ — Machine learning extensions

- `evolution.py` — (mu+lambda) evolutionary strategy (AgentGenome with Big Five personality, EvolutionEngine)
- `beliefs.py` — Bayesian belief update model
- `bandit.py` — Multi-armed bandit strategy selection (UCB)
- Install extras: `pip install -r requirements-ml.txt` (nolds, SALib)

### parallel/ — Multi-run orchestration

- `tick_scheduler.py` — MultiRunScheduler (RunConfig → RunResult), capacity-aware with asyncio.Semaphore respecting vllm_max_seqs

## Analysis Toolkit (simulation/dynamics.py)

A: Coupled ODE | B: Bifurcation sweep | C: Lyapunov | D: Sobol S2 | E: Transfer entropy | F: Emergence index | G-pre: Attractor basins | G: Stochastic resonance | H: WorldState bridge

SA pipeline: Morris -> History Matching (NROY, via history_matching.py) -> Sobol S1+S2.

## FLAME GPU 2 (optional 3rd GPU)

Set `FLAME_ENABLED=1` or `flame.enabled: true` in scenario YAML. Population amplifier: N LLM participants become influencer agents seeding dynamics into thousands of simpler agents via spatial messaging.

Boot sequence (`flame_setup.py`): auto-configures engine + W&B + Optuna. Status: `GET /simulation/flame`.

## Optimization & Tracking (optional)

**Optuna**: Bayesian optimization. Fast (surrogate, no GPU) or full (live sim). Param space from scenario config.
**W&B**: Experiment tracking. Auto-logs per-tick scores, FLAME stats, run artifacts. No-op if not installed.

## Key Files

| File | Purpose |
|---|---|
| `simulation/sim_loop.py` | Tick loop, agents, scoring, SA, export, FLAME integration |
| `simulation/scenario.py` | ScenarioConfig loader + Pydantic validator + dry-run CLI |
| `simulation/agents.py` | AgentFactory, AgentSpec, AgentSet, prompt template rendering |
| `simulation/scoring.py` | ScoreEngine base + EMA/Logistic implementations |
| `simulation/transitions.py` | Transition function registry + built-ins |
| `simulation/event_stream.py` | Unified event stream (SimEvent, EventStream) |
| `simulation/action_schema.py` | Structured action schemas + JSON parsing + fallback |
| `simulation/agent_memory.py` | Per-agent memory store + DuckDB persistence backend |
| `simulation/safety.py` | Observer mode (ANALYSE/INTERVENE) + safety classification |
| `simulation/react_observer.py` | ReACT observer (live) + PostSimAnalyser (post-sim) |
| `simulation/graph_memory.py` | Real-time graph memory + GraphRAG seeding |
| `simulation/topology.py` | Dual-environment topology manager |
| `simulation/ontology_generator.py` | LLM-generated ontology + persona generation |
| `simulation/storage.py` | DuckDB storage layer (entities, memories, personas, reports) |
| `simulation/graph_rag.py` | Document → Knowledge Graph extraction pipeline |
| `simulation/possibility_report.py` | Structured possibility branches output |
| `simulation/dynamics.py` | 8-module analysis toolkit (ODE, bifurcation, Lyapunov, etc.) |
| `simulation/history_matching.py` | Morris + NROY history matching for SA |
| `simulation/agent_rolesv3.py` | Legacy roles, anchors, codebook, compliance |
| `simulation/flame/` | FLAME GPU 2 engine, bridge, models |
| `simulation/flame_setup.py` | FLAME boot sequence |
| `simulation/tracking.py` | W&B tracking (optional) |
| `simulation/optimize.py` | Optuna optimization (optional) |
| `stats/core.py` | Stance drift, polarization, opinion clusters, influence network |
| `ml/evolution.py` | Evolutionary strategy (AgentGenome, EvolutionEngine) |
| `ml/beliefs.py` | Bayesian belief updates |
| `ml/bandit.py` | Multi-armed bandit strategy selection |
| `parallel/tick_scheduler.py` | MultiRunScheduler (capacity-aware parallel runs) |
| `orchestrator.py` | httpx HTTP/2 async client for dual vLLM backends |
| `gateway.py` | FastAPI proxy, doc store, embedding, sim API, UI |

## Gateway API

`/` UI | `/health` | `/v1/chat/completions` proxy | `/v1/embeddings` e5 | `/v1/models` | `/v1/documents` CRUD+query+extract_graph | `/v1/graph` GET/DELETE | `/v1/graph/query` POST | `/v1/memories` GET | `/v1/memories/{run_id}` GET | `/v1/interview` POST | `/simulation/preflight` | `/simulation/detect` | `/simulation/start` POST (supports `continue_from`) | `/simulation/status` GET | `/simulation/results` GET | `/simulation/analyse` POST | `/simulation/analyse/status` GET | `/simulation/analyse/report` GET

## Testing

Tests use pytest with manual `sys.path` insertion (no conftest.py). LLM calls are mocked — no vLLM needed. Pydantic model fixtures provide minimal valid configs.

## CI/CD

GitHub Actions (`.github/workflows/docker-build.yml`): builds and pushes Docker image on push to `main` when Dockerfile, entrypoint, requirements, or start scripts change. Image: `giansn/dualmirakl:runpod-cu128` + `:latest`. Cache: GitHub Actions (gha type).

## Deployment

RunPod: `giansn/dualmirakl:runpod-cu128`. Docker Compose: `docker compose -f docker/docker-compose.yml up -d`. Entrypoint modes: all|authority|swarm|gateway|sim|shell. See docs/DOCKER.md for Blackwell GPU (sm_120) CUDA 12.9 requirements.

## Rules

- Engagement anchors and calibration targets are domain-specific placeholders -- replace with empirical data before claims
- Never hardcode domain knowledge into engine code -- use scenario.yaml
- Push to BOTH main and runpod branches
- Run tests after code changes
- Validate scenarios before committing: `python -m simulation.scenario validate`

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **dualmirakl** (2756 symbols, 11813 relationships, 238 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/dualmirakl/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/dualmirakl/context` | Codebase overview, check index freshness |
| `gitnexus://repo/dualmirakl/clusters` | All functional areas |
| `gitnexus://repo/dualmirakl/processes` | All execution flows |
| `gitnexus://repo/dualmirakl/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
