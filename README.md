# dualmirakl

[![CodeQL](https://github.com/Giansn/dualmirakl/actions/workflows/codeql.yml/badge.svg)](https://github.com/Giansn/dualmirakl/actions/workflows/codeql.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-508-green)

Multi-GPU vLLM orchestration framework for large-scale agent-based simulation research. Domain-agnostic YAML scenarios, dual vLLM backends, optional **FLAME GPU 2** population amplifier.

## Architecture

```
GPU 0 — vLLM authority :8000    reasoning / observer agents
GPU 1 — vLLM swarm     :8001    generative / persona agents
GPU 2 — FLAME GPU 2             population dynamics (optional)
CPU   — gateway         :9000    e5-small-v2 embeddings + proxy + UI
```

Deploys on RunPod (2x RTX PRO 4500 Blackwell, 32 GB VRAM each). Orchestrator uses httpx HTTP/2 async.

## Quick Start

```bash
bash scripts/setup.sh                                      # first-time setup (deps, models, .env)
bash scripts/start_all.sh                                  # start all services
python -m simulation.sim_loop --scenario scenarios/social_dynamics.yaml  # run
python -m simulation.scenario validate scenarios/foo.yaml  # validate (no GPU)
python -m pytest tests/ -v                                 # 696 tests
```

## Scenario System

All domain logic lives in `scenarios/*.yaml` — swap one file, run a different domain:

| Scenario | Domain |
|----------|--------|
| `social_dynamics.yaml` | Behavioral dynamics + observer intervention (default) |
| `network_resilience.yaml` | Infrastructure failure cascading |
| `market_ecosystem.yaml` | Trader agent herd dynamics |
| `_template.yaml` | Annotated reference for new scenarios |

Each YAML configures: agents, archetypes, actions, scoring (EMA/logistic), transitions, memory, safety, topologies, FLAME, and environment.

## Simulation Engine

**Tick loop** (6 phases with GPU/CPU overlap):

```
Pre     persona injection (memory + archetype)
A       batch stimuli (environment → participants)
B       concurrent participant responses (swarm)
C+D     overlap: CPU scoring + GPU observer analysis/intervention
E       event stream → graph memory update
F       FLAME population dynamics (optional)
```

**Scoring** — EMA (`score += d*α*(signal-score)`) or logistic mode. Per-agent heterogeneity via Beta-distributed susceptibility/resilience. Optional peer coupling (κ).

**Subsystems** — event stream, typed action schemas, per-agent semantic memory (DuckDB), safety gates (ANALYSE/INTERVENE + AUTO/REVIEW/APPROVE tiers), ReACT multi-step observer, real-time graph memory, dual-environment topologies, LLM ontology generator, transition function registry.

**Export** — `data/{run_id}/`: config, trajectories, observations, interventions, event stream, memories, FLAME snapshots.

## FLAME GPU 2 Integration

Optional population amplifier on GPU 2. Scales N LLM participants to thousands of reactive agents without proportional LLM cost.

```
dualmirakl (Phases A–E)              FLAME GPU 2 (Phase F)
┌─────────────────────┐              ┌────────────────────────────────┐
│ N LLM participants  │── bridge ──> │ N influencer agents (1:1 map)  │
│ produce scores via  │              │ seed N_pop population agents   │
│ vLLM backends       │<─ bridge ──  │ (10K default, Beta traits)     │
│                     │              │ spatial messaging + peer κ     │
│ WorldState updated  │              │ EMA/logistic + RTC CUDA C++   │
│ with pop. stats     │              │ sub-stepped (default 10/tick)  │
└─────────────────────┘              └────────────────────────────────┘
```

**Enable** via env var or scenario YAML:

```bash
export FLAME_ENABLED=1
```

```yaml
flame:
  enabled: true
  population_size: 10000    # reactive agents
  kappa: 0.05               # coupling strength
  influencer_weight: 0.8    # signal amplification
  sub_steps: 4              # FLAME steps per tick
```

**Components** (`simulation/flame/`):

| File | Purpose |
|------|---------|
| `engine.py` | FlameEngine — pyflamegpu wrapper (init, step, get/set state) |
| `bridge.py` | FlameBridge — bidirectional data shuttle (scores ↔ population stats) |
| `models.py` | FLAME model description (RTC agent functions, spatial messaging) |
| `flame_setup.py` | Boot sequence — auto-configures engine + W&B + Optuna |

**Per-tick output** — `PopulationSnapshot`: mean/std/min/max scores, 10-bin histogram, influencer scores, spatial clustering metric. Exported to `flame_population.json`.

All FLAME dependencies are optional — graceful fallback when `pyflamegpu` is missing.

## Analysis Toolkit

`simulation/dynamics.py` — 8 modules on score trajectories (no GPU needed):

| Module | Analysis |
|--------|----------|
| A | Coupled ODE (emergent synchronization) |
| B | Bifurcation sweep (parameter transitions) |
| C | Lyapunov exponents (chaos detection) |
| D | Sobol sensitivity S1+S2 (parameter importance) |
| E | Transfer entropy (directed information flow) |
| F | Emergence index (system-level novelty) |
| G | Attractor basins (state-space geometry) |
| H | Stochastic resonance (noise-induced order) |

SA pipeline: Morris → History Matching (NROY) → Sobol. Post-sim ReACT analysis via `POST /simulation/analyse`.

## Knowledge Pipeline

Documents → chunking + e5-small-v2 embedding → GraphRAG extraction (authority) → DuckDB storage (FLOAT[384] vectors) → context injection at tick 0 → graph memory seeding → ReACT observer queries.

Persistent DuckDB tables: `entities`, `relations`, `agent_memories`, `generated_personas`, `analysis_reports`.

## Gateway API

FastAPI on port 9000. Key endpoints:

`/health` | `/v1/chat/completions` (proxy) | `/v1/embeddings` | `/v1/documents` (CRUD + GraphRAG) | `/v1/graph` | `/v1/memories` | `/simulation/start` | `/simulation/status` | `/simulation/results` | `/simulation/analyse` | `/simulation/flame`

## Deployment

```bash
docker compose up -d                         # Docker Compose
docker pull giansn/dualmirakl:runpod-cu128   # or pull directly
```

Entrypoint modes: `all|authority|swarm|gateway|sim|shell`. Requires CUDA 12.9 for Blackwell (sm_120). See [DOCKER.md](DOCKER.md).

## Models

| Slot | Model | GPU | Port |
|------|-------|-----|------|
| authority | *(set in `config/authority.env`)* | 0 | 8000 |
| swarm | nemotron-nano-30b (NVFP4) | 1 | 8001 |
| embedding | e5-small-v2 | CPU | — |

Swap a model: edit `config/<slot>.env`. Nothing else changes.

## Dependencies

**Core**: vLLM, FastAPI, httpx, sentence-transformers, SimPy, JAX, NumPyro, DuckDB, Pydantic

**Optional**: pyflamegpu (FLAME GPU 2), wandb (tracking), optuna (optimization)
