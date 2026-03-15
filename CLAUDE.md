# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
bash start_authority.sh        # GPU 0, port 8000
bash start_swarm.sh            # GPU 1, port 8001
bash start_gateway.sh          # CPU, port 9000 — embedding + proxy + web UI + sim API
bash start_all.sh              # all three with health polling
bash stop_all.sh               # kill all vLLM processes
bash pull_models.sh            # download models to $HF_HOME via huggingface-cli
bash chmod.sh                  # set +x on all scripts
bash audit_env.sh              # check GPU, CUDA, model files, ports
python orchestrator.py         # dual health check + test query
python -m simulation.sim_loop  # run simulation (interactive CLI)
python -m pytest tests/ -v     # run test suite (73 tests, no vLLM needed)
python simulation/dynamics.py  # demo all dynamics analysis (A-H)
```

Logs: `logs/authority.log`, `logs/swarm.log`, `logs/gateway.log`
Sim output: `data/{run_id}/` — config, trajectories, observations, compliance, interventions, dynamics_analysis

## Architecture

```
GPU 0 — authority slot  :8000   (environment + observer_a + observer_b)
GPU 1 — swarm slot      :8001   (participant agents)
CPU   — gateway         :9000   (e5-small-v2 embedding + proxy + web UI + sim API + doc store)
```

### GPU Load Balance

Environment routes to authority (not swarm) to balance GPU utilization.
Per 12-tick run (N=4, K=4): authority ~56 calls, swarm ~48 calls (~1.2:1 ratio).

Three backend keys in `BACKEND_CONFIG` (agent_rolesv3.py):
- `analyst` → authority (observer_a, observer_b)
- `environment` → authority (environment agent — DDA-aware stimulus generation)
- `persona` → swarm (participant agents)

Override at runtime: `BACKEND_ENVIRONMENT=swarm python -m simulation.sim_loop`

### Model Slots

Model configs live in `models/authority.env` and `models/swarm.env`.
To swap a model: edit `MODEL` and `QUANT_FLAGS` in the relevant `.env` file.
The served name is always `authority` or `swarm` — no other code changes needed.

### Three-Layer Cooperation

- **Authority** — reasoning model; environment stimulus generation, observer analysis/intervention.
- **Swarm** — persona model; participant responses, persona summaries, compression (chat UI).
- **Embedding (e5-small-v2)** — 384-dim CPU embedding; behavioral scoring, RAG retrieval, document search.

## Simulation (sim_loop.py)

### Tick Architecture

Optimized four-phase tick with batch stimulus, C+D overlap, and concurrent persona summaries:

- **Pre-phase**: persona summaries for all participants (concurrent, swarm GPU, every PERSONA_SUMMARY_INTERVAL ticks)
- **Phase A** (batched): environment generates stimuli for all N participants in one call → authority
- **Phase B** (concurrent): participants respond via asyncio.gather → swarm
- **Phase C** (CPU): batch embedding → score update with heterogeneous agent modifiers (e5-small-v2, no GPU)
- **Phase D** (every K ticks): overlapped with Phase C on observer ticks — observer_a analyses (authority GPU) runs concurrently with embedding (CPU), then observer_b intervenes sequentially

### Score Dynamics

Two score update modes (selectable via `score_mode` parameter):
- **EMA** (default): `S(t+1) = S(t) + d·α·(signal - S)`. Linear, analytically tractable for SA.
- **Logistic**: sigmoid-transformed signal, saturates at extremes. `logistic_k` controls steepness.

Agent modifiers applied per-participant:
- `susceptibility` ~ Beta(2,3): scales signal strength (higher = more affected by stimuli)
- `resilience` ~ Beta(2,5): baseline dampening (higher = resistant to score change)

Both injected into participant system prompt as personality traits and exported in results.

### Coupled Dynamics (dynamics.py)

Optional inter-agent coupling for emergent group behavior:
```
S_i(t+1) = S_i(t) + d·α·(signal_i - S_i) + κ·g(S̄ - S_i)
```
- `κ > 0`: conformity (peer pressure), `κ < 0`: polarization
- Three coupling functions: linear, sigmoid, threshold
- `coupled_batch_update()` as drop-in replacement for Phase C

### Analysis Toolkit (dynamics.py — modules A through H)

| Module | Function | What it measures |
|--------|----------|-----------------|
| **(A)** Coupled ODE | `coupled_batch_update()` | Inter-agent influence via mean-field κ |
| **(B)** Bifurcation | `bifurcation_sweep()` | Qualitative transitions when sweeping parameters |
| **(C)** Lyapunov | `estimate_system_lyapunov()` | Chaos detection (λ > 0 = chaotic) |
| **(D)** Sobol S2 | `sobol_second_order()` | Superadditive parameter interactions |
| **(E)** Transfer entropy | `transfer_entropy_matrix()` | Directed information flow between agents |
| **(F)** Emergence | `compute_emergence()` | Population-level structure beyond individuals |
| **(G-pre)** Attractor basins | `map_attractor_basins()` | Stable states and their catchment areas |
| **(G)** Stochastic resonance | `stochastic_resonance_curve()` | Optimal noise for intervention effectiveness |
| **(H)** WorldState bridge | `analyze_simulation()` | Runs full suite on simulation output |

`analyze_from_json()` loads exported trajectories for post-hoc analysis.

### Sensitivity Analysis

Three-stage calibration pipeline (no vLLM needed — uses synthetic objective):
1. **Morris screening** → rank 7 parameters by influence (μ*), drop negligible ones
2. **History Matching** (history_matching.py) → iterative NROY space reduction via POM targets
3. **Sobol first + second order** → quantify variance contribution and interactions on narrowed space

Parameters: alpha, K, threshold, dampening, susceptibility, resilience, logistic_k.

### Neutrality

Engagement anchors and calibration targets are domain-specific placeholders. They must be replaced with empirically grounded values before using results for any claims. The simulation engine is domain-neutral — theory enters via configuration, not hardcoded assumptions.

## Document → Simulation Bridge

Upload PDFs via web interface → gateway chunks + embeds via e5-small-v2 → stored as `context/world_context.json` → sim_loop reads on startup → injected into environment and observer system prompts as `[World Context]`. Participants do NOT see uploaded context (maintains domain blindness).

Gateway document API:
- `POST /v1/documents` — upload text, chunk, embed, store (append mode)
- `GET /v1/documents` — list uploaded documents
- `DELETE /v1/documents` — clear all
- `POST /v1/documents/query` — semantic search over stored chunks

### Preflight & Context Detection

Two checks at different lifecycle points:

**Preflight** (`GET /simulation/preflight`) — on instance/pod arrival. Checks infrastructure: vLLM servers, embedding model, output dir, context file. Blocks if critical.

**Detection** (`GET /simulation/detect`) — when user starts a simulation. Scans world_context.json for 5 categories: scenario_description, population_characteristics, outcome_criteria, intervention_rules, temporal_structure. Reports what's present, what's missing, and WHY each missing piece matters for result validity. Non-blocking — user decides.

`POST /simulation/start` includes detection warnings in response.

## Gateway API Summary

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Web interface (interface.html) |
| `/health` | GET | vLLM + embedding health check |
| `/v1/chat/completions` | POST | Proxy to authority/swarm by model name |
| `/v1/embeddings` | POST | CPU e5-small-v2 embedding |
| `/v1/models` | GET | List available models |
| `/v1/documents` | POST/GET/DELETE | Document context store |
| `/v1/documents/query` | POST | Semantic search over documents |
| `/simulation/preflight` | GET | Infrastructure check |
| `/simulation/detect` | GET | World context detection |
| `/simulation/start` | POST | Start sim run (background thread) |
| `/simulation/status` | GET | Current sim state |
| `/simulation/results` | GET | All output JSON including dynamics analysis |

## Key Files

- `orchestrator.py` (106 lines) — `agent_turn()`: single agent turn via httpx HTTP/2 client.
- `gateway.py` (394 lines) — FastAPI; web UI, chat/embed proxy, document store, simulation API. CORS enabled.
- `interface.html` (966 lines) — single-file chat UI with RAG, compression, streaming, PDF+OCR, simulation panel.
- `simulation/sim_loop.py` (1439 lines) — v3 simulation loop: batched Phase A, C+D overlap, heterogeneous agents, dual score modes, world context injection, preflight/detection, data export, SA framework.
- `simulation/dynamics.py` (1706 lines) — 8-module analysis toolkit: coupled ODE, bifurcation, Lyapunov, Sobol S2, transfer entropy, emergence, attractor basins, stochastic resonance, WorldState bridge.
- `simulation/history_matching.py` (440 lines) — iterative NROY parameter space reduction, POM pattern targets, Latin Hypercube sampling.
- `simulation/agent_rolesv3.py` (753 lines) — agent roles, `BACKEND_CONFIG`, engagement anchors (replaceable), intervention codebook, compliance patterns. English + German.
- `models/authority.env` — authority slot: MAX_MODEL_LEN=8192, MAX_NUM_SEQS=12, prefix caching + chunked prefill.
- `models/swarm.env` — swarm slot: nemotron-nano-30b NVFP4, MAX_MODEL_LEN=8192, MAX_NUM_SEQS=12.
- `tests/` — 73 tests across sim_loop, dynamics, and history matching. No vLLM required.

## Deployment

**RunPod**: Docker image `giansn/dualmirakl:runpod-cu128`. Post-start hook clones repo to `/per.volume/dualmirakl/`, runs `autostart.sh`. Models on persistent volume.

**Docker Compose**: `docker compose up -d` starts authority/swarm/gateway services. `docker compose run --profile simulation sim` runs simulation. Models mounted via `models_cache` volume.

**Entrypoint modes**: Set `ENTRYPOINT_MODE` env var: `all` (default), `authority`, `swarm`, `gateway`, `sim`, `shell`.

## vLLM Notes

- `CUDA_VISIBLE_DEVICES` set per start script — do not override globally.
- Nemotron (swarm): NVFP4 auto-detected; requires `VLLM_USE_FLASHINFER_MOE_FP4=1`.
- Both slots: prefix caching enabled (KV reuse for shared system prompts), chunked prefill enabled, `gpu-memory-utilization=0.90`.
- Ports configurable via `AUTHORITY_PORT`, `SWARM_PORT`, `GATEWAY_PORT` env vars.

## Planned

- **Swarm reranker**: chunk reranking between e5-small-v2 retrieval and authority context injection.
- **FLAME GPU 2**: population-scale simulation runs (N > 1000, LLMs become calibration tool, not runtime engine).
- **NumPyro causal model**: Bayesian DAG for agent parameter posteriors — build from qualitative data when theoretical framework is empirically grounded.
