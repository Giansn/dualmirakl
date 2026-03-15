# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
bash start_all.sh              # start authority(:8000) + swarm(:8001) + gateway(:9000)
bash stop_all.sh               # kill all vLLM processes
bash pull_models.sh             # download models to $HF_HOME
python -m simulation.sim_loop  # run simulation (interactive CLI)
python -m pytest tests/ -v     # 73 tests, no vLLM needed
python simulation/dynamics.py  # dynamics analysis demo (A-H)
```

## Architecture

```
GPU 0 — authority :8000  (environment + observers)     MAX_MODEL_LEN=8192, seqs=12
GPU 1 — swarm     :8001  (participants)                MAX_MODEL_LEN=8192, seqs=12
CPU   — gateway   :9000  (e5-small-v2 + proxy + UI + sim API + doc store)
```

GPU balance ~1.2:1 (authority:swarm). Both have prefix caching + chunked prefill, gpu-mem=0.90.

Backend routing (`agent_rolesv3.py` BACKEND_CONFIG):
- `analyst` → authority (observer_a, observer_b)
- `environment` → authority
- `persona` → swarm (participants)

## Simulation

Optimized tick: Pre(persona) → A(batch stimuli, 1 call) → B(concurrent responses) → C+D(overlap: embed CPU + observer GPU)

Score modes: EMA (default) or logistic (saturation). Heterogeneous agents: susceptibility~Beta(2,3), resilience~Beta(2,5). Optional coupling κ for peer influence.

Progress: compact CLI output + visual progress bar in web UI with milestone events.

Data export: `data/{run_id}/` — config, trajectories, observations, compliance, interventions, dynamics_analysis.

## Analysis Toolkit (dynamics.py)

A: Coupled ODE (κ coupling) | B: Bifurcation sweep | C: Lyapunov λ | D: Sobol S2 interactions | E: Transfer entropy | F: Emergence index | G-pre: Attractor basins | G: Stochastic resonance | H: WorldState bridge (`analyze_simulation()`, `analyze_from_json()`)

SA pipeline: Morris → History Matching (NROY) → Sobol S1+S2. 7 params: alpha, K, threshold, dampening, susceptibility, resilience, logistic_k.

## Document → Simulation Bridge

PDF upload → e5 embed → `context/world_context.json` → injected into environment + observer prompts. Participants stay domain-blind.

Detection (`detect_missing_context()`): scans for 5 categories — scenario_description, population_characteristics, outcome_criteria, intervention_rules, temporal_structure. Reports why each matters. Non-blocking.

Preflight (`preflight_check()`): infrastructure check on instance arrival.

## Gateway API

`/` UI | `/health` | `/v1/chat/completions` proxy | `/v1/embeddings` e5 | `/v1/models` | `/v1/documents` CRUD+query | `/simulation/preflight` | `/simulation/detect` | `/simulation/start` POST | `/simulation/status` GET (pct/scores/events) | `/simulation/results` GET

## Key Files

- `simulation/sim_loop.py` (1450 lines) — tick loop, agents, score dynamics, SA, data export, context detection
- `simulation/dynamics.py` (1700 lines) — 8-module analysis toolkit
- `simulation/history_matching.py` (440 lines) — NROY parameter reduction
- `simulation/agent_rolesv3.py` (750 lines) — roles, anchors (replaceable), codebook, compliance. EN+DE
- `gateway.py` (400 lines) — FastAPI proxy, doc store, sim API, UI serving
- `interface.html` (970 lines) — chat UI + RAG + PDF/OCR + sim progress panel
- `orchestrator.py` (105 lines) — httpx HTTP/2 client for vLLM

## Deployment

RunPod: `giansn/dualmirakl:runpod-cu128`. Docker Compose: `docker compose up -d`. Entrypoint modes: all|authority|swarm|gateway|sim|shell.

## Rules

- Engagement anchors and calibration targets are domain-specific placeholders — replace with empirical data before claims
- Never hardcode theoretical frameworks into simulation code
- Push to BOTH main and runpod branches
- Run tests after code changes
