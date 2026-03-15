# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
bash start_authority.sh        # GPU 0, port 8000
bash start_swarm.sh            # GPU 1, port 8001
bash start_gateway.sh          # CPU, port 9000 — embedding + proxy + web UI
bash start_all.sh              # all three with health polling
bash stop_all.sh               # kill all vLLM processes
bash pull_models.sh            # download models to $HF_HOME via huggingface-cli
bash chmod.sh                  # set +x on all scripts
bash audit_env.sh              # check GPU, CUDA, model files, ports
python orchestrator.py         # dual health check + test query
python -m simulation.sim_loop  # run simulation (interactive CLI)
python -m pytest tests/ -v     # run test suite (53 tests, no vLLM needed)
python simulation/dynamics.py  # demo coupled dynamics + Lyapunov estimation
```

Logs: `logs/authority.log`, `logs/swarm.log`, `logs/gateway.log`
Sim output: `data/{run_id}/` (config, trajectories, observations, compliance, interventions)

## Architecture

```
GPU 0 — authority slot  :8000   (environment + observer_a + observer_b)
GPU 1 — swarm slot      :8001   (participant agents)
CPU   — gateway         :9000   (e5-small-v2 embedding + proxy + web UI)
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
- **Embedding (e5-small-v2)** — 384-dim CPU embedding; behavioral scoring, RAG retrieval.

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

### Analysis Tools (dynamics.py)

**Phase portrait**: extract (score, velocity, acceleration) from trajectories. `find_fixed_points()` detects attractors.

**Bifurcation analysis**: `bifurcation_sweep(param, values, base_params)` sweeps any parameter with synthetic objective, `detect_bifurcation_points()` flags qualitative transitions.

**Lyapunov exponent λ**: quantifies sensitivity to initial conditions.
- `lyapunov_exponent_twin()`: twin trajectory divergence rate
- `lyapunov_from_timeseries()`: Rosenstein (1993) method, works on existing output
- `estimate_system_lyapunov()`: population-level regime classification (chaotic/stable/marginal)

### Sensitivity Analysis

Three-stage calibration pipeline (no vLLM needed — uses synthetic objective):
1. **Morris screening** → rank 7 parameters by influence (μ*), drop negligible ones
2. **History Matching** (history_matching.py) → iterative NROY space reduction via POM targets
3. **Sobol first-order** → quantify variance contribution on the narrowed space

Parameters: alpha, K, threshold, dampening, susceptibility, resilience, logistic_k.

### Neutrality

Engagement anchors and calibration targets are domain-specific placeholders. They must be replaced with empirically grounded values before using results for any claims. The simulation engine is domain-neutral — theory enters via configuration, not hardcoded assumptions.

## Key Files

- `orchestrator.py` — `agent_turn()`: single agent turn via httpx HTTP/2 client. `chat()`, `dual_query()`, `health_check()`.
- `gateway.py` — FastAPI; serves web UI at `/`; routes `/v1/chat/completions` by model name; `/v1/embeddings` via CPU e5-small-v2. CORS enabled.
- `interface.html` — single-file chat UI with RAG, compression, streaming, PDF+OCR, simulation panel.
- `simulation/sim_loop.py` — v3 simulation loop: batched Phase A, C+D overlap, heterogeneous agents, dual score modes, data export, SA framework.
- `simulation/agent_rolesv3.py` — agent role definitions, `BACKEND_CONFIG`, engagement anchors (domain-specific, replaceable), intervention codebook, compliance patterns. English + German.
- `simulation/dynamics.py` — coupled ODE score dynamics (mean-field coupling), phase portrait extraction, bifurcation sweep/detection, Lyapunov exponent estimation (twin + Rosenstein).
- `simulation/history_matching.py` — iterative NROY parameter space reduction, POM pattern targets, Latin Hypercube sampling, implausibility measure.
- `models/authority.env` — authority slot: MODEL, QUANT_FLAGS, MAX_MODEL_LEN=8192, MAX_NUM_SEQS=12, prefix caching + chunked prefill.
- `models/swarm.env` — swarm slot: nemotron-nano-30b NVFP4, MAX_MODEL_LEN=8192, MAX_NUM_SEQS=12.
- `tests/` — 53 tests across sim_loop, dynamics, and history matching. No vLLM required.

## Deployment

**RunPod**: Docker image `giansn/dualmirakl:runpod-cu128`. Post-start hook clones repo to `/per.volume/dualmirakl/`, runs `autostart.sh`. Models on persistent volume.

**Docker Compose**: `docker compose up -d` starts authority/swarm/gateway services. `docker compose run sim` runs simulation. Models mounted via `models_cache` volume.

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
