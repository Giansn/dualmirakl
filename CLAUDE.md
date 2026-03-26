# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
bash start_all.sh              # start authority(:8000) + swarm(:8001) + gateway(:9000)
bash stop_all.sh               # kill all vLLM processes
bash pull_models.sh             # download models to $HF_HOME
python -m simulation.sim_loop  # run simulation (interactive CLI, legacy mode)
python -m simulation.sim_loop --scenario scenarios/social_dynamics.yaml  # scenario-driven
python -m simulation.scenario validate scenarios/foo.yaml  # dry-run validation (zero GPU)
python -m pytest tests/ -v     # 442 tests, no vLLM needed
python simulation/dynamics.py  # dynamics analysis demo (A-H)
```

## Architecture

```
GPU 0 -- authority :8000  (environment + observers)     MAX_MODEL_LEN=8192, seqs=12
GPU 1 -- swarm     :8001  (participants)                MAX_MODEL_LEN=8192, seqs=12
GPU 2 -- flame            (FLAME GPU 2 population dynamics, optional)
CPU   -- gateway   :9000  (e5-small-v2 + proxy + UI + sim API + doc store)
```

GPU balance ~1.2:1 (authority:swarm). Both have prefix caching + chunked prefill, gpu-mem=0.90.

## Scenario System (domain-agnostic framework)

dualmirakl is a general-purpose multi-agent simulation framework. All domain-specific configuration lives in `scenarios/*.yaml` files. Swap one file to run a completely different simulation domain.

```bash
# Available scenarios
scenarios/social_dynamics.yaml        # behavioral dynamics (default)
scenarios/network_resilience.yaml     # infrastructure failure cascading
scenarios/market_ecosystem.yaml       # trader agent herd dynamics
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

Typed JSON schemas for agent outputs. Participant: respond, disengage, escalate. Observer A: analyse (with clustering/concern enums). Observer B: intervene (with codebook enum) or no_intervention. Schema injected into prompts, parsed from output, falls back to free-text.

### Agent Memory (simulation/agent_memory.py)

Per-agent persistent memory store across ticks. Semantic retrieval via e5-small-v2 + tag-based lookup. Deduplication by cosine similarity. LRU eviction. Agents can store memories via structured output.

### Safety Gates (simulation/safety.py)

Observer mode enforcement (ANALYSE vs INTERVENE) + action safety classification (AUTO/REVIEW/APPROVE). dynamics_dampening requires APPROVE tier. Allowlist is the only override mechanism. Tool safety tiers: tool.query_* → AUTO, tool.interview_agent → REVIEW.

### ReACT Observer (simulation/react_observer.py)

MiroFish-inspired multi-step observer. Instead of single-pass analysis, iteratively reasons → calls tools → observes results → final answer. Enabled via `react.enabled: true` in scenario YAML. Tools: query_scores, query_events, check_interventions, query_memory, interview_agent, query_graph. Bounded by max_steps (default 5). Drop-in replacement for ObserverAgent.analyse() in Phase D1.

### Graph Memory (simulation/graph_memory.py)

Real-time knowledge graph feedback loop. EventStream events distilled into a NetworkX-style shared graph after each tick. Entities: agents, score regions, intervention nodes, behavioral patterns. Temporal edges with valid_at/invalid_at. Queryable by the ReACT observer via query_graph tool. Exported as part of run data.

## Analysis Toolkit (dynamics.py)

A: Coupled ODE | B: Bifurcation sweep | C: Lyapunov | D: Sobol S2 | E: Transfer entropy | F: Emergence index | G-pre: Attractor basins | G: Stochastic resonance | H: WorldState bridge

SA pipeline: Morris -> History Matching (NROY) -> Sobol S1+S2.

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
| `simulation/scenario.py` | ScenarioConfig loader + validator + dry-run CLI |
| `simulation/agents.py` | AgentFactory, AgentSpec, prompt rendering |
| `simulation/scoring.py` | ScoreEngine base + EMA/Logistic implementations |
| `simulation/transitions.py` | Transition function registry + built-ins |
| `simulation/event_stream.py` | Unified event stream |
| `simulation/action_schema.py` | Structured action schemas + parsing |
| `simulation/agent_memory.py` | Per-agent memory store |
| `simulation/safety.py` | Observer mode + safety classification |
| `simulation/react_observer.py` | ReACT observer with tool use (MiroFish-inspired) |
| `simulation/graph_memory.py` | Real-time graph memory feedback loop |
| `simulation/dynamics.py` | 8-module analysis toolkit |
| `simulation/agent_rolesv3.py` | Legacy roles, anchors, codebook, compliance |
| `simulation/flame/` | FLAME GPU 2 engine, bridge, models |
| `simulation/flame_setup.py` | FLAME boot sequence |
| `simulation/tracking.py` | W&B tracking (optional) |
| `simulation/optimize.py` | Optuna optimization (optional) |
| `orchestrator.py` | httpx HTTP/2 client for vLLM |
| `gateway.py` | FastAPI proxy, doc store, sim API |

## Gateway API

`/` UI | `/health` | `/v1/chat/completions` proxy | `/v1/embeddings` e5 | `/v1/models` | `/v1/documents` CRUD+query | `/simulation/preflight` | `/simulation/detect` | `/simulation/start` POST | `/simulation/status` GET | `/simulation/results` GET

## Deployment

RunPod: `giansn/dualmirakl:runpod-cu128`. Docker Compose: `docker compose up -d`. Entrypoint modes: all|authority|swarm|gateway|sim|shell.

## Rules

- Engagement anchors and calibration targets are domain-specific placeholders -- replace with empirical data before claims
- Never hardcode domain knowledge into engine code -- use scenario.yaml
- Push to BOTH main and runpod branches
- Run tests after code changes
- Validate scenarios before committing: `python -m simulation.scenario validate`
