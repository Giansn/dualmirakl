# dualmirakl

Dual-GPU vLLM orchestration stack for MA thesis agent-based simulation research.

## Architecture

```
RunPod Pod (2x RTX PRO 4500 Blackwell, 32GB VRAM each)
│
├── GPU 0 : vLLM → authority slot      (port 8000)  — reasoning / analyst agents
├── GPU 1 : vLLM → swarm slot          (port 8001)  — generative / persona agents
│
├── Gateway (gateway.py)               (port 9000)  — unified /v1 + local embeddings
│
└── Orchestrator (orchestrator.py)
    └── Simulation stack
        ├── SimPy          — discrete-event time engine
        ├── JAX / NumPyro  — probabilistic state models
        └── FLAME GPU 2    — large-scale population runs
```

## Quick Start

```bash
# 1. Start servers manually
bash start_authority.sh   # GPU 0 — set MODEL in models/authority.env first
bash start_swarm.sh       # GPU 1 — ready (nemotron-nano-30b)
bash start_gateway.sh     # embedding + unified proxy

# or all at once:
bash start_all.sh

# 2. Check health
python orchestrator.py

# 3. Run simulation
python simulation/sim_loop.py
```

## Models

| Slot | Model | Path | GPU | Port |
|------|-------|------|-----|------|
| authority | *(not set — edit models/authority.env)* | — | 0 | 8000 |
| swarm | nemotron-nano-30b (NVFP4) | `hub/nemotron-nano-30b` | 1 | 8001 |
| embedding | e5-small-v2 | `hub/e5-small-v2` | CPU | — |

To swap a model: edit the 2-3 variables at the top of `models/<slot>.env`. Nothing else changes.

## Project Structure

```
dualmirakl/
├── models/
│   ├── authority.env       # Authority slot: model path + vLLM flags
│   └── swarm.env           # Swarm slot: model path + vLLM flags
├── start_authority.sh      # Launch authority slot on GPU 0
├── start_swarm.sh          # Launch swarm slot on GPU 1
├── start_gateway.sh        # Launch embedding gateway
├── start_all.sh            # Launch all three
├── stop_all.sh             # Stop all vLLM processes
├── audit_env.sh            # Environment / model health check
├── orchestrator.py         # Async dual-backend client
├── gateway.py              # Unified /v1 endpoint + embeddings
├── requirements.txt
├── .env.example
└── simulation/
    ├── agent_rolesv3.py        # Agent persona definitions (v3)
    ├── sim_loop_v3_patch.py    # Simulation loop patches (v3)
    └── sim_loop.py             # SimPy + LLM simulation loop
```
