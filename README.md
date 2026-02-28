# dualmirakl

Dual-GPU vLLM orchestration stack for MA thesis research on media addiction dynamics.

## Architecture

```
RunPod Pod (2x RTX PRO 4500 Blackwell, 32GB VRAM each)
│
├── GPU 0 : vLLM → Command-R 7B AWQ  (port 8000)  — reasoning / synthesis agents
├── GPU 1 : vLLM → Qwen 2.5 7B AWQ  (port 8001)  — generative / persona agents
│
└── Orchestrator (orchestrator.py)
    └── Simulation stack
        ├── SimPy          — discrete-event time engine
        ├── JAX / NumPyro  — probabilistic state models
        └── FLAME GPU 2    — large-scale population runs
```

## Quick Start

```bash
# 1. Start both vLLM servers
bash start_all.sh

# 2. Check health
python orchestrator.py

# 3. Run simulation
python simulation/sim_loop.py
```

## Models (pre-downloaded to /per.volume/huggingface/hub/)

| Model | Path | GPU | Port |
|-------|------|-----|------|
| Command-R 7B AWQ | `hub/command-r7b-awq` | 0 | 8000 |
| Qwen 2.5 7B AWQ  | `hub/qwen2.5-7b-awq`  | 1 | 8001 |

## Project Structure

```
dualmirakl/
├── start_gpu0.sh        # Launch Command-R on GPU0
├── start_gpu1.sh        # Launch Qwen on GPU1
├── start_all.sh         # Launch both + health poll
├── stop_all.sh          # Stop all vLLM processes
├── orchestrator.py      # Async dual-backend client
├── requirements.txt
├── .env.example
└── simulation/
    ├── agent_roles.py   # Agent persona definitions
    └── sim_loop.py      # SimPy + LLM simulation loop
```

## Research Context

MA thesis: **Computational modelling of media addiction dynamics**
- Agents: media users, platform algorithms, researchers, policy analysts
- State: addiction score evolves via probabilistic transitions (NumPyro)
- Scale: population-level runs on FLAME GPU 2
- Orchestration: Claude Code directs local LLM agents via vLLM OpenAI-compatible API
