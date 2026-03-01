# dualmirakl Project Memory

## Project
- Path: `/per.volume/dualmirakl`
- GitHub: `https://github.com/Giansn/dualmirakl.git` branch `runpod`
- Docker Hub: `giansn/dualmirakl:runpod-cu128` (auto-built by GitHub Actions on push to `Dockerfile`, `docker-post-start.sh`, `requirements.txt`, `.dockerignore`)
- Git identity: `dualmirakl@runpod` / `dualmirakl`

## Hardware
- 2x RTX PRO 4500 Blackwell, 32GB VRAM each, compute capability 12.0
- vLLM 0.16.0 installed globally
- Persistent volume: `/per.volume` (MooseFS network storage, 195TB available)
- Local disk: overlay 20GB (only 4.5GB free) — not suitable for large models
- `/dev/shm`: 58GB RAM-disk

## Architecture — Function Slots
- **authority** → GPU 0, port 8000 — analyst agents (observer_a, observer_b)
- **swarm** → GPU 1, port 8001 — persona agents (environment, participant)
- **embedding** → CPU, port 9000 — e5-small-v2 via gateway.py

## Model Configs
- `models/authority.env` — MODEL blank (sage-reasoning-32b placeholder, not downloaded)
- `models/swarm.env` — nemotron-nano-30b NVFP4, fully downloaded (19GB), ready
- Swapping models: edit MODEL + QUANT_FLAGS in `models/<slot>.env` only — nothing else changes
- Served names: always `authority` and `swarm` regardless of underlying model

## Downloaded Models
- `/per.volume/huggingface/hub/nemotron-nano-30b` — 19GB, complete ✅
- `/per.volume/huggingface/hub/e5-small-v2` — 775MB, complete ✅
- authority slot: empty (sage-reasoning-32b is gated, needs HF token + access request)

## Key Model Facts
- **nemotron-nano-30b**: NemotronH hybrid (Mamba+MoE), 30B/3B-active, NVFP4+FP8-KV
  - Requires: `VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_FLASHINFER_MOE_BACKEND=throughput`
  - Custom parser: `${MODEL}/nano_v3_reasoning_parser.py` (ships with weights)
  - No `--quantization` flag needed (auto-detected from hf_quant_config.json)
  - vLLM 0.16.0 supports NemotronHForCausalLM + NVFP4 natively
  - No external mamba_ssm needed (vLLM has built-in Triton Mamba ops)
- **sage-reasoning-32b**: Qwen2.5-32B finetune, BF16 (64GB unquantized — does NOT fit in 32GB)
  - Gated repo, requires HF token with approved access
  - For 32GB fit: need AWQ version or runtime bitsandbytes INT4

## HuggingFace
- Token saved: `~/.cache/huggingface/token` via `HfFolder.save_token()`
- Token also in `/per.volume/dualmirakl/.env` as `HF_TOKEN=<redacted>`

## Key Scripts
- `start_authority.sh` — sources `models/authority.env`, serves as "authority"
- `start_swarm.sh` — sources `models/swarm.env`, serves as "swarm"
- `start_gateway.sh` — uvicorn gateway.py port 9000
- `start_all.sh` — launches all three with health polling
- `stop_all.sh` — kills vLLM + uvicorn, removes pids.txt
- `audit_env.sh` — disk, GPU, model files, port status

## Environment Variables (.env)
- `HF_HOME=/per.volume/huggingface`
- `HF_TOKEN=<redacted>`
- `AUTHORITY_URL=http://localhost:8000/v1`
- `SWARM_URL=http://localhost:8001/v1`
- `GATEWAY_URL=http://localhost:9000/v1`

## Startup Chain
- Pod boot → `/post_start.sh` (baked in Docker image) → clones repo on first boot → runs `autostart.sh`
- `autostart.sh`: sets env, sources .env with `set -a`, sets git identity — does NOT start servers
- Servers started manually: `bash start_all.sh`

## GitHub Token
- Was: `<redacted>` (returned 401 — likely expired)
- gh binary: `/tmp/gh_2.65.0_linux_amd64/bin/gh`
- To trigger Docker rebuild: push any change to a watched file (Dockerfile, docker-post-start.sh, requirements.txt, .dockerignore)

## Simulation Stack
- `simulation/agent_rolesv3.py` — BACKEND_CONFIG maps "analyst"→"authority", "persona"→"swarm"
- `orchestrator.py` — async dual-backend client, reads AUTHORITY_URL/SWARM_URL from env
- `gateway.py` — FastAPI, routes "authority"→GPU0, else→GPU1; /v1/embeddings via e5-small-v2
- No sim_loop.py exists yet (was deleted in v3 restructure)

## Docker Image
- Base: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204`
- anthropic removed from both Dockerfile and requirements.txt
- Image: `giansn/dualmirakl:runpod-cu128`

## Disk Quota Note
- MooseFS quota errors (errno 122) are transient — happen during heavy concurrent downloads
- Do NOT cancel downloads based on quota errors; retry or wait

## Authority Slot — Next Steps
- Need to choose a 32GB-compatible reasoning model with built-in quantization (AWQ)
- Candidates: `Qwen/QwQ-32B-AWQ`, `bartowski/Qwen2.5-32B-Instruct-AWQ`, or Llama-3.3-70B-AWQ
- Once chosen: set MODEL in `models/authority.env`, download, push to trigger rebuild
