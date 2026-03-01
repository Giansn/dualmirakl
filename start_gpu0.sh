#!/bin/bash
# GPU 0 — GLM-5 — port 8000

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/per.volume/huggingface

MODEL=/per.volume/huggingface/hub/glm-5

echo "[GPU0] Starting GLM-5 on port 8000..."

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --quantization awq \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.93 \
  --dtype auto \
  --max-model-len 8192 \
  --served-model-name glm-5 \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-log-requests
