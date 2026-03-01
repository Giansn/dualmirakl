#!/bin/bash
# GPU 1 — DeepSeek-V3.2-Special — port 8001

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/per.volume/huggingface

MODEL=/per.volume/huggingface/hub/deepseek-v3.2-special

echo "[GPU1] Starting DeepSeek-V3.2-Special on port 8001..."

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --quantization awq \
  --port 8001 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.93 \
  --dtype auto \
  --max-model-len 8192 \
  --served-model-name deepseek-v3.2-special \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-log-requests
