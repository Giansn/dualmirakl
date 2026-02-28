#!/bin/bash
# GPU 1 — Qwen 2.5 7B AWQ — port 8001

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/per.volume/huggingface

MODEL=/per.volume/huggingface/hub/qwen2.5-7b-awq

echo "[GPU1] Starting Qwen 2.5 7B AWQ on port 8001..."

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --quantization awq \
  --port 8001 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.93 \
  --dtype auto \
  --max-model-len 8192 \
  --served-model-name qwen-7b \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-log-requests
