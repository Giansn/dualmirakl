#!/bin/bash
# GPU 0 — Command-R 7B AWQ — port 8000

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/per.volume/huggingface

MODEL=/per.volume/huggingface/hub/command-r7b-awq

echo "[GPU0] Starting Command-R 7B AWQ on port 8000..."

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --quantization awq \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --served-model-name command-r-7b \
  --device cuda \
  --trust-remote-code
