#!/bin/bash
# =============================================================================
# Swarm — generative / persona / organizing agents — GPU 1, port 8001
# Model config: models/swarm.env  (edit that file to swap models)
# =============================================================================
set -e
PROJ=/per.volume/dualmirakl
source "$PROJ/models/swarm.env"

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/per.volume/huggingface
eval "export $EXTRA_ENV" 2>/dev/null || true

if [ -z "$MODEL" ]; then
  echo "[swarm] ERROR: MODEL is not set in models/swarm.env"
  exit 1
fi

echo "[swarm] Model:   $(basename $MODEL)"
echo "[swarm] Context: $MAX_MODEL_LEN tokens | Seqs: $MAX_NUM_SEQS"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name swarm \
  --port 8001 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.93 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --trust-remote-code \
  --disable-log-requests \
  $QUANT_FLAGS \
  $EXTRA_FLAGS
