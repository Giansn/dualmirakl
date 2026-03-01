#!/bin/bash
# =============================================================================
# Authority — reasoning / synthesis / analyst agents — GPU 0, port 8000
# Model config: models/authority.env  (edit that file to swap models)
# =============================================================================
set -e
PROJ=/per.volume/dualmirakl
source "$PROJ/models/authority.env"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/per.volume/huggingface
eval "export $EXTRA_ENV" 2>/dev/null || true

if [ -z "$MODEL" ]; then
  echo "[authority] ERROR: MODEL is not set in models/authority.env"
  exit 1
fi

echo "[authority] Model:   $(basename $MODEL)"
echo "[authority] Context: $MAX_MODEL_LEN tokens | Seqs: $MAX_NUM_SEQS"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name authority \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.93 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --trust-remote-code \
  --disable-log-requests \
  $QUANT_FLAGS \
  $EXTRA_FLAGS
