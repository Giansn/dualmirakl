#!/bin/bash
# =============================================================================
# Gateway — unified /v1 proxy + local embedding (e5-small-v2, CPU)
# =============================================================================
PROJ="${DUALMIRAKL_ROOT:-/workspace/dualmirakl}"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
GATEWAY_PORT="${GATEWAY_PORT:-9000}"

echo "[gateway] Starting gateway + embedding on port $GATEWAY_PORT..."

exec uvicorn gateway:app \
  --host 0.0.0.0 \
  --port "$GATEWAY_PORT" \
  --workers 1 \
  --loop asyncio \
  --app-dir "$PROJ"
