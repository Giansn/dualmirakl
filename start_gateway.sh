#!/bin/bash
# =============================================================================
# Gateway — unified /v1 proxy + local embedding (e5-small-v2, CPU) — port 9000
# =============================================================================
PROJ=/per.volume/dualmirakl
export HF_HOME=/per.volume/huggingface

echo "[gateway] Starting gateway + embedding on port 9000..."

exec uvicorn gateway:app \
  --host 0.0.0.0 \
  --port 9000 \
  --workers 1 \
  --loop asyncio \
  --app-dir "$PROJ"
