#!/bin/bash
# =============================================================================
# dualmirakl autostart
# Called by /post_start.sh on every pod boot.
# Persists on /per.volume — survives pod restarts and template resets.
# =============================================================================

PROJ=/per.volume/dualmirakl
mkdir -p "$PROJ/logs"

LOG="$PROJ/logs/autostart.log"
echo "" >> "$LOG"
echo "════════════════════════════════════════" >> "$LOG"
echo "[autostart] $(date)" >> "$LOG"
echo "════════════════════════════════════════" >> "$LOG"
exec >> "$LOG" 2>&1

# ── Environment ──────────────────────────────────────────────────────────────
export HF_HOME=/per.volume/huggingface
export PATH="/usr/local/bin:$PATH"

[ -f "$PROJ/.env" ] && source "$PROJ/.env" && echo "[autostart] .env loaded"

# ── Git identity (for commits from this pod) ──────────────────────────────────
git -C "$PROJ" config user.email "dualmirakl@runpod" 2>/dev/null
git -C "$PROJ" config user.name  "dualmirakl"        2>/dev/null

# ── Model path check ─────────────────────────────────────────────────────────
MISSING=0
for MODEL_DIR in glm-5 deepseek-v3.2-special e5-small-v2; do
    if [ ! -f "/per.volume/huggingface/hub/${MODEL_DIR}/config.json" ]; then
        echo "[autostart] WARN: model not found: /per.volume/huggingface/hub/${MODEL_DIR}"
        MISSING=$((MISSING + 1))
    fi
done
if [ "$MISSING" -gt 0 ]; then
    echo "[autostart] $MISSING model(s) missing — vLLM startup will fail."
    echo "[autostart] Run audit_env.sh to check model files."
fi

# ── Start vLLM servers ────────────────────────────────────────────────────────
# start_all.sh is sequential and blocks until both GPUs are ready (~3 min each)
echo "[autostart] Starting vLLM servers..."
cd "$PROJ" && bash start_all.sh
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo "[autostart] ERROR: vLLM startup failed (exit $STATUS)"
    echo "[autostart] Check: $PROJ/logs/gpu0.log and $PROJ/logs/gpu1.log"
    exit 1
fi

echo "[autostart] Both vLLM servers up."

# ── Start gateway ─────────────────────────────────────────────────────────────
echo "[autostart] Starting gateway on :9000..."
nohup uvicorn gateway:app \
    --host 0.0.0.0 \
    --port 9000 \
    >> "$PROJ/logs/gateway.log" 2>&1 &
GATEWAY_PID=$!
echo "[autostart] Gateway PID: $GATEWAY_PID"
echo "$GATEWAY_PID" >> "$PROJ/logs/pids.txt"

echo "[autostart] Done — all services running. $(date)"
