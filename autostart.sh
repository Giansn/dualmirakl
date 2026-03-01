#!/bin/bash
# =============================================================================
# dualmirakl autostart — environment setup only
# Called by /post_start.sh on pod boot.
# Servers are started manually: bash start_all.sh
# =============================================================================

PROJ=/per.volume/dualmirakl
mkdir -p "$PROJ/logs"

LOG="$PROJ/logs/autostart.log"
echo "" >> "$LOG"
echo "════════════════════════════════════════" >> "$LOG"
echo "[autostart] $(date)" >> "$LOG"
echo "════════════════════════════════════════" >> "$LOG"
exec >> "$LOG" 2>&1

# ── Environment ───────────────────────────────────────────────────────────────
export HF_HOME=/per.volume/huggingface
export PATH="/usr/local/bin:$PATH"

[ -f "$PROJ/.env" ] && source "$PROJ/.env" && echo "[autostart] .env loaded"

# ── Git identity ──────────────────────────────────────────────────────────────
git -C "$PROJ" config user.email "dualmirakl@runpod" 2>/dev/null
git -C "$PROJ" config user.name  "dualmirakl"        2>/dev/null

echo "[autostart] Ready. Start servers with: bash /per.volume/dualmirakl/start_all.sh"
