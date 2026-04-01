#!/bin/bash
# =============================================================================
# dualmirakl — single entry point
#
# Usage:  bash scripts/go.sh [--restart] [--setup] [--status]
#
# Detects what's needed and does it:
#   1. Creates .env if missing
#   2. Runs setup if not done (or --setup to force)
#   3. Starts services if not running (or --restart to force)
#   4. Prints status
# =============================================================================

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"

# Parse flags
DO_RESTART=0; DO_SETUP=0; STATUS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --restart) DO_RESTART=1 ;;
    --setup)   DO_SETUP=1 ;;
    --status)  STATUS_ONLY=1 ;;
  esac
done

# ── Status only ─────────────────────────────────────────────────────────────
if [ $STATUS_ONLY -eq 1 ]; then
  bash scripts/status.sh
  exit $?
fi

echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║   dualmirakl — go               ║"
echo "  ╚══════════════════════════════════╝"
echo ""

# ── 1. Ensure .env ──────────────────────────────────────────────────────────
if [ ! -f .env ] && [ -f config/.env.example ]; then
  cp config/.env.example .env
  echo "[go] .env created from template"
fi
[ -f .env ] && { set -a; source .env; set +a; }

# ── 2. Setup (idempotent) ──────────────────────────────────────────────────
if [ $DO_SETUP -eq 1 ]; then
  echo "[go] Forced setup..."
  bash scripts/setup.sh --force || { echo "[go] Setup failed."; exit 1; }
elif [ ! -f logs/.setup_done ]; then
  echo "[go] First run — running setup..."
  bash scripts/setup.sh || { echo "[go] Setup failed."; exit 1; }
else
  REQ_HASH=$(md5sum requirements.txt 2>/dev/null | cut -d' ' -f1 || echo "none")
  STAMP_HASH=$(cat logs/.setup_done 2>/dev/null)
  if [ "$REQ_HASH" != "$STAMP_HASH" ]; then
    echo "[go] requirements.txt changed — re-running setup..."
    bash scripts/setup.sh --force || { echo "[go] Setup failed."; exit 1; }
  else
    echo "[go] Setup up to date."
  fi
fi

# ── 3. Services ─────────────────────────────────────────────────────────────
AUTHORITY_PORT="${AUTHORITY_PORT:-8000}"
SWARM_PORT="${SWARM_PORT:-8001}"
GATEWAY_PORT="${GATEWAY_PORT:-9000}"

services_running() {
  ss -tlnp 2>/dev/null | grep -q ":${AUTHORITY_PORT} " && \
  ss -tlnp 2>/dev/null | grep -q ":${SWARM_PORT} " && \
  ss -tlnp 2>/dev/null | grep -q ":${GATEWAY_PORT} "
}

if [ $DO_RESTART -eq 1 ]; then
  echo "[go] Restarting services..."
  bash scripts/stop_all.sh 2>/dev/null
  sleep 2
  bash scripts/start_all.sh
elif services_running; then
  echo "[go] Services already running."
else
  echo "[go] Starting services..."
  bash scripts/start_all.sh
fi

# ── 4. Status ───────────────────────────────────────────────────────────────
echo ""
bash scripts/status.sh
