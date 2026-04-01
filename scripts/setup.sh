#!/bin/bash
# =============================================================================
# dualmirakl setup — fresh RunPod pod (generic Docker, no custom image)
#
# Usage:  bash scripts/setup.sh [--force]
#
# Idempotent: skips pip install + model download if already done and
# requirements.txt hasn't changed. Use --force to re-run everything.
# =============================================================================

set -e

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"

FORCE=0
[ "$1" = "--force" ] && FORCE=1

STAMP="logs/.setup_done"
REQ_HASH=$(md5sum requirements.txt 2>/dev/null | cut -d' ' -f1 || echo "none")

echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║   dualmirakl — RunPod setup      ║"
echo "  ╚══════════════════════════════════╝"
echo ""

# ── 0. Check if already set up ──────────────────────────────────────────────
if [ $FORCE -eq 0 ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$REQ_HASH" ]; then
    echo "[setup] Already set up (requirements unchanged)."
    echo "[setup] Use --force to re-run.  Stamp: $STAMP"
    echo ""
    exit 0
fi

# ── 1. Create .env from example ──────────────────────────────────────────────
if [ ! -f .env ] && [ -f config/.env.example ]; then
    cp config/.env.example .env
    echo "[setup] .env created from config/.env.example"
    echo "[setup] Edit .env to set GPU assignments and model paths"
fi

if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ── 2. Install Python dependencies ──────────────────────────────────────────
echo "[setup] Installing Python dependencies..."
pip install -q -r requirements.txt
echo "[setup] Core dependencies installed"

if [ -f requirements-ml.txt ]; then
    pip install -q -r requirements-ml.txt 2>/dev/null && \
        echo "[setup] ML extensions installed" || \
        echo "[setup] ML extensions skipped (optional)"
fi

# ── 3. Download models ──────────────────────────────────────────────────────
echo "[setup] Downloading models..."
bash scripts/pull_models.sh

# ── 4. Create runtime directories ───────────────────────────────────────────
mkdir -p logs data

# ── 5. Git identity (for RunPod pods) ───────────────────────────────────────
git config user.email "dualmirakl@runpod" 2>/dev/null || true
git config user.name  "dualmirakl"        2>/dev/null || true

# ── 6. Write stamp ─────────────────────────────────────────────────────────
echo "$REQ_HASH" > "$STAMP"

# ── 7. Environment audit ────────────────────────────────────────────────────
echo ""
bash scripts/audit_env.sh

echo ""
echo "  Setup complete."
echo "  Start servers:  bash scripts/go.sh"
echo "  Run tests:      python -m pytest tests/ -v"
echo ""
