#!/bin/bash
# =============================================================================
# dualmirakl setup — fresh RunPod pod (generic Docker, no custom image)
#
# Usage:  bash scripts/setup.sh
# =============================================================================

set -e

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"

echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║   dualmirakl — RunPod setup      ║"
echo "  ╚══════════════════════════════════╝"
echo ""

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

# ── 6. Environment audit ────────────────────────────────────────────────────
echo ""
bash scripts/audit_env.sh

echo ""
echo "  Setup complete."
echo "  Start servers:  bash scripts/start_all.sh"
echo "  Run tests:      python -m pytest tests/ -v"
echo ""
