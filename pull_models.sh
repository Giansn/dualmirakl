#!/bin/bash
# =============================================================================
# pull_models.sh — download required models to $HF_HOME
#
# Usage:
#   bash pull_models.sh                  # download all models
#   bash pull_models.sh --embeddings     # embeddings only (e5-small-v2)
#   bash pull_models.sh --authority      # authority model only
#   bash pull_models.sh --swarm          # swarm model only
#
# Requires: huggingface-cli (pip install huggingface-hub)
# Set HF_TOKEN env var for gated models.
# =============================================================================
set -e

HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HOME

PROJ="$(cd "$(dirname "$0")" && pwd)"

# ── Model definitions ────────────────────────────────────────────────────────

EMBED_REPO="intfloat/e5-small-v2"
EMBED_LOCAL="$HF_HOME/hub/e5-small-v2"

# Read authority/swarm models from their .env files if available
if [ -f "$PROJ/models/authority.env" ]; then
    source "$PROJ/models/authority.env"
    AUTHORITY_REPO="${AUTHORITY_HF_REPO:-}"
    AUTHORITY_LOCAL="$MODEL"
fi

if [ -f "$PROJ/models/swarm.env" ]; then
    source "$PROJ/models/swarm.env"
    SWARM_REPO="${SWARM_HF_REPO:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
    SWARM_LOCAL="$MODEL"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

check_hf_cli() {
    if ! command -v huggingface-cli &>/dev/null; then
        echo "[pull_models] huggingface-cli not found. Installing..."
        pip install -q huggingface-hub
    fi
}

pull_model() {
    local repo="$1"
    local local_path="$2"
    local label="$3"

    if [ -z "$repo" ]; then
        echo "[$label] No HF repo configured — skipping"
        return 0
    fi

    if [ -d "$local_path" ] && [ -f "$local_path/config.json" ]; then
        echo "[$label] Already present at $local_path — skipping"
        return 0
    fi

    echo "[$label] Downloading $repo → $local_path ..."
    mkdir -p "$local_path"
    huggingface-cli download "$repo" --local-dir "$local_path" \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
    echo "[$label] Done."
}

# ── Main ──────────────────────────────────────────────────────────────────────

TARGET="${1:-all}"

check_hf_cli

case "$TARGET" in
    --embeddings|-e)
        pull_model "$EMBED_REPO" "$EMBED_LOCAL" "embeddings"
        ;;
    --authority|-a)
        pull_model "$AUTHORITY_REPO" "$AUTHORITY_LOCAL" "authority"
        ;;
    --swarm|-s)
        pull_model "$SWARM_REPO" "$SWARM_LOCAL" "swarm"
        ;;
    all|*)
        echo "[pull_models] Downloading all models to $HF_HOME ..."
        echo ""
        pull_model "$EMBED_REPO" "$EMBED_LOCAL" "embeddings"
        pull_model "$AUTHORITY_REPO" "$AUTHORITY_LOCAL" "authority"
        pull_model "$SWARM_REPO" "$SWARM_LOCAL" "swarm"
        echo ""
        echo "[pull_models] All downloads complete."
        ;;
esac
