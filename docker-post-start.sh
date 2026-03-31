#!/bin/bash
# =============================================================================
# /post_start.sh — baked into the container image
# Runs after SSH is up on every pod boot (called by RunPod's /start.sh).
# =============================================================================

PROJ=/workspace/dualmirakl
REPO=https://github.com/Giansn/dualmirakl.git
BRANCH=runpod

mkdir -p "$PROJ/logs"

# ── First-boot: clone project to persistent volume ────────────────────────────
if [ ! -d "$PROJ/.git" ]; then
    echo "[post_start] First boot — cloning dualmirakl to $PROJ..."

    # Use GITHUB_TOKEN env var if set (needed for private repo)
    if [ -n "$GITHUB_TOKEN" ]; then
        CLONE_URL="https://${GITHUB_TOKEN}@github.com/Giansn/dualmirakl.git"
    else
        CLONE_URL="$REPO"
    fi

    git clone "$CLONE_URL" "$PROJ" --branch "$BRANCH" --single-branch
    git -C "$PROJ" config user.email "dualmirakl@runpod"
    git -C "$PROJ" config user.name  "dualmirakl"

    # Strip token from stored remote URL
    git -C "$PROJ" remote set-url origin "$REPO"

    echo "[post_start] Clone done."
fi

# ── Create .env if missing ────────────────────────────────────────────────────
if [ ! -f "$PROJ/.env" ] && [ -f "$PROJ/.env.example" ]; then
    cp "$PROJ/.env.example" "$PROJ/.env"
    echo "[post_start] .env created from .env.example"
fi

# ── Launch stack in background ────────────────────────────────────────────────
chmod +x "$PROJ/autostart.sh"
nohup bash "$PROJ/autostart.sh" &
echo "[post_start] dualmirakl autostart launched (PID $!)"
echo "[post_start] Monitor: tail -f $PROJ/logs/autostart.log"
