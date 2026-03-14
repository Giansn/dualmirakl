#!/bin/bash
# =============================================================================
# chmod.sh — set correct permissions on all project scripts
# Run after cloning or syncing on a fresh instance.
# =============================================================================
set -e

PROJ="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ"

echo "[chmod] Setting executable permissions on shell scripts..."

chmod +x \
    start_authority.sh \
    start_swarm.sh \
    start_gateway.sh \
    start_all.sh \
    stop_all.sh \
    autostart.sh \
    audit_env.sh \
    pull_models.sh \
    chmod.sh \
    docker-post-start.sh \
    entrypoint.sh \
    2>/dev/null

echo "[chmod] Done."
