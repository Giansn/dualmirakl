#!/bin/bash
# =============================================================================
# entrypoint.sh — generic container entrypoint (works on RunPod, Docker, K8s)
#
# Modes (set ENTRYPOINT_MODE env var):
#   all       — start authority + swarm + gateway (default)
#   authority — start authority vLLM server only
#   swarm     — start swarm vLLM server only
#   gateway   — start gateway only
#   sim       — run simulation (requires vLLM servers to be reachable)
#   shell     — drop to bash (for debugging)
# =============================================================================
set -e

PROJ="${DUALMIRAKL_ROOT:-/app}"
MODE="${ENTRYPOINT_MODE:-all}"

cd "$PROJ"

# Source .env if present
[ -f .env ] && set -a && source .env && set +a

case "$MODE" in
    authority)
        exec bash start_authority.sh
        ;;
    swarm)
        exec bash start_swarm.sh
        ;;
    gateway)
        exec bash start_gateway.sh
        ;;
    sim)
        exec python -m simulation.sim_loop
        ;;
    shell)
        exec bash
        ;;
    all)
        exec bash start_all.sh
        ;;
    *)
        # Default: keep container alive with SSH accessible.
        # Start services manually: bash start_all.sh
        echo "[dualmirakl] Ready. Run 'bash start_all.sh' to start services."
        exec sleep infinity
        ;;
esac
