#!/bin/bash
# =============================================================================
# dualmirakl — quick health check
# Usage: bash scripts/status.sh
# =============================================================================

PROJ="${DUALMIRAKL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJ"

AUTHORITY_PORT="${AUTHORITY_PORT:-8000}"
SWARM_PORT="${SWARM_PORT:-8001}"
GATEWAY_PORT="${GATEWAY_PORT:-9000}"

GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; DIM='\033[2m'; NC='\033[0m'
OK="${GREEN}✓${NC}"; FAIL="${RED}✗${NC}"

echo ""
echo -e "  ${CYAN}dualmirakl${NC} — status"
echo "  ─────────────────────"

DOWN=0

# Authority
if curl -sf "http://localhost:${AUTHORITY_PORT}/v1/models" -o /dev/null 2>&1; then
  MODEL=$(curl -sf "http://localhost:${AUTHORITY_PORT}/v1/models" 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['data'][0]['id'])" 2>/dev/null || echo "?")
  echo -e "  ${OK}  authority  :${AUTHORITY_PORT}  ${DIM}${MODEL}${NC}"
else
  echo -e "  ${FAIL}  authority  :${AUTHORITY_PORT}  ${RED}down${NC}"
  DOWN=$((DOWN+1))
fi

# Swarm
if curl -sf "http://localhost:${SWARM_PORT}/v1/models" -o /dev/null 2>&1; then
  MODEL=$(curl -sf "http://localhost:${SWARM_PORT}/v1/models" 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['data'][0]['id'])" 2>/dev/null || echo "?")
  echo -e "  ${OK}  swarm      :${SWARM_PORT}  ${DIM}${MODEL}${NC}"
else
  echo -e "  ${FAIL}  swarm      :${SWARM_PORT}  ${RED}down${NC}"
  DOWN=$((DOWN+1))
fi

# Gateway
if curl -sf "http://localhost:${GATEWAY_PORT}/health" -o /dev/null 2>&1; then
  echo -e "  ${OK}  gateway    :${GATEWAY_PORT}  ${DIM}e5 + proxy + UI${NC}"
else
  echo -e "  ${FAIL}  gateway    :${GATEWAY_PORT}  ${RED}down${NC}"
  DOWN=$((DOWN+1))
fi

# FLAME
if [ "${FLAME_ENABLED:-0}" = "1" ]; then
  echo -e "  ${OK}  flame      GPU ${FLAME_GPU:-2}  ${DIM}population dynamics${NC}"
else
  echo -e "  ${DIM}  ·  flame      off${NC}"
fi

# GPU
echo ""
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=', ' read -r idx name util mem_used mem_total temp; do
      PCTV=$((mem_used * 100 / (mem_total + 1)))
      echo -e "  GPU ${idx}  ${name}  ${CYAN}${util}%${NC} util  ${mem_used}/${mem_total} MB  ${temp}°C"
    done
else
  echo -e "  ${DIM}nvidia-smi not available${NC}"
fi

# Setup stamp
echo ""
if [ -f logs/.setup_done ]; then
  echo -e "  ${OK}  setup complete"
else
  echo -e "  ${FAIL}  setup not run  ${DIM}→ bash scripts/go.sh${NC}"
fi

# Verdict
echo ""
if [ $DOWN -eq 0 ]; then
  echo -e "  ${GREEN}READY${NC} — all services up"
else
  echo -e "  ${RED}NOT READY${NC} — ${DOWN} service(s) down"
  echo -e "  ${DIM}Start: bash scripts/go.sh${NC}"
fi
echo ""
