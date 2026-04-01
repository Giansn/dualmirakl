#!/bin/bash
# Start authority + swarm + gateway with health checks
# Authority and swarm start in PARALLEL (different GPUs)
# Model configs: config/authority.env  config/swarm.env

PROJ="${DUALMIRAKL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJ"
mkdir -p logs

AUTHORITY_PORT="${AUTHORITY_PORT:-8000}"
SWARM_PORT="${SWARM_PORT:-8001}"
GATEWAY_PORT="${GATEWAY_PORT:-9000}"

# --- Preflight ---
echo "[dualmirakl] Preflight checks..."
if ! python3 -c "import httpx, h2, simpy" 2>/dev/null; then
  pip install -q "httpx>=0.27.0" "h2>=4.0.0" "simpy>=4.1.1" \
    || { echo "[ERROR] Dependency install failed."; exit 1; }
fi

PORTS_IN_USE=0
for PORT in $AUTHORITY_PORT $SWARM_PORT $GATEWAY_PORT; do
  if ss -tlnp 2>/dev/null | grep -q ":${PORT} " || \
     (echo >/dev/tcp/localhost/$PORT) 2>/dev/null; then
    echo "[WARN] Port ${PORT} already in use."
    PORTS_IN_USE=1
  fi
done
if [ $PORTS_IN_USE -eq 1 ]; then
  echo ""
  echo "Services may already be running. Options:"
  echo "  bash scripts/stop_all.sh     # stop first, then re-run"
  echo "  bash scripts/go.sh --restart # stop + start in one step"
  echo "  bash scripts/status.sh       # check what's running"
  exit 1
fi

# --- Authority + Swarm in PARALLEL (different GPUs) ---
echo "[dualmirakl] Starting authority (GPU 0) + swarm (GPU 1) in parallel..."
mv logs/authority.log logs/authority.last 2>/dev/null || true
mv logs/swarm.log logs/swarm.last 2>/dev/null || true

bash scripts/start_authority.sh > logs/authority.log 2>&1 &
AUTH_PID=$!
echo "  authority PID: $AUTH_PID"

bash scripts/start_swarm.sh > logs/swarm.log 2>&1 &
SWARM_PID=$!
echo "  swarm PID: $SWARM_PID"

# Poll both simultaneously
AUTH_READY=0; SWARM_READY=0
for i in $(seq 1 60); do
  if [ $AUTH_READY -eq 0 ]; then
    if curl -sf http://localhost:${AUTHORITY_PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"authority","prompt":"hi","max_tokens":1}' -o /dev/null 2>&1; then
      echo "  authority :${AUTHORITY_PORT}: READY"
      AUTH_READY=1
    fi
  fi
  if [ $SWARM_READY -eq 0 ]; then
    if curl -sf http://localhost:${SWARM_PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"swarm","prompt":"hi","max_tokens":1}' -o /dev/null 2>&1; then
      echo "  swarm :${SWARM_PORT}: READY"
      SWARM_READY=1
    fi
  fi
  [ $AUTH_READY -eq 1 ] && [ $SWARM_READY -eq 1 ] && break
  sleep 3
done

if [ $AUTH_READY -eq 0 ]; then
  echo "[ERROR] Authority did not become ready. Check logs/authority.log"
  kill $AUTH_PID $SWARM_PID 2>/dev/null; exit 1
fi
if [ $SWARM_READY -eq 0 ]; then
  echo "[ERROR] Swarm did not become ready. Check logs/swarm.log"
  kill $AUTH_PID $SWARM_PID 2>/dev/null; exit 1
fi

# --- Gateway (port 9000) ---
echo "[dualmirakl] Starting gateway on port $GATEWAY_PORT..."
mv logs/gateway.log logs/gateway.last 2>/dev/null || true
bash scripts/start_gateway.sh > logs/gateway.log 2>&1 &
GW_PID=$!
echo "  gateway PID: $GW_PID"

READY=0
for i in $(seq 1 20); do
  if curl -sf http://localhost:${GATEWAY_PORT}/health -o /dev/null 2>&1; then
    echo "  gateway :${GATEWAY_PORT}: READY"
    READY=1; break
  fi
  sleep 1
done
if [ $READY -eq 0 ]; then
  echo "  [WARN] Gateway health check timed out (may still be loading)"
fi

# --- Final validation ---
echo "[dualmirakl] Validating inference..."
sleep 2
for svc in "authority:${AUTHORITY_PORT}" "swarm:${SWARM_PORT}"; do
  NAME="${svc%%:*}"; PORT="${svc##*:}"
  if ! curl -sf "http://localhost:${PORT}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${NAME}\",\"prompt\":\"test\",\"max_tokens\":1}" -o /dev/null 2>&1; then
    echo "  [WARN] $NAME passed health check but inference failed — may need more warmup"
  else
    echo "  $NAME: inference OK"
  fi
done

echo ""
echo "All services running."
echo "  authority : http://localhost:${AUTHORITY_PORT}/v1  (GPU 0)"
echo "  swarm     : http://localhost:${SWARM_PORT}/v1  (GPU 1)"
echo "  gateway   : http://localhost:${GATEWAY_PORT}    (CPU — UI + API)"

# FLAME GPU 2 (optional 3rd GPU)
if [ "${FLAME_ENABLED:-0}" = "1" ]; then
  echo "  flame     : GPU ${FLAME_GPU:-2}  (FLAME GPU 2 — ${FLAME_N_POPULATION:-10000} population agents)"
  echo ""
  echo "3-GPU mode active. FLAME population dynamics will run during simulation."
else
  echo ""
  echo "2-GPU mode (standard). Set FLAME_ENABLED=1 in .env for 3-GPU mode."
fi

echo "Logs: logs/authority.log  logs/swarm.log  logs/gateway.log"
echo "Stop: bash scripts/stop_all.sh"
echo "$AUTH_PID $SWARM_PID $GW_PID" > logs/pids.txt
