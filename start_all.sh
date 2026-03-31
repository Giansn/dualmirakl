#!/bin/bash
# Start authority + swarm + gateway
# Model configs: models/authority.env  models/swarm.env

PROJ="${DUALMIRAKL_ROOT:-/workspace/dualmirakl}"
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

for PORT in $AUTHORITY_PORT $SWARM_PORT $GATEWAY_PORT; do
  if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "[ERROR] Port ${PORT} already in use. Run: bash stop_all.sh"
    exit 1
  fi
done

# --- Authority (GPU 0, port 8000) ---
echo "[dualmirakl] Starting authority on port $AUTHORITY_PORT..."
mv logs/authority.log logs/authority.last 2>/dev/null || true
bash start_authority.sh > logs/authority.log 2>&1 &
AUTH_PID=$!
echo "  PID: $AUTH_PID"

READY=0
for i in $(seq 1 60); do
  if curl -sf http://localhost:${AUTHORITY_PORT}/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"authority","prompt":"hi","max_tokens":1}' -o /dev/null 2>&1; then
    echo "  port ${AUTHORITY_PORT}: READY"
    READY=1; break
  fi
  sleep 3
done
if [ $READY -eq 0 ]; then
  echo "[ERROR] Authority did not become ready. Check logs/authority.log"
  kill $AUTH_PID 2>/dev/null; exit 1
fi

# --- Swarm (GPU 1, port 8001) ---
echo "[dualmirakl] Starting swarm on port $SWARM_PORT..."
mv logs/swarm.log logs/swarm.last 2>/dev/null || true
bash start_swarm.sh > logs/swarm.log 2>&1 &
SWARM_PID=$!
echo "  PID: $SWARM_PID"

READY=0
for i in $(seq 1 60); do
  if curl -sf http://localhost:${SWARM_PORT}/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"swarm","prompt":"hi","max_tokens":1}' -o /dev/null 2>&1; then
    echo "  port ${SWARM_PORT}: READY"
    READY=1; break
  fi
  sleep 3
done
if [ $READY -eq 0 ]; then
  echo "[ERROR] Swarm did not become ready. Check logs/swarm.log"
  kill $AUTH_PID $SWARM_PID 2>/dev/null; exit 1
fi

# --- Gateway (port 9000) ---
echo "[dualmirakl] Starting gateway on port $GATEWAY_PORT..."
mv logs/gateway.log logs/gateway.last 2>/dev/null || true
bash start_gateway.sh > logs/gateway.log 2>&1 &
GW_PID=$!
echo "  PID: $GW_PID"

echo ""
echo "All services running."
echo "  authority : http://localhost:${AUTHORITY_PORT}/v1  (GPU 0)"
echo "  swarm     : http://localhost:${SWARM_PORT}/v1  (GPU 1)"
echo "  gateway   : http://localhost:${GATEWAY_PORT}/v1  (CPU — unified + embeddings)"

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
echo "Stop: bash stop_all.sh"
echo "$AUTH_PID $SWARM_PID $GW_PID" > logs/pids.txt
