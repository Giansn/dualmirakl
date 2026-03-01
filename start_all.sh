#!/bin/bash
# Start authority + swarm + gateway
# Model configs: models/authority.env  models/swarm.env

PROJ=/per.volume/dualmirakl
cd "$PROJ"
mkdir -p logs

# --- Preflight ---
echo "[dualmirakl] Preflight checks..."
if ! python3 -c "import httpx, h2, simpy" 2>/dev/null; then
  pip install -q "httpx>=0.27.0" "h2>=4.0.0" "simpy>=4.1.1" \
    || { echo "[ERROR] Dependency install failed."; exit 1; }
fi

for PORT in 8000 8001 9000; do
  if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "[ERROR] Port ${PORT} already in use. Run: bash stop_all.sh"
    exit 1
  fi
done

# --- Authority (GPU 0, port 8000) ---
echo "[dualmirakl] Starting authority on port 8000..."
mv logs/authority.log logs/authority.last 2>/dev/null || true
bash start_authority.sh > logs/authority.log 2>&1 &
AUTH_PID=$!
echo "  PID: $AUTH_PID"

READY=0
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"authority","prompt":"hi","max_tokens":1}' -o /dev/null 2>&1; then
    echo "  port 8000: READY"
    READY=1; break
  fi
  sleep 3
done
if [ $READY -eq 0 ]; then
  echo "[ERROR] Authority did not become ready. Check logs/authority.log"
  kill $AUTH_PID 2>/dev/null; exit 1
fi

# --- Swarm (GPU 1, port 8001) ---
echo "[dualmirakl] Starting swarm on port 8001..."
mv logs/swarm.log logs/swarm.last 2>/dev/null || true
bash start_swarm.sh > logs/swarm.log 2>&1 &
SWARM_PID=$!
echo "  PID: $SWARM_PID"

READY=0
for i in $(seq 1 60); do
  if curl -sf http://localhost:8001/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"swarm","prompt":"hi","max_tokens":1}' -o /dev/null 2>&1; then
    echo "  port 8001: READY"
    READY=1; break
  fi
  sleep 3
done
if [ $READY -eq 0 ]; then
  echo "[ERROR] Swarm did not become ready. Check logs/swarm.log"
  kill $AUTH_PID $SWARM_PID 2>/dev/null; exit 1
fi

# --- Gateway (port 9000) ---
echo "[dualmirakl] Starting gateway on port 9000..."
mv logs/gateway.log logs/gateway.last 2>/dev/null || true
bash start_gateway.sh > logs/gateway.log 2>&1 &
GW_PID=$!
echo "  PID: $GW_PID"

echo ""
echo "All services running."
echo "  authority : http://localhost:8000/v1"
echo "  swarm     : http://localhost:8001/v1"
echo "  gateway   : http://localhost:9000/v1  (unified + embeddings)"
echo "Logs: logs/authority.log  logs/swarm.log  logs/gateway.log"
echo "Stop: bash stop_all.sh"
echo "$AUTH_PID $SWARM_PID $GW_PID" > logs/pids.txt
