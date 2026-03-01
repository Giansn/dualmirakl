#!/bin/bash
# Start both vLLM servers in background.
# Sequential launch is intentional: both models load from /per.volume NFS.
# Parallel reads over a shared network path cause I/O contention; stagger avoids it.

mkdir -p logs

# --- [1] Preflight: dependencies ---
echo "[dualmirakl] Preflight checks..."
if ! python3 -c "import httpx, h2, simpy" 2>/dev/null; then
  echo "[WARN] Missing dependencies — installing..."
  pip install -q "httpx>=0.27.0" "h2>=4.0.0" "simpy>=4.1.1" \
    || { echo "[ERROR] Dependency install failed. Aborting."; exit 1; }
fi

# --- [2] Port collision check ---
for PORT in 8000 8001; do
  if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "[ERROR] Port ${PORT} already in use. Run: bash stop_all.sh"
    exit 1
  fi
done

# --- [3] Launch GPU0 (GLM-5) ---
echo "[dualmirakl] Starting GPU0 (GLM-5) on port 8000..."
mv logs/gpu0.log logs/gpu0.last 2>/dev/null
bash start_gpu0.sh > logs/gpu0.log 2>&1 &
GPU0_PID=$!
echo "  PID: $GPU0_PID"

# Poll completions endpoint directly — proves HTTP server AND engine are ready
READY=0
for i in $(seq 1 60); do
  if curl -sf "http://localhost:8000/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"glm-5","prompt":"hi","max_tokens":1}' \
      -o /dev/null 2>&1; then
    echo "  port 8000: READY"
    READY=1
    break
  fi
  sleep 3
done
if [ $READY -eq 0 ]; then
  echo "[ERROR] GPU0 (port 8000) did not become ready after 3 min."
  echo "  Check logs/gpu0.log for details."
  nvidia-smi >> logs/gpu0.log 2>&1
  kill $GPU0_PID 2>/dev/null
  exit 1
fi

# --- [4] Launch GPU1 (DeepSeek-V3.2-Special) ---
echo "[dualmirakl] Starting GPU1 (DeepSeek-V3.2-Special) on port 8001..."
mv logs/gpu1.log logs/gpu1.last 2>/dev/null
bash start_gpu1.sh > logs/gpu1.log 2>&1 &
GPU1_PID=$!
echo "  PID: $GPU1_PID"

READY=0
for i in $(seq 1 60); do
  if curl -sf "http://localhost:8001/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"deepseek-v3.2-special","prompt":"hi","max_tokens":1}' \
      -o /dev/null 2>&1; then
    echo "  port 8001: READY"
    READY=1
    break
  fi
  sleep 3
done
if [ $READY -eq 0 ]; then
  echo "[ERROR] GPU1 (port 8001) did not become ready after 3 min."
  echo "  Check logs/gpu1.log for details."
  nvidia-smi >> logs/gpu1.log 2>&1
  kill $GPU0_PID $GPU1_PID 2>/dev/null
  exit 1
fi

echo ""
echo "Both servers running. Logs: logs/gpu0.log  logs/gpu1.log"
echo "Stop with: bash stop_all.sh"
echo "$GPU0_PID $GPU1_PID" > logs/pids.txt
echo "PIDs saved to logs/pids.txt"
