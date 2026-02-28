#!/bin/bash
# Start both vLLM servers in background, tail logs

mkdir -p logs

# --- [1] Check h2 is importable (HTTP/2 support for orchestrator) ---
if ! python3 -c "import h2" 2>/dev/null; then
  echo "[WARN] h2 not found — installing for HTTP/2 support..."
  pip install -q "h2>=4.0.0" || { echo "[ERROR] h2 install failed. Aborting."; exit 1; }
fi

# --- [2] Stagger GPU launches to avoid simultaneous VRAM pressure ---
echo "[dualmirakl] Starting GPU0 (Command-R 7B) on port 8000..."
bash start_gpu0.sh > logs/gpu0.log 2>&1 &
GPU0_PID=$!
echo "  PID: $GPU0_PID"

echo "  Waiting for GPU0 before launching GPU1..."
READY=0
for i in $(seq 1 60); do
  if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
    echo "  port 8000: READY"
    READY=1
    break
  fi
  sleep 3
done
if [ $READY -eq 0 ]; then
  echo "[ERROR] GPU0 (port 8000) did not become healthy after 3 min."
  echo "  Check logs/gpu0.log for details."
  kill $GPU0_PID 2>/dev/null
  exit 1
fi

echo "[dualmirakl] Starting GPU1 (Qwen 2.5 7B) on port 8001..."
bash start_gpu1.sh > logs/gpu1.log 2>&1 &
GPU1_PID=$!
echo "  PID: $GPU1_PID"

# --- [3] Health check with explicit timeout failure for GPU1 ---
READY=0
for i in $(seq 1 60); do
  if curl -sf "http://localhost:8001/health" > /dev/null 2>&1; then
    echo "  port 8001: READY"
    READY=1
    break
  fi
  sleep 3
done
if [ $READY -eq 0 ]; then
  echo "[ERROR] GPU1 (port 8001) did not become healthy after 3 min."
  echo "  Check logs/gpu1.log for details."
  kill $GPU0_PID $GPU1_PID 2>/dev/null
  exit 1
fi

echo ""
echo "Both servers running. Logs: logs/gpu0.log  logs/gpu1.log"
echo "Stop with: bash stop_all.sh"
echo "$GPU0_PID $GPU1_PID" > logs/pids.txt
echo "PIDs saved to logs/pids.txt"
