#!/bin/bash
# Start both vLLM servers in background, tail logs

mkdir -p logs

echo "[dualmirakl] Starting GPU0 (Command-R 7B) on port 8000..."
bash start_gpu0.sh > logs/gpu0.log 2>&1 &
GPU0_PID=$!
echo "  PID: $GPU0_PID"

echo "[dualmirakl] Starting GPU1 (Qwen 2.5 7B) on port 8001..."
bash start_gpu1.sh > logs/gpu1.log 2>&1 &
GPU1_PID=$!
echo "  PID: $GPU1_PID"

echo ""
echo "Waiting for servers to be ready..."
for PORT in 8000 8001; do
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
      echo "  port $PORT: READY"
      break
    fi
    sleep 3
  done
done

echo ""
echo "Both servers running. Logs: logs/gpu0.log  logs/gpu1.log"
echo "Stop with: kill $GPU0_PID $GPU1_PID"
echo "PIDs saved to logs/pids.txt"
echo "$GPU0_PID $GPU1_PID" > logs/pids.txt
