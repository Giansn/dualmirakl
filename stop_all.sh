#!/bin/bash
if [ -f logs/pids.txt ]; then
  PIDS=$(cat logs/pids.txt)
  echo "Stopping servers (PIDs: $PIDS)..."
  kill $PIDS 2>/dev/null
else
  echo "No PID file found. Killing by process name..."
  pkill -f "vllm.entrypoints" 2>/dev/null && echo "vLLM stopped." || echo "No vLLM running."
  pkill -f "uvicorn gateway" 2>/dev/null && echo "Gateway stopped." || echo "No gateway running."
fi
rm -f logs/pids.txt
echo "Done."
