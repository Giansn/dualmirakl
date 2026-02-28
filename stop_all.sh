#!/bin/bash
if [ -f logs/pids.txt ]; then
  PIDS=$(cat logs/pids.txt)
  echo "Stopping vLLM servers (PIDs: $PIDS)..."
  kill $PIDS 2>/dev/null
  echo "Done."
else
  echo "No PID file found. Killing all vllm processes..."
  pkill -f "vllm.entrypoints" && echo "Done." || echo "None running."
fi
