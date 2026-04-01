#!/bin/bash
# Stop all dualmirakl services safely

if [ -f logs/pids.txt ]; then
  echo "Stopping servers..."
  while read -r PID; do
    [ -z "$PID" ] && continue
    if [ -d "/proc/$PID" ]; then
      CMD=$(cat "/proc/$PID/cmdline" 2>/dev/null | tr '\0' ' ')
      if echo "$CMD" | grep -qiE "vllm|gateway|uvicorn"; then
        kill "$PID" 2>/dev/null && echo "  killed PID $PID"
      else
        echo "  [WARN] PID $PID is not a dualmirakl process, skipping"
      fi
    fi
  done <<< "$(cat logs/pids.txt | tr ' ' '\n')"
else
  echo "No PID file found. Killing by process name..."
  pkill -f "vllm.entrypoints" 2>/dev/null && echo "vLLM stopped." || echo "No vLLM running."
  pkill -f "uvicorn gateway" 2>/dev/null && echo "Gateway stopped." || echo "No gateway running."
fi
rm -f logs/pids.txt
echo "Done."
