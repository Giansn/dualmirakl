#!/bin/bash
PROJ="${DUALMIRAKL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "--- [SYSTEM AUDIT] ---"
date
echo ""

echo "--- [DISK] ---"
if mountpoint -q /per.volume 2>/dev/null; then
  echo "  /per.volume : $(df -h /per.volume | tail -n 1 | awk '{print $3 " used of " $2 " (" $5 ")"}')"
else
  echo "  /per.volume : not mounted (non-RunPod environment)"
fi
HF_DIR="${HF_HOME:-/workspace/huggingface}"
echo "  hub         : $(du -sh "$HF_DIR/hub" 2>/dev/null | cut -f1 || echo 'not found')"
echo ""

echo "--- [GPU] ---"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s — %s | %s / %s\n", $1, $2, $3, $4}'
else
  echo "  nvidia-smi not available"
fi
echo ""

echo "--- [PYTHON & LIBS] ---"
python3 -c "
try:
    import vllm, torch
    print(f'  vLLM    : {vllm.__version__}')
    print(f'  PyTorch : {torch.__version__}')
    print(f'  CUDA    : {torch.version.cuda}')
except ImportError as e:
    print(f'  [WARN] {e}')
" 2>/dev/null || echo "  [WARN] Could not import vllm/torch"
echo ""

echo "--- [MODEL FILES] ---"
# Authority
AUTH_MODEL=$(grep '^MODEL=' "$PROJ/config/authority.env" 2>/dev/null | cut -d= -f2 | tr -d '"')
if [ -z "$AUTH_MODEL" ]; then
  echo "  [WARN] authority : MODEL not set in config/authority.env"
elif [ -f "$AUTH_MODEL/config.json" ]; then
  echo "  [OK]   authority : $(basename $AUTH_MODEL)"
else
  echo "  [MISS] authority : MODEL set but not found at $AUTH_MODEL"
fi
# Swarm
SWARM_MODEL=$(grep '^MODEL=' "$PROJ/config/swarm.env" 2>/dev/null | cut -d= -f2 | tr -d '"')
if [ -z "$SWARM_MODEL" ]; then
  echo "  [WARN] swarm     : MODEL not set in config/swarm.env"
elif [ -f "$SWARM_MODEL/config.json" ]; then
  echo "  [OK]   swarm     : $(basename $SWARM_MODEL)"
else
  echo "  [MISS] swarm     : MODEL set but not found at $SWARM_MODEL"
fi
# Embedding
EMB_PATH="${HF_DIR}/hub/e5-small-v2"
[ -f "$EMB_PATH/config.json" ] \
  && echo "  [OK]   embedding : e5-small-v2" \
  || echo "  [MISS] embedding : e5-small-v2 not found at $EMB_PATH"
echo ""

echo "--- [PORTS] ---"
for PORT in 8000 8001 9000; do
  if ss -tlnp 2>/dev/null | grep -q ":${PORT} " || \
     (echo >/dev/tcp/localhost/$PORT) 2>/dev/null; then
    echo "  :${PORT} — IN USE"
  else
    echo "  :${PORT} — free"
  fi
done
echo ""

echo "--- [CLEANUP POTENTIAL] ---"
PIP_CACHE=$(du -sh /root/.cache/pip 2>/dev/null | cut -f1)
[ -n "$PIP_CACHE" ] && echo "  pip cache: $PIP_CACHE (safe to delete)" || echo "  No pip cache."
