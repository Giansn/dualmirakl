#!/bin/bash
echo "--- [SYSTEM AUDIT] ---"
date
echo ""

echo "--- [DISK] ---"
echo "  /per.volume : $(df -h /per.volume | tail -n 1 | awk '{print $3 " used of " $2 " (" $5 ")"}')"
echo "  hub         : $(du -sh /per.volume/huggingface/hub 2>/dev/null | cut -f1)"
echo ""

echo "--- [GPU] ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | \
  awk -F', ' '{printf "  GPU %s — %s | %s / %s\n", $1, $2, $3, $4}'
echo ""

echo "--- [PYTHON & LIBS] ---"
python3 -c "
import vllm, torch
print(f'  vLLM    : {vllm.__version__}')
print(f'  PyTorch : {torch.__version__}')
print(f'  CUDA    : {torch.version.cuda}')
" 2>/dev/null || echo "  [WARN] Could not import vllm/torch"
echo ""

echo "--- [MODEL FILES] ---"
# Authority slot
AUTH_MODEL=$(grep '^MODEL=' /per.volume/dualmirakl/models/authority.env 2>/dev/null | cut -d= -f2 | tr -d '"')
if [ -z "$AUTH_MODEL" ]; then
  echo "  ⚠️  authority : MODEL not set in models/authority.env"
elif [ -f "$AUTH_MODEL/config.json" ]; then
  echo "  ✅ authority : $(basename $AUTH_MODEL)"
else
  echo "  ❌ authority : MODEL set but not found at $AUTH_MODEL"
fi
# Swarm slot
[ -f /per.volume/huggingface/hub/nemotron-nano-30b/config.json ] \
  && echo "  ✅ swarm     : nemotron-nano-30b" \
  || echo "  ❌ swarm     : nemotron-nano-30b MISSING"
# Embedding
[ -f /per.volume/huggingface/hub/e5-small-v2/config.json ] \
  && echo "  ✅ embedding : e5-small-v2" \
  || echo "  ❌ embedding : e5-small-v2 MISSING"
echo ""

echo "--- [PORTS] ---"
for PORT in 8000 8001 9000; do
  if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "  :${PORT} — IN USE"
  else
    echo "  :${PORT} — free"
  fi
done
echo ""

echo "--- [CLEANUP POTENTIAL] ---"
du -sh /root/.cache/pip 2>/dev/null && echo "  (pip cache — safe to delete)" || echo "  No pip cache."
