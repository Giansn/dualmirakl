#!/bin/bash
echo "--- [SYSTEM AUDIT] ---"
date
echo "Disk Usage /per.volume: $(df -h /per.volume | tail -n 1 | awk '{print $3 " used of " $2}')"
echo "GPU Status: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | tr '\n' ' | ')"

echo -e "\n--- [PYTHON & LIBS] ---"
source /per.volume/venv/bin/activate
which python3
python3 -c "import vllm, torch; print(f'vLLM: {vllm.__version__} | PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}')"

echo -e "\n--- [MODEL FILES] ---"
[ -f /per.volume/huggingface/hub/glm-5/config.json ] && echo "✅ GLM-5: OK" || echo "❌ GLM-5: MISSING"
[ -f /per.volume/huggingface/hub/deepseek-v3.2-special/config.json ] && echo "✅ DeepSeek-V3.2-Special: OK" || echo "❌ DeepSeek-V3.2-Special: MISSING"
[ -f /per.volume/huggingface/hub/e5-small-v2/config.json ] && echo "✅ e5-small-v2: OK" || echo "❌ e5-small-v2: MISSING"

echo -e "\n--- [CLEANUP POTENTIAL] ---"
du -sh /per.volume/pip_cache /per.volume/pip_tmp 2>/dev/null || echo "Cache already empty."
