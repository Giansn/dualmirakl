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
[ -f /per.volume/huggingface/hub/command-r7b-awq/config.json ] && echo "✅ Command-R: OK" || echo "❌ Command-R: MISSING"
[ -f /per.volume/huggingface/hub/qwen2.5-7b-awq/config.json ] && echo "✅ Qwen: OK" || echo "❌ Qwen: MISSING"
[ -f /per.volume/huggingface/hub/gte-small/config.json ] && echo "✅ GTE-small: OK" || echo "❌ GTE-small: MISSING"

echo -e "\n--- [CLEANUP POTENTIAL] ---"
du -sh /per.volume/pip_cache /per.volume/pip_tmp 2>/dev/null || echo "Cache already empty."
