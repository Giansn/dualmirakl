# DOCKER.md — Build & Runtime Requirements for Blackwell GPUs

This file documents all system-level dependencies required to run dualmirakl on **NVIDIA Blackwell** (RTX PRO 4500, sm_120) GPUs with vLLM + flashinfer.

## Base Image

```
runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204
```

- Ubuntu 22.04, Python 3.12, PyTorch 2.9.1, CUDA runtime 12.9

## Critical: CUDA 12.9 Toolkit (nvcc)

Blackwell GPUs use `sm_120a` / `sm_120f` compute capability. flashinfer JIT-compiles CUDA kernels at first startup and requires a system `nvcc` that supports these architectures.

| CUDA Version | Blackwell Support |
|---|---|
| 12.4 | `nvcc fatal: Unsupported gpu architecture 'compute_120a'` |
| 12.8 | `SM 12.x requires CUDA >= 12.9` |
| **12.9+** | Works |

### Install CUDA 12.9 nvcc

```bash
# Add NVIDIA repo (if not present)
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get install -y --no-install-recommends \
    cuda-nvcc-12-9 \
    cuda-cudart-dev-12-9 \
    libcurand-dev-12-9
```

### Required NVIDIA Headers

flashinfer's JIT compilation needs headers that the minimal nvcc package does not include. Copy them from pip-installed NVIDIA packages:

```bash
# Copy ALL NVIDIA headers into the CUDA include directory
for d in /usr/local/lib/python3.11/dist-packages/nvidia/*/include/; do
    cp -n "$d"*.h /usr/local/cuda/include/ 2>/dev/null
done
```

Key headers needed (will fail one-by-one if missing):
- `curand_kernel.h` — from `libcurand-dev-12-9` or `nvidia-cuda-curand`
- `cublasLt.h` — from `nvidia-cublas` pip package
- `nvrtc.h` — from `nvidia-cuda-nvrtc` pip package

### Required Link Libraries

flashinfer MOE kernels link against three libraries (`-lcuda -lcudart -lnvrtc`). These must be findable in the CUDA lib path:

```bash
# Symlink from pip-installed NVIDIA packages into CUDA lib directory
CUDA_LIB=/usr/local/cuda/lib64

# libcuda.so (driver stub)
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so $CUDA_LIB/libcuda.so

# libcudart.so
ln -sf /usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12 \
    $CUDA_LIB/libcudart.so

# libnvrtc.so
ln -sf /usr/local/lib/python3.11/dist-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 \
    $CUDA_LIB/libnvrtc.so
```

Or set `LIBRARY_PATH` at runtime:

```bash
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
```

## Disk Space

The flashinfer JIT compilation generates large intermediate files. The root filesystem needs sufficient space:

| Component | Size |
|---|---|
| CUDA 12.9 toolkit (nvcc + headers) | ~3 GB |
| flashinfer JIT cache (`~/.cache/flashinfer/`) | ~2 GB |
| vLLM torch.compile cache (`~/.cache/vllm/`) | ~500 MB |
| Model weights (Nemotron Nano 30B NVFP4) | ~16 GB |

**Minimum free disk**: 6 GB on root filesystem during JIT compilation. If running on RunPod with a small root volume, either:
- Install CUDA toolkit on the network volume (`/workspace/cuda-12.9`)
- Remove old CUDA versions: `rm -rf /usr/local/cuda-12.4 /usr/local/cuda-12.8`

### Persistent CUDA on Network Volume (RunPod)

```bash
# Install to /workspace (persistent across pod restarts)
cp -a /usr/local/cuda-12.9 /workspace/cuda-12.9
ln -sfn /workspace/cuda-12.9 /usr/local/cuda

# Remove root copy to free space
rm -rf /usr/local/cuda-12.9
```

## VRAM Budget (32 GB Blackwell)

Nemotron Nano 30B NVFP4 with flashinfer MOE kernels on Blackwell:

| Component | VRAM |
|---|---|
| Model weights (NVFP4) | ~18.6 GiB |
| PyTorch reserved (fragmentation) | ~6.6 GiB |
| KV cache (fp8) | remaining (~5-6 GiB) |

### Working vLLM flags for 32 GB

```bash
python -m vllm.entrypoints.openai.api_server \
    --model $HF_HOME/hub/nemotron-nano-30b \
    --served-model-name swarm \
    --gpu-memory-utilization 0.90 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --kv-cache-dtype fp8 --dtype auto \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    --no-enable-log-requests
```

Key constraints:
- `max-model-len=8192` causes OOM — use `2048` or `4096`
- `max-num-seqs=12` may OOM — use `8`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps with fragmentation

## flashinfer JIT Compilation

First startup after a fresh install takes **5-10 minutes** while flashinfer compiles CUDA kernels for Blackwell. Progress is silent in the log (stuck after `Cache the graph of compile range`). Check with:

```bash
ps aux | grep -c nvcc    # active compiler processes
df -h /                  # disk space during compilation
```

Compiled kernels are cached in `~/.cache/flashinfer/`. Subsequent startups skip JIT.

### Cache Locations

```
~/.cache/flashinfer/0.6.7/120f/cached_ops/    # flashinfer kernels (Blackwell sm_120f)
~/.cache/vllm/torch_compile_cache/             # torch.compile graphs
```

To force recompilation (e.g., after CUDA upgrade):

```bash
rm -rf ~/.cache/flashinfer/
```

## Dockerfile Additions

To add Blackwell JIT support to the existing Dockerfile:

```dockerfile
# ── CUDA 12.9 nvcc for Blackwell JIT compilation ─────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-12-9 \
        cuda-cudart-dev-12-9 \
        libcurand-dev-12-9 \
    && rm -rf /var/lib/apt/lists/*

# Copy NVIDIA headers from pip packages into CUDA include path
RUN for d in /usr/local/lib/python3.*/dist-packages/nvidia/*/include/; do \
        cp -n "$d"*.h /usr/local/cuda/include/ 2>/dev/null || true; \
    done

# Symlink NVIDIA libraries for flashinfer linker
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so /usr/local/cuda/lib64/libcuda.so && \
    ln -sf $(find /usr/local/lib -name "libcudart.so.12" -path "*/nvidia/*" | head -1) \
        /usr/local/cuda/lib64/libcudart.so && \
    ln -sf $(find /usr/local/lib -name "libnvrtc.so.12" -path "*/nvidia/*" | head -1) \
        /usr/local/cuda/lib64/libnvrtc.so
```

## Environment Variables

```bash
# Required
HF_HOME=/workspace/huggingface          # model storage
CUDA_VISIBLE_DEVICES=1                  # GPU assignment for swarm

# Recommended
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_USE_FLASHINFER_MOE_FP4=1          # set in swarm.env
VLLM_FLASHINFER_MOE_BACKEND=throughput  # set in swarm.env
LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

# Optional
FLASHINFER_CACHE_DIR=~/.cache/flashinfer  # default location
```

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `nvcc fatal: Unsupported gpu architecture 'compute_120a'` | CUDA < 12.9 | Install `cuda-nvcc-12-9` |
| `SM 12.x requires CUDA >= 12.9` | CUDA 12.8 | Upgrade to 12.9 |
| `fatal error: curand_kernel.h: No such file` | Missing cuRAND headers | `apt install libcurand-dev-12-9` or copy from pip |
| `fatal error: cublasLt.h: No such file` | Missing cuBLAS headers | Copy from `nvidia/cublas` pip package |
| `fatal error: nvrtc.h: No such file` | Missing NVRTC headers | Copy from `nvidia/cuda_nvrtc` pip package |
| `cannot find -lnvrtc` | Missing NVRTC library | Symlink `libnvrtc.so` into CUDA lib64 |
| `No space left on device` during JIT | Root disk full | Remove old CUDA versions or use /workspace |
| `CUDA out of memory` at KV cache alloc | VRAM exhausted | Reduce `max-model-len` to 2048, `max-num-seqs` to 8 |
| Log stuck at `Cache the graph of compile range` | flashinfer JIT compiling | Wait 5-10 min, check `ps aux | grep nvcc` |
