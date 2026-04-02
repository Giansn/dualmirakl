# =============================================================================
# dualmirakl — container image (Blackwell RTX PRO 4500 / sm_120)
#
# Base: runpod/pytorch (CUDA 12.9, PyTorch 2.9.1, Python 3.12, Ubuntu 22.04)
# vLLM 0.18.2 (Blackwell sm_120 + flashinfer MOE FP4)
#
# Portable: works on RunPod, Docker Compose, K8s.
# Models are NOT baked in — mount or download at runtime.
# =============================================================================

FROM runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204

# ── Environment (overridable at runtime) ──────────────────────────────────────
ENV HF_HOME=${HF_HOME:-/models} \
    DUALMIRAKL_ROOT=/app \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LIBRARY_PATH=/usr/local/cuda/lib64

# ── System packages + CUDA 12.9 toolkit (nvcc for flashinfer JIT) ────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        cuda-nvcc-12-9 \
        cuda-cudart-dev-12-9 \
        libcurand-dev-12-9 \
    && rm -rf /var/lib/apt/lists/*

# ── vLLM (separate layer — largest dep, benefits from cache) ──────────────────
RUN pip install vllm==0.18.2

# ── NVIDIA headers for flashinfer JIT (Blackwell sm_120) ─────────────────────
RUN for d in /usr/local/lib/python3.*/dist-packages/nvidia/*/include/; do \
        cp -n "$d"*.h /usr/local/cuda/include/ 2>/dev/null || true; \
    done

# ── Library symlinks for flashinfer linker (-lcuda -lcudart -lnvrtc) ─────────
RUN CUDA_LIB=/usr/local/cuda/lib64 && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so $CUDA_LIB/libcuda.so && \
    ln -sf $(find /usr/local/lib -name "libcudart.so.12" -path "*/nvidia/*" | head -1) \
        $CUDA_LIB/libcudart.so && \
    ln -sf $(find /usr/local/lib -name "libnvrtc.so.12" -path "*/nvidia/*" | head -1) \
        $CUDA_LIB/libnvrtc.so

# ── Project Python dependencies ──────────────────────────────────────────────
RUN pip install \
    openai==2.24.0 \
    fastapi==0.134.0 \
    "sentence-transformers==5.2.3" \
    "uvicorn[standard]==0.41.0" \
    "httpx[http2]==0.27.2" \
    h2==4.3.0 \
    pydantic==2.12.5 \
    pydantic-settings==2.13.1 \
    simpy==4.1.1 \
    jax==0.9.0.1 \
    jaxlib==0.9.0.1 \
    numpyro==0.20.0 \
    numpy==2.2.6 \
    scipy==1.17.1 \
    duckdb==1.2.0 \
    pynvml==12.0.0 \
    neuralprophet==0.8.0 \
    python-dotenv==1.2.1 \
    loguru==0.7.3 \
    rich==14.3.3 \
    pytest==8.3.5

# ── Copy project code ────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app/

RUN chmod +x /app/scripts/*.sh /app/docker/*.sh /app/*.sh 2>/dev/null || true

# ── RunPod compatibility (post_start hook) ────────────────────────────────────
COPY docker/post-start.sh /post_start.sh
RUN chmod +x /post_start.sh

# ── Volumes (models, data output, logs, flashinfer JIT cache) ────────────────
VOLUME ["/models", "/app/data", "/app/logs", "/root/.cache"]

# ── Ports ────────────────────────────────────────────────────────────────────
EXPOSE 8000 8001 9000

# ── Default entrypoint ──────────────────────────────────────────────────────
# Default: container stays alive with SSH. Run 'bash start_all.sh' to boot vLLM.
# Set ENTRYPOINT_MODE=all to auto-start everything.
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
