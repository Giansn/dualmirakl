# =============================================================================
# dualmirakl — container image
#
# Base: runpod/pytorch (CUDA 12.8, PyTorch 2.9.1, Python 3.11, Ubuntu 22.04)
#
# Portable: works on RunPod, Docker Compose, K8s.
# Models are NOT baked in — mount or download at runtime.
# =============================================================================

FROM runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204

# ── Environment (overridable at runtime) ──────────────────────────────────────
ENV HF_HOME=${HF_HOME:-/models} \
    DUALMIRAKL_ROOT=/app \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── vLLM (separate layer — largest dep, benefits from cache) ──────────────────
RUN pip install vllm==0.17.1

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
    python-dotenv==1.2.1 \
    loguru==0.7.3 \
    rich==14.3.3 \
    pytest==8.3.5

# ── Copy project code ────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app/

RUN chmod +x /app/*.sh

# ── RunPod compatibility (post_start hook) ────────────────────────────────────
COPY docker-post-start.sh /post_start.sh
RUN chmod +x /post_start.sh

# ── Volumes (models, data output, logs) ──────────────────────────────────────
VOLUME ["/models", "/app/data", "/app/logs"]

# ── Default entrypoint ──────────────────────────────────────────────────────
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
