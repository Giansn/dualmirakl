# =============================================================================
# dualmirakl — custom RunPod container image
#
# Base: runpod/pytorch (CUDA 12.8, PyTorch 2.9.1, Python 3.11, Ubuntu 22.04)
# Bakes in all Python dependencies and the /post_start.sh hook so the stack
# auto-starts on every pod boot — including after pod recreations.
#
# Models are NOT baked in — they live on /per.volume/huggingface/hub/
# Code  is NOT baked in — cloned to /per.volume/dualmirakl/ on first boot
# =============================================================================

FROM runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204

# ── Environment ───────────────────────────────────────────────────────────────
ENV HF_HOME=/per.volume/huggingface \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── vLLM (separate layer — largest dep, slow to install, benefits from cache) ─
RUN pip install vllm==0.16.0

# ── Project Python dependencies (pinned to current environment) ───────────────
RUN pip install \
    anthropic==0.84.0 \
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
    rich==14.3.3

# ── Startup hook — baked in, survives pod recreations ─────────────────────────
COPY docker-post-start.sh /post_start.sh
RUN chmod +x /post_start.sh
