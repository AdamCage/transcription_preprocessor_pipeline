# -- Stage 1: base with system deps + uv ----------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_cache \
    UV_CACHE_DIR=/opt/uv_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# -- Stage 2: install dependencies (cached until pyproject.toml changes) --------
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# -- Stage 3: application code + model pre-download -----------------------------
COPY segmentation_service/ segmentation_service/

ARG SEG_HF_TOKEN=""
RUN uv run python -c "\
from pyannote.audio import Model; \
from pyannote.audio.pipelines import VoiceActivityDetection; \
m = Model.from_pretrained('pyannote/segmentation-3.0', use_auth_token='${SEG_HF_TOKEN}'); \
p = VoiceActivityDetection(segmentation=m); \
p.instantiate({'min_duration_on': 0.0, 'min_duration_off': 0.0}); \
print('Model pre-downloaded successfully') \
"

# -- Runtime --------------------------------------------------------------------
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8001/health || exit 1

CMD ["uv", "run", "uvicorn", "segmentation_service.app:app", \
     "--host", "0.0.0.0", "--port", "8001"]
