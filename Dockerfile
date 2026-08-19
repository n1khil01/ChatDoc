# One image, two roles (web / worker) selected by CMD -- see PROJECT_PLAN.md §7 Phase 4.
# Render free has no persistent disk, so model weights are baked into the image at build
# time (FASTEMBED_CACHE_PATH inside the repo dir) rather than re-downloaded on every cold
# start on top of the ~60s spin-down wake.
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/app/.model_cache \
    HF_HOME=/app/.model_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY api ./api
COPY ingest ./ingest
COPY db ./db
COPY eval ./eval

# Bake model weights into the image: import the embedding + reranker modules and run one
# throwaway call each, so `_model()`'s lru_cache-wrapped constructor downloads and caches
# both ONNX checkpoints under FASTEMBED_CACHE_PATH now, not on the first real request.
RUN uv run python -c "\
from ingest.embeddings import embed_texts; \
from ingest.reranker import rerank; \
embed_texts(['warm the cache']); \
rerank('warm', ['the cache'])"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
