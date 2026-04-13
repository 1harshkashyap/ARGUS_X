# ═══════════════════════════════════════════════════════════════════════════════
# ARGUS-X — SOLE BUILD SYSTEM (Dockerfile)
# This project uses ONLY Docker for Railway deployment (see railway.json).
# DO NOT add nixpacks.toml, Procfile, or a second railway.json.
# Those files were removed to eliminate non-deterministic build conflicts.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder (install deps with build tools) ─────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-deploy.txt

# ── Stage 2: Model Fetcher (download at build time, not runtime) ─────────────
# This avoids 2-5 minute cold starts and HuggingFace availability dependency.
# The model is baked into the image so startup is instant.
# If HF_MODEL_REPO is not set, this stage is a no-op (rule-only mode).
FROM python:3.11-slim AS model-fetcher

COPY --from=builder /install /usr/local

ARG HF_MODEL_REPO=""
ENV HF_MODEL_REPO=${HF_MODEL_REPO}

RUN mkdir -p /models && \
    if [ -n "$HF_MODEL_REPO" ]; then \
        python -c "\
from huggingface_hub import hf_hub_download; \
from transformers import AutoTokenizer; \
import os; \
repo = os.environ['HF_MODEL_REPO']; \
print(f'Downloading model from {repo}...'); \
hf_hub_download(repo_id=repo, filename='argus_classifier.onnx', local_dir='/models'); \
tok = AutoTokenizer.from_pretrained(repo); \
tok.save_pretrained('/models/tokenizer'); \
print('Model + tokenizer baked into image.'); \
        "; \
    else \
        echo "No HF_MODEL_REPO set — skipping model download (rule-only mode)"; \
    fi

# ── Stage 3: Runner (lean runtime image) ─────────────────────────────────────
FROM python:3.11-slim AS runner

WORKDIR /app

# Copy only installed packages from builder (no gcc, no pip cache)
COPY --from=builder /install /usr/local

# Copy pre-downloaded models (empty dir if no HF_MODEL_REPO was set)
COPY --from=model-fetcher /models /app/argus/backend/models

# Copy project
COPY . .

# Set working directory to backend
WORKDIR /app/argus/backend

# Point model loader to pre-baked models directory
ENV MODEL_DIR=/app/argus/backend/models
# Railway sets PORT dynamically
ENV PORT=8000

# Start
CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT
