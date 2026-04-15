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
ARG HF_TOKEN=""
ENV HF_MODEL_REPO=${HF_MODEL_REPO}
ENV HF_TOKEN=${HF_TOKEN}

RUN mkdir -p /models && \
    if [ -n "$HF_MODEL_REPO" ]; then \
        python -c "\
from huggingface_hub import hf_hub_download; \
from transformers import AutoTokenizer; \
import os; \
repo = os.environ['HF_MODEL_REPO']; \
token = os.environ.get('HF_TOKEN') or None; \
print(f'Downloading model from {repo}...'); \
hf_hub_download(repo_id=repo, filename='argus_classifier.onnx', local_dir='/models', token=token); \
tok = AutoTokenizer.from_pretrained(repo, token=token); \
tok.save_pretrained('/models/tokenizer'); \
print('Model + tokenizer baked into image.'); \
        "; \
    else \
        echo "No HF_MODEL_REPO set — skipping model download (rule-only mode)"; \
    fi

# ── Stage 3: Frontend Builder (compile React/Vite app) ──────────────────────
# Builds the frontend to static /dist files. The Python backend serves these.
FROM node:20-slim AS frontend-builder

WORKDIR /frontend

# Install dependencies first (layer caching)
COPY argus/frontend/package.json argus/frontend/package-lock.json ./
RUN npm ci --ignore-scripts

# Copy source and build
COPY argus/frontend/ ./

# Set production API URL (backend on same origin in Docker)
ENV VITE_API_URL=""
ENV VITE_WS_URL=""

RUN npm run build

# ── Stage 4: Runner (lean runtime image, non-root) ───────────────────────────
FROM python:3.11-slim AS runner

# SECURITY: Create non-root user before copying anything
RUN groupadd -r argus && useradd -r -g argus -d /app -s /sbin/nologin argus

WORKDIR /app

# Copy only installed packages from builder (no gcc, no pip cache)
COPY --from=builder /install /usr/local

# Copy pre-downloaded models (empty dir if no HF_MODEL_REPO was set)
COPY --from=model-fetcher /models /app/argus/backend/models

# Copy project
COPY --chown=argus:argus . .

# Copy built frontend dist into the expected location
COPY --from=frontend-builder --chown=argus:argus /frontend/dist /app/argus/frontend/dist

# Set working directory to backend
WORKDIR /app/argus/backend

# Point model loader to pre-baked models directory
ENV MODEL_DIR=/app/argus/backend/models
# Railway sets PORT dynamically
ENV PORT=8000

# SECURITY: Run as non-root user
USER argus

# Start
CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT

