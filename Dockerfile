# ═══════════════════════════════════════════════════════════════════════════════
# ARGUS-X — SOLE BUILD SYSTEM (Dockerfile)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Frontend Builder (compile React/Vite app) ──────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
# Install dependencies first (layer caching)
COPY argus/frontend/package.json argus/frontend/package-lock.json ./
RUN npm ci --ignore-scripts
# Copy source and build
COPY argus/frontend/ ./
ENV VITE_API_URL=""
ENV VITE_WS_URL=""
RUN npm run build

# ── Stage 2: Runner ───────────────────────────
FROM python:3.11-slim AS runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

RUN pip install huggingface-hub && \
    python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='cross-encoder/ms-marco-MiniLM-L-6-v2', \
    local_dir='/app/models/onnx')"

RUN mkdir -p /app && useradd -m appuser && chown -R appuser /app
USER appuser

WORKDIR /app

# Copy project
COPY --chown=appuser:appuser . .

# Copy built frontend dist into the expected location
COPY --from=frontend-builder --chown=appuser:appuser /frontend/dist /app/argus/frontend/dist

# Set working directory to backend
WORKDIR /app/argus/backend

# Point model loader to pre-baked models directory
ENV MODEL_DIR=/app/models/onnx
# Railway sets PORT dynamically
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/health')"

# Start
CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT
