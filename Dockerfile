# ═══════════════════════════════════════════════════════════════════════════════
# ARGUS-X — SOLE BUILD SYSTEM (Dockerfile)
# This project uses ONLY Docker for Railway deployment (see railway.json).
# DO NOT add nixpacks.toml, Procfile, or a second railway.json.
# Those files were removed to eliminate non-deterministic build conflicts.
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

# Copy and install slim requirements (no torch/sentence-transformers = ~1.5GB image)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy project
COPY . .

# Set working directory to backend
WORKDIR /app/argus/backend

# Railway sets PORT dynamically
ENV PORT=8000

# Start
CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT
