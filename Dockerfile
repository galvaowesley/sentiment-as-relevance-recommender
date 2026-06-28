# syntax=docker/dockerfile:1

# ── Stage 1: build do frontend React ───────────────────────────────────────
FROM node:22-slim AS frontend
WORKDIR /app/05_webapp/frontend
COPY 05_webapp/frontend/package.json 05_webapp/frontend/package-lock.json ./
RUN npm ci
COPY 05_webapp/frontend/ ./
RUN npm run build

# ── Stage 2: runtime Python ────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# torch CPU-only (evita wheels CUDA ~2GB); +cpu satisfaz torch==2.6.0
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código + artefatos + data/processed (commitados)
COPY . .
# SPA buildado do stage 1 (sobrescreve a pasta vazia)
COPY --from=frontend /app/05_webapp/frontend/dist ./05_webapp/frontend/dist

CMD ["sh", "-c", "uvicorn app:app --app-dir 05_webapp/backend --host 0.0.0.0 --port ${PORT:-8000}"]
