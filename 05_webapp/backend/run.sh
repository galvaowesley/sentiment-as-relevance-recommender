#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHONPATH="$REPO_ROOT/04_recommender" \
  uvicorn app:app \
    --app-dir "$SCRIPT_DIR" \
    --port 8000 \
    --reload
