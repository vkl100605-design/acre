#!/bin/bash
set -e
cd /app

# OpenEnv (FastAPI) — internal; nginx exposes /health, /reset, /step, /state, /api, /docs, …
uvicorn openenv_server:app --host 127.0.0.1 --port 8000 &
export UVICORN_PID=$!

# Full Streamlit demo (same as local: streamlit run app.py) — public root /
streamlit run app.py --server.port=8501 --server.address=127.0.0.1 --server.headless=true &

# Wait for backends
sleep 2

# Single public port 7860 (Hugging Face Spaces)
exec nginx -g "daemon off;"
