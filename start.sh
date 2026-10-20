#!/usr/bin/env bash
set -e

echo "Starting ArchGraph FastAPI Intelligence Engine on 127.0.0.1:8000..."
uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1 &
BACKEND_PID=$!

# Wait up to 30s for FastAPI health check
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null; then
        echo "FastAPI Backend is live!"
        break
    fi
    sleep 1
done

echo "Launching Decoupled Streamlit Frontend on 0.0.0.0:7860..."
exec streamlit run frontend/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
