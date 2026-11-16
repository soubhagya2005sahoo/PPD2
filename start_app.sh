#!/usr/bin/env bash
# ==============================================================================
# Aerial-Ground Person Re-Identification Launcher Script
# Authors: Soubhagya Ranjan Sahoo (23110238) & Aryan Pradhan (23110197)
# Guide  : Dr. Rajalaxmi Padhy
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 1. Activate virtual environment if present
if [ -d ".venv" ]; then
    source ".venv/bin/activate"
elif [ -d "venv" ]; then
    source "venv/bin/activate"
fi

# 2. Check if port 8000 is already in use by a stale python process
if lsof -i :8000 >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is currently occupied."
    STALE_PID=$(lsof -ti :8000 | head -n 1)
    echo "   Attempting to free port 8000 (PID: $STALE_PID)..."
    kill -15 "$STALE_PID" 2>/dev/null || true
    sleep 0.5
fi

# 3. Launch server with automatic free-port detection fallback
python run_server.py "$@"

