"""Aerial-Ground Person Re-Identification Web Application.

FastAPI Application Server integrating the trained PyTorch Re-ID model,
40,473 AG-ReID.v2 gallery database, and static React frontend.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.app.core.config import settings, ROOT_DIR
from backend.app.api import api_router
from backend.app.services.model_service import model_service
from backend.app.services.gallery_service import gallery_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up model and gallery cache on server startup."""
    print("=" * 70)
    print("Initializing Aerial-Ground Re-ID Web Application Engine...")
    print(f"Project Directory: {ROOT_DIR}")
    print("=" * 70)

    try:
        model_service.load_model()
    except Exception as e:
        print(f"[Warning] Could not preload model checkpoint: {e}")

    try:
        gallery_service.load_gallery()
    except Exception as e:
        print(f"[Warning] Could not preload gallery database: {e}")

    print("✓ Re-ID Web Application startup complete.")
    yield
    print("Re-ID Web Application shutting down...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Research Web Application for View-Aware Multi-Scale Image Enhancement and Feature Fusion for Aerial–Ground Person Re-Identification",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React frontend & local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


# Mount Static Frontend if built
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
if FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "index.html").is_file():
    # Mount assets directory
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # SPA catch-all fallback
    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        # Don't intercept API routes
        if full_path.startswith("api"):
            return HTMLResponse(status_code=404, content="API route not found")
        
        safe_rel = full_path.lstrip("/")
        direct_file = (FRONTEND_DIST / safe_rel).resolve()
        try:
            direct_file.relative_to(FRONTEND_DIST.resolve())
            if direct_file.is_file():
                return FileResponse(direct_file)
        except ValueError:
            return HTMLResponse(status_code=403, content="Access denied")

        # Only serve index.html for extensionless SPA routes or html requests
        if "." in full_path.split("/")[-1]:
            return HTMLResponse(status_code=404, content="File not found")

        # Fallback to index.html for React Router
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    async def root():
        return {
            "status": "online",
            "message": "Aerial-Ground Person Re-ID API is running.",
            "docs_url": "/docs",
            "api_prefix": "/api",
        }


def find_available_port(default_port: int = 8000, max_attempts: int = 20) -> int:
    """Find a free port starting from default_port."""
    import socket
    for port in range(default_port, default_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return default_port


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Aerial-Ground Re-ID Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # If requested port is busy, find next free port
    chosen_port = args.port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((args.host, chosen_port))
        except OSError:
            new_port = find_available_port(chosen_port + 1)
            print(f"[Notice] Port {chosen_port} is busy. Automatically switching to free port: {new_port}")
            chosen_port = new_port

    print(f"\n🚀 Server starting on: http://localhost:{chosen_port}\n")
    uvicorn.run("backend.app.main:app", host=args.host, port=chosen_port, reload=args.reload)
