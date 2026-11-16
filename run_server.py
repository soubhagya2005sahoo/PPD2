#!/usr/bin/env python3
"""Run Aerial-Ground Person Re-ID Application Server.

Starts the FastAPI backend serving both the REST API and the React SPA.
Automatically handles port conflicts if port 8000 is already in use.
"""

import os
import sys
import socket
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a network port is currently occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def find_free_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    """Find the first available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return start_port


def main():
    parser = argparse.ArgumentParser(description="Aerial-Ground Person Re-ID Web Application")
    parser.add_argument("--port", type=int, default=8000, help="Target port to bind server (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable code auto-reloading")
    args = parser.parse_args()

    port = args.port
    if is_port_in_use(port, host=args.host):
        alt_port = find_free_port(port + 1)
        print("=" * 70)
        print(f"[NOTE] Port {port} is currently occupied by another process.")
        print(f"       Automatically switching to free port: {alt_port}")
        print("=" * 70)
        port = alt_port

    url = f"http://localhost:{port}"
    print("=" * 70)
    print(" Aerial-Ground Person Re-Identification Web Application")
    print(" Authors: Soubhagya Ranjan Sahoo (23110238) & Aryan Pradhan (23110197)")
    print(" Guide  : Dr. Rajalaxmi Padhy")
    print("=" * 70)
    print(f"\n🚀 Web Application URL: {url}")
    print(f"📖 Interactive API Docs: {url}/docs\n")
    print("Press CTRL+C to stop the server.\n")

    import uvicorn
    uvicorn.run("backend.app.main:app", host=args.host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()

