# run_web.py
"""
Web UI entry point for WitsV3.

Starts the full WitsV3 system and serves the browser interface (desktop +
phone). Run with:  python run_web.py
"""

import asyncio
import logging
import os
import socket
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import uvicorn

from core.config import load_config
from run import WitsV3System
from web.server import create_app

logger = logging.getLogger("WitsV3.WebUI")


def _lan_ip() -> str:
    """Best-effort LAN IP for the 'open this on your phone' hint."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


async def main() -> int:
    config = load_config()

    if not config.web_ui.enabled:
        print("web_ui.enabled is false in config.yaml - nothing to do.")
        return 1

    system = WitsV3System(config)
    await system.initialize()

    app = create_app(system)

    host, port = config.web_ui.host, config.web_ui.port
    print()
    print("=" * 56)
    print("  WitsV3 Web UI is starting")
    print(f"    This PC:    http://localhost:{port}")
    if host == "0.0.0.0":
        print(f"    Your phone: http://{_lan_ip()}:{port}   (same Wi-Fi)")
    if os.getenv("WITSV3_WEB_TOKEN"):
        print("    Access token: set (see WITSV3_WEB_TOKEN in .env)")
    print("=" * 56)
    print()

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    try:
        await server.serve()
    finally:
        await system.shutdown()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
