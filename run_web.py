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


def _port_in_use(host: str, port: int) -> bool:
    """Check whether something is already listening on the target port."""
    probe_host = "127.0.0.1" if host == "0.0.0.0" else host
    try:
        with socket.create_connection((probe_host, port), timeout=1.0):
            return True
    except OSError:
        return False


async def main() -> int:
    config = load_config()

    if not config.web_ui.enabled:
        print("web_ui.enabled is false in config.yaml - nothing to do.")
        return 1

    if _port_in_use(config.web_ui.host, config.web_ui.port):
        port = config.web_ui.port
        print()
        print(f"  WitsV3 Web UI looks like it's ALREADY RUNNING on port {port}.")
        print(f"  Just open:  http://localhost:{port}")
        print()
        print("  If you actually want to restart it, stop the other instance first:")
        print(f"    Get-NetTCPConnection -LocalPort {port} -State Listen | ")
        print("        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }")
        print()
        return 1

    system = WitsV3System(config)
    await system.initialize()

    app = create_app(system)

    host, port = config.web_ui.host, config.web_ui.port
    web_token = os.getenv("WITSV3_WEB_TOKEN", "")
    token_suffix = f"/?token={web_token}" if web_token else "/"
    print()
    print("=" * 72)
    print("  WitsV3 Web UI is starting")
    print(f"    This PC:    http://localhost:{port}{token_suffix}")
    if host == "0.0.0.0":
        print(f"    Your phone: http://{_lan_ip()}:{port}{token_suffix}")
        print("                (same Wi-Fi; the link logs you in automatically)")
    print("=" * 72)
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
