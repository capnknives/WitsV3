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
from core.guest_access import guest_access_enabled
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


def startup_urls(config, port: int, web_token: str) -> tuple[str, str | None]:
    """Return (localhost_url, lan_url_for_phone).

    When guest access is enabled, the LAN URL must NOT embed the owner token —
    family testers use /join with an invite code instead.
    """
    lan_ip = _lan_ip()
    if web_token:
        localhost = f"http://localhost:{port}/?owner_token={web_token}"
    else:
        localhost = f"http://localhost:{port}/"

    if guest_access_enabled(config):
        return localhost, f"http://{lan_ip}:{port}/join"

    if web_token:
        return localhost, f"http://{lan_ip}:{port}/?owner_token={web_token}"
    return localhost, f"http://{lan_ip}:{port}/"


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
    localhost_url, phone_url = startup_urls(config, port, web_token)
    guests_on = guest_access_enabled(config)
    print()
    print("=" * 72)
    print("  WitsV3 Web UI is starting")
    print(f"    This PC (owner):  {localhost_url}")
    if host == "0.0.0.0" and phone_url:
        if guests_on:
            print(f"    Family testers:   {phone_url}")
            print("                      (invite code required — never share your owner token)")
        else:
            print(f"    Your phone:       {phone_url}")
            print("                      (localhost-only owner_token; prefer typing token on phone)")
        try:
            import qrcode

            qr = qrcode.QRCode(border=1)
            qr.add_data(phone_url)
            label = "family tester" if guests_on else "phone"
            print(f"\n  Scan for {label} (no owner token in this QR):\n")
            qr.print_ascii(invert=True)
        except ImportError:
            pass
    print("=" * 72)
    print()

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="info", access_log=False)
    )
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
