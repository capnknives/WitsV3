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
import time
from contextlib import suppress
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import uvicorn

from core.config import load_config
from core.logging_config import apply_logging_level
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


_PROJECT_ROOT = Path(__file__).resolve().parent

# Source that, when changed, should trigger a web-UI restart.
_RELOAD_WATCH_EXTENSIONS = (".py", ".yaml", ".yml", ".html", ".js", ".css")

# Volatile / generated directories whose changes must NOT trigger a restart
# (logs and memory churn constantly; a restart loop would be the result).
_RELOAD_IGNORED_DIR_PARTS = (
    "/.git/",
    "/__pycache__/",
    "/.venv/",
    "/.pytest_cache/",
    "/var/",
    "/logs/",
    "/data/",
    "/memory/",
    "/exports/",
    "/sessions/",
    "/documents/",
    "/user_files/",
    "/workspace/",
    "/node_modules/",
    "/.mypy_cache/",
    "/.ruff_cache/",
)


def _reload_watch_filter(_change, path: str) -> bool:
    """watchfiles predicate: True only for editable project source files."""
    normalized = path.replace("\\", "/").lower()
    if any(part in normalized for part in _RELOAD_IGNORED_DIR_PARTS):
        return False
    return normalized.endswith(_RELOAD_WATCH_EXTENSIONS)


async def _wait_for_source_change() -> bool:
    """Block until a watched source file changes; return True when one does.

    Returns False if file watching is unavailable (watchfiles missing), so the
    caller can fall back to serving without auto-reload.
    """
    try:
        from watchfiles import awatch
    except ImportError:
        logger.warning(
            "watchfiles is not installed; auto-reload disabled "
            "(install uvicorn[standard] to enable it)."
        )
        return False

    async for changes in awatch(_PROJECT_ROOT, watch_filter=_reload_watch_filter):
        names = sorted({Path(p).name for _, p in changes})
        preview = ", ".join(names[:5]) + (" …" if len(names) > 5 else "")
        logger.info("Source change detected (%s); restarting web UI.", preview)
        print(f"\n[auto-reload] change detected: {preview} — restarting web UI...\n")
        return True
    return False


async def _serve_with_auto_reload(server: "uvicorn.Server") -> bool:
    """Run the server alongside a file watcher.

    Returns True if a source change requested a restart, False if the server
    stopped on its own (e.g. Ctrl-C).
    """
    serve_task = asyncio.create_task(server.serve())
    watch_task = asyncio.create_task(_wait_for_source_change())

    done, _pending = await asyncio.wait(
        {serve_task, watch_task}, return_when=asyncio.FIRST_COMPLETED
    )

    restart = False
    if watch_task in done and watch_task.result():
        # A change fired first — ask uvicorn to shut down gracefully, then wait.
        restart = True
        server.should_exit = True
        await serve_task
    else:
        # Server exited on its own; stop watching.
        watch_task.cancel()
        with suppress(asyncio.CancelledError):
            await watch_task
        if not serve_task.done():
            await serve_task

    return restart


def _restart_process() -> None:
    """Re-exec this process with the same interpreter and arguments.

    The server socket has already been released (graceful shutdown awaited
    before this call), so the fresh process can rebind the port cleanly.
    """
    print("[auto-reload] restarting now...\n")
    sys.stdout.flush()
    time.sleep(0.5)  # small margin for the OS to release the listening socket
    os.execv(sys.executable, [sys.executable, *sys.argv])


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
    apply_logging_level(config.logging_level)

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
            print(
                "                      (localhost-only owner_token; prefer typing token on phone)"
            )
        try:
            import qrcode

            qr = qrcode.QRCode(border=1)
            qr.add_data(phone_url)
            label = "family tester" if guests_on else "phone"
            print(f"\n  Scan for {label} (no owner token in this QR):\n")
            qr.print_ascii(invert=True)
        except ImportError:
            pass
    auto_reload = config.web_ui.auto_reload
    if auto_reload:
        print("  Auto-reload:      ON (editing project source restarts the web UI)")
    print("=" * 72)
    print()

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="info", access_log=False)
    )
    restart_requested = False
    try:
        if auto_reload:
            restart_requested = await _serve_with_auto_reload(server)
        else:
            await server.serve()
    finally:
        await system.shutdown()

    if restart_requested:
        _restart_process()  # replaces this process; does not return
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
