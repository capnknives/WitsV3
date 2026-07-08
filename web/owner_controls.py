"""Owner-only process controls for the Web UI.

Richard (creator) can type /shutdown or /restart in chat, or call the
matching API. Both paths require a matching WITSV3_WEB_TOKEN — if the
token is unset in .env, these controls refuse rather than trust an
open network UI.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("WitsV3.OwnerControls")

# Exact chat-line commands (case-insensitive). Natural-language requests
# still go through the agent and are intentionally NOT killed this way.
SHUTDOWN_COMMANDS = frozenset({"/shutdown", "/stop", "/kill", "/quit"})
RESTART_COMMANDS = frozenset({"/restart"})
ALL_OWNER_COMMANDS = SHUTDOWN_COMMANDS | RESTART_COMMANDS


def parse_owner_command(message: str) -> str | None:
    """Return 'shutdown' or 'restart' if *message* is an owner slash command."""
    text = (message or "").strip().lower()
    if not text:
        return None
    # Allow optional trailing punctuation: /shutdown!  /restart.
    core = text.rstrip(".!")
    if core in SHUTDOWN_COMMANDS:
        return "shutdown"
    if core in RESTART_COMMANDS:
        return "restart"
    return None


def extract_bearer_token(request: Request) -> str:
    """Read the bearer token from the Authorization header only.

    Query-string tokens are intentionally not accepted — they leak via URLs,
    QR codes, browser history, and Referer headers.
    """
    header = request.headers.get("authorization", "")
    if header.startswith("Bearer "):
        return header.removeprefix("Bearer ").strip()
    return ""


def configured_web_token() -> str:
    return os.getenv("WITSV3_WEB_TOKEN", "").strip()


def owner_auth_failure(request: Request) -> JSONResponse | None:
    """Return a 401/403 response if the caller is not the configured owner."""
    detail = owner_auth_detail(request)
    if detail is None:
        return None
    status = 403 if "WITSV3_WEB_TOKEN" in detail else 401
    return JSONResponse({"detail": detail}, status_code=status)


def owner_auth_detail(request: Request) -> str | None:
    """Return an error detail string if unauthorized, else None."""
    expected = configured_web_token()
    if not expected:
        return (
            "Owner controls require WITSV3_WEB_TOKEN in .env so WITS "
            "can verify the requester is you."
        )
    presented = extract_bearer_token(request)
    if presented != expected:
        return "unauthorized — your web token does not match WITSV3_WEB_TOKEN"
    return None


def _relaunch_then_exit() -> None:
    """Relaunch the same entry script, then exit this process."""
    script_path = sys.argv[0]
    cmd = [sys.executable, script_path]
    cmd += [a for a in sys.argv[1:] if a != "--restart"]
    cmd.append("--restart")
    logger.warning("Owner-triggered restart: %s", " ".join(cmd))
    subprocess.Popen(cmd, cwd=os.getcwd())
    os._exit(0)


def _hard_exit() -> None:
    logger.warning("Owner-triggered shutdown — exiting web process")
    os._exit(0)


def schedule_owner_action(action: str, delay_seconds: float = 1.0) -> dict[str, Any]:
    """Schedule a process shutdown (or restart) after the current response finishes."""
    if action not in ("shutdown", "restart"):
        raise ValueError(f"unknown owner action: {action}")

    delay = max(0.3, min(float(delay_seconds), 10.0))
    loop = asyncio.get_running_loop()
    if action == "restart":
        loop.call_later(delay, _relaunch_then_exit)
        message = f"Restarting WITS in {delay:.0f}s. The web UI will come back up on the same port."
    else:
        loop.call_later(delay, _hard_exit)
        message = (
            f"Shutting down WITS in {delay:.0f}s. "
            "Port 8000 will be free afterward — start again with start_web_ui.bat when ready."
        )
    return {"success": True, "action": action, "delay_seconds": delay, "message": message}


def owner_command_reply(action: str) -> str:
    if action == "restart":
        return "Owner command accepted. Restarting the WITS web process now so fresh code can load…"
    return (
        "Owner command accepted. Force-shutting down the WITS web process now. "
        "You can relaunch with start_web_ui.bat when you're ready."
    )
