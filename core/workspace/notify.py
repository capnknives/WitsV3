"""Owner phone push for new workspace approvals (contract sections 6 and 8 item 6).

The push carries the client name and the words "decision waiting" only. No title, body, project
or approval id leaves the PC. The ntfy topic is a secret read from ``WITSV3_NTFY_TOPIC`` and is
never logged, so no log line or exception message here includes the request URL.
"""

import logging
import os

import httpx
from fastapi import FastAPI

from core.config import load_config

from . import approvals

logger = logging.getLogger("WitsV3.Workspace.Notify")

TOPIC_ENV = "WITSV3_NTFY_TOPIC"
PUSH_TIMEOUT_SECONDS = 5.0

# Reason: tests inject an httpx.MockTransport here so no real network call is made.
_transport: httpx.AsyncBaseTransport | None = None
_registered_listener = None


def _build_listener(url: str):
    """Return a created-listener that POSTs to ``url`` (which embeds the secret topic)."""

    async def push(approval_id: str, client_name: str) -> None:
        # Reason: approval_id is deliberately unused; it must not leave the PC.
        message = f"{client_name}: decision waiting"
        try:
            async with httpx.AsyncClient(
                timeout=PUSH_TIMEOUT_SECONDS, transport=_transport
            ) as client:
                response = await client.post(
                    url,
                    content=message.encode("utf-8"),
                    headers={"Content-Type": "text/plain; charset=utf-8"},
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("ntfy push failed: HTTP %s", exc.response.status_code)
        except httpx.HTTPError as exc:
            # Reason: log only the class; httpx messages and tracebacks can carry the URL.
            logger.warning("ntfy push failed: %s", type(exc).__name__)

    return push


def register(app: FastAPI) -> None:
    """Add the ntfy created-listener when ``workspace.push.provider`` is ``ntfy``.

    Safe to call twice: a second call replaces nothing and adds no second listener.

    Args:
        app: The FastAPI application (unused; kept for the shared ``register`` signature).
    """
    global _registered_listener
    push = load_config().workspace.push
    if push.provider != "ntfy":
        return
    topic = os.getenv(TOPIC_ENV, "").strip().strip("/")
    if not topic:
        logger.warning(
            "workspace.push.provider is ntfy but %s is unset; no push will be sent", TOPIC_ENV
        )
        return
    if _registered_listener is not None and _registered_listener in approvals._listeners:
        return
    # Reason: httpx logs "HTTP Request: POST <url>" at INFO, and the URL holds the secret topic.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _registered_listener = _build_listener(f"{push.ntfy_url.rstrip('/')}/{topic}")
    approvals.add_created_listener(_registered_listener)
