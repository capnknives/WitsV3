"""MCP endpoint for the shared agent workspace at ``/mcp`` (contract sections 3, 5 and 7).

``register(app)`` mounts an ``MCPServer`` with the five workspace tools (built in
``web.workspace_mcp_tools``) behind an ASGI gate, the only check on ``/mcp`` (the app's
middleware guards ``/api/*`` only). It fails closed: no verified client token means HTTP 401
before the SDK sees the request, and the identity reaches the handlers through a ``ContextVar``
whose default is ``None``. No client-chosen text reaches the host log: the gate answers calls to
unknown tools itself and the SDK's loggers are held at WARNING.
"""

import hmac
import json
import logging
from collections import deque

from fastapi import FastAPI
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import ClientDisconnect
from starlette.routing import Match, Mount, Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from core.config import load_config
from core.workspace import clients, ratelimit, service
from core.workspace.clients import ClientIdentity
from web.owner_controls import configured_web_token
from web.workspace_mcp_tools import TOOL_NAMES, _identity, build_tools
from web.workspace_routes import add_lifespan, get_db

__all__ = ["MOUNT_PATH", "build_tools", "register"]

logger = logging.getLogger("WitsV3.Workspace.MCP")

MOUNT_PATH = "/mcp"
MAX_BODY_BYTES = 256 * 1024


# --------------------------------------------------------------------------- auth gate


async def _respond(
    scope: Scope, send: Send, status: int, error: str, headers: list | None = None
) -> None:
    # Reason: a HEAD response must not carry a body.
    body = b"" if scope.get("method") == "HEAD" else json.dumps({"error": error}).encode("ascii")
    base = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": base + (headers or [])})
    await send({"type": "http.response.body", "body": body})


def _header_values(scope: Scope, name: bytes) -> list[bytes]:
    return [v for k, v in scope.get("headers", []) if k == name]


def _bearer(scope: Scope) -> str:
    """Return the bearer token from the one Authorization header, or "" (never the query)."""
    values = _header_values(scope, b"authorization")
    if len(values) != 1:
        return ""
    scheme, _, token = values[0].decode("latin-1").partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _is_owner_token(token: str) -> bool:
    expected = configured_web_token()
    if not expected:
        return False
    try:
        return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))
    except UnicodeEncodeError:
        return False


def _split_host(value: str) -> tuple[str, str | None] | None:
    """Split a Host header or allowlist entry into (name, port); None if malformed.

    An IPv6 literal is bracketed or has two or more colons; one colon means host:port.
    """
    value = value.strip().lower()
    if value.startswith("["):
        name, sep, rest = value.partition("]")
        if not sep or (rest and not rest.startswith(":")):
            return None
        name, port = name + "]", rest[1:] if rest else None
    elif value.count(":") >= 2:
        name, port = f"[{value}]", None
    else:
        name, _, port = value.partition(":")
        port = port if ":" in value else None
    if not name or name == "[]":
        return None
    if port is not None and not (port.isascii() and port.isdigit() and len(port) <= 5):
        return None
    return name, port


def _parse_allowed(entries: list[str]) -> list[tuple[str, str | None]]:
    return [parsed for parsed in map(_split_host, entries) if parsed is not None]


def _host_allowed(scope: Scope, allowed: list[tuple[str, str | None]]) -> bool:
    """Exact host match; an entry without a port allows that host on any numeric port."""
    values = _header_values(scope, b"host")
    if len(values) != 1:
        return False
    parsed = _split_host(values[0].decode("latin-1"))
    if parsed is None or parsed[0].startswith("[") != parsed[0].endswith("]"):
        return False
    name, port = parsed
    return any(name == a_name and a_port in (None, port) for a_name, a_port in allowed)


class _Disconnected(Exception):
    """The client went away while its body was being read."""


async def _read_body(receive: Receive) -> tuple[list[Message], bool]:
    """Buffer the body; return (messages, over the limit). Raises _Disconnected."""
    messages: list[Message] = []
    size = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            raise _Disconnected
        messages.append(message)
        size += len(message.get("body", b""))
        if size > MAX_BODY_BYTES:
            return messages, True
        if not message.get("more_body", False):
            return messages, False


def _jsonrpc_id(value: object) -> str | int | None:
    """Echo a request id only when it is short and plain; anything else becomes null."""
    if isinstance(value, str) and len(value) <= 64:
        return value
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _unknown_tool_reply(messages: list[Message]) -> bytes | None:
    """Return a JSON-RPC error body for a tools/call naming no workspace tool, else None.

    Reason: the SDK logs the requested name of a missing tool, and the host log is read by an
    LLM-driven repair agent, so such a call is answered here and never reaches the SDK. A body
    that is not a JSON object passes through unchanged for the SDK to refuse.
    """
    try:
        data = json.loads(b"".join(m.get("body", b"") for m in messages))
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or data.get("method") != "tools/call":
        return None
    params = data.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if isinstance(name, str) and name in TOOL_NAMES:
        return None
    error = {"code": -32602, "message": "Unknown tool"}
    reply = {"jsonrpc": "2.0", "id": _jsonrpc_id(data.get("id")), "error": error}
    return json.dumps(reply).encode("ascii")


class _AuthGate:
    """ASGI wrapper that admits only verified client tokens to the MCP app.

    Order: token (401/429/503), method (405), per-client HTTP bucket (429), Host (421),
    Origin (403), body size (413). Refusals never log header values: the host's log is read
    by an LLM-driven repair agent, so client text must not reach it.
    """

    def __init__(self, app: ASGIApp, allowed_hosts: list[tuple[str, str | None]]) -> None:
        self._app = app
        self._allowed = allowed_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            return
        identity = await self._verify(scope, send)
        if identity is None:
            return
        if not await self._admit(scope, send, identity):
            return
        declared = b"".join(_header_values(scope, b"content-length")[:1])
        # Reason: a chunked body has no Content-Length, so the limit is also enforced while
        # reading; the SDK then reads the buffered copy and never sees an oversized body.
        try:
            too_large = declared.isdigit() and int(declared) > MAX_BODY_BYTES
            if not too_large:
                buffered, too_large = await _read_body(receive)
        except _Disconnected:
            return
        if too_large:
            await _respond(scope, send, 413, "request body too large")
            return
        refusal = _unknown_tool_reply(buffered)
        if refusal is not None:
            headers = [(b"content-type", b"application/json")]
            headers.append((b"content-length", str(len(refusal)).encode()))
            await send({"type": "http.response.start", "status": 200, "headers": headers})
            await send({"type": "http.response.body", "body": refusal})
            return
        pending = deque(buffered)

        async def replay() -> Message:
            return pending.popleft() if pending else await receive()

        token = _identity.set(identity)
        try:
            await self._app(scope, replay, send)
        except ClientDisconnect:
            return
        finally:
            _identity.reset(token)

    async def _admit(self, scope: Scope, send: Send, identity: ClientIdentity) -> bool:
        """Apply the method, HTTP rate, Host and Origin checks to a verified request."""
        # Reason: GET would open a handshake-era SSE stream that outlives a revoke; every
        # exchange this endpoint serves is one POST.
        if scope.get("method") != "POST":
            await _respond(scope, send, 405, "method not allowed", [(b"allow", b"POST")])
            return False
        # Reason: the handlers' bucket covers tool calls only; this one caps everything else
        # an authenticated client can send (tools/list, ping, unknown tools, malformed calls).
        if not ratelimit.allow_call(identity.id + ":http", identity.rate_per_min * 4):
            await _respond(scope, send, 429, "rate limit exceeded", [(b"retry-after", b"60")])
            return False
        if not _host_allowed(scope, self._allowed):
            await _respond(scope, send, 421, "host not allowed")
            return False
        if _header_values(scope, b"origin"):
            await _respond(scope, send, 403, "origin not allowed")
            return False
        return True

    async def _verify(self, scope: Scope, send: Send) -> ClientIdentity | None:
        """Return the verified identity, or answer 401, 429 or 503 and return None."""
        token = _bearer(scope)
        identity = None
        if token and not _is_owner_token(token):
            try:
                identity = await clients.verify_token(get_db(), token)
            except Exception as exc:
                # Reason: a verifier failure must refuse, never admit; only the exception
                # type is logged so no token or query text can reach the log.
                logger.error("workspace token verification failed (%s)", type(exc).__name__)
                await _respond(scope, send, 503, "authentication unavailable")
                return None
        if isinstance(identity, ClientIdentity):
            return identity
        # Reason: the address comes from the connection itself; X-Forwarded-For and similar
        # headers are client-controlled and would let one caller dodge the limiter.
        client = scope.get("client")
        addr = str(client[0]) if client else "unknown"
        if ratelimit.auth_failure_blocked(addr):
            await _respond(
                scope, send, 429, "too many failed authentications", [(b"retry-after", b"60")]
            )
        else:
            await _respond(scope, send, 401, "unauthorized", [(b"www-authenticate", b"Bearer")])
        return None


class _ExactPath:
    """Serve ``/mcp`` (no trailing slash) through the mount instead of a slash redirect."""

    def __init__(self, mount: Mount) -> None:
        self._mount = mount

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope["path"] + "/"
        child = {**scope, "path": path, "raw_path": path.encode("latin-1")}
        await self._mount(child, receive, send)


class _EventStreamGet(Route):
    """Matches only ``GET /mcp`` asking for an event stream, so it reaches the gate.

    Reason: the host serves an HTML settings page at ``GET /mcp``; a handshake-era client
    opening its stream there would parse HTML as events. A browser (``Accept: text/html``)
    still falls through to the page.
    """

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        if scope["type"] != "http" or scope.get("method") != "GET":
            return Match.NONE, {}
        accept = b",".join(_header_values(scope, b"accept")).lower()
        if b"text/event-stream" not in accept:
            return Match.NONE, {}
        return super().matches(scope)


# --------------------------------------------------------------------------- mounting


def _sdk_allowed_hosts(allowed: list[tuple[str, str | None]]) -> list[str]:
    """The same list in the SDK's form; the gate's exact check runs first."""
    hosts: list[str] = []
    for name, port in allowed:
        hosts.extend([f"{name}:{port}"] if port else [name, f"{name}:*"])
    return hosts


def _build_server() -> MCPServer:
    """Construct the MCPServer without letting it change the host's root logger.

    MCPServer's constructor calls ``logging.basicConfig``; with no root handler yet, the SDK's
    handler would become the root handler and the host's own logging setup would do nothing.
    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        return MCPServer("wits-workspace", tools=build_tools())
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def register(app: FastAPI) -> None:
    """Mount the MCP endpoint at ``/mcp`` behind the client-token gate.

    Called by ``register_workspace`` only when the workspace is enabled. Applies the lease and
    approval defaults from config, registers the session manager with the host lifespan (a
    mounted app's own lifespan never runs) and adds the mount plus an exact ``/mcp`` route,
    and in front of every route one that sends event-stream ``GET /mcp`` to the gate.

    Args:
        app: The FastAPI application.
    """
    settings = load_config().workspace
    # Reason: the SDK logs an INFO line per request, some carrying client-chosen text, and the
    # host log is read by an LLM-driven repair agent.
    logging.getLogger("mcp").setLevel(logging.WARNING)
    service.configure(settings.lease_minutes, settings.approval_ttl_minutes)
    server = _build_server()
    allowed = _parse_allowed(settings.allowed_hosts)
    # Reason: a second layer behind the gate's own Host and Origin checks; with no allowed
    # origins a browser page (which always sends Origin) is refused, while agents send none.
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_sdk_allowed_hosts(allowed),
        allowed_origins=[],
    )
    # Reason: stateless so a handshake-era client also gets one exchange per request; with
    # the gate admitting POST only, no session or stream outlives the request whose token
    # was checked.
    mcp_app = server.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        max_request_body_size=MAX_BODY_BYTES,
        transport_security=security,
    )
    add_lifespan(lambda: server.session_manager.run())
    gate = _AuthGate(mcp_app, allowed)
    mount = Mount(MOUNT_PATH, app=gate)
    app.router.routes.insert(0, _EventStreamGet(MOUNT_PATH, endpoint=gate, include_in_schema=False))
    app.router.routes.append(mount)
    # Reason: a GET /mcp page route registered earlier still wins for GET; this route takes
    # the other methods so POST /mcp needs no slash redirect.
    app.router.routes.append(Route(MOUNT_PATH, endpoint=_ExactPath(mount)))
