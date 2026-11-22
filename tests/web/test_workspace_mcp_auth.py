"""Tests for the workspace MCP endpoint (WP 1.4b): the auth gate, limits and mounting.

Shares the started-app fixture and helpers with ``test_workspace_mcp.py``.
"""

import pytest
from fastapi.routing import Mount

from core.guest_access import issue_guest_token
from core.workspace import clients
from tests.web.test_workspace_mcp import (  # noqa: F401 (fixtures are used by name)
    OWNER,
    _isolate,
    make_app,
    mcp_session,
    new_client,
    raw_http,
    rpc_body,
    rpc_headers,
    running,
)
from web import workspace_mcp
from web.workspace_routes import get_db

BAD_TOKEN = "wits_0123abcd_" + "A" * 43


@pytest.fixture
async def app(monkeypatch, tmp_path):
    """A started app, as in ``test_workspace_mcp.py``."""
    async with running(make_app(monkeypatch, tmp_path)) as application:
        yield application


async def _post(app, headers: dict, path: str = "/mcp/", base_url="http://localhost", **kw):
    async with raw_http(app, base_url) as http:
        return await http.post(path, headers=headers, **kw)


async def test_valid_token_lists_tools_over_raw_jsonrpc(app):
    _, token = await new_client("raw")
    resp = await _post(app, rpc_headers(token), json=rpc_body())
    assert resp.status_code == 200
    assert len(resp.json()["result"]["tools"]) == 5


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "Bearer ",
        "Bearer",
        "Bearer not-a-token",
        BAD_TOKEN,
        "Bearer " + BAD_TOKEN,
        "Bearer " + OWNER,
        "Basic " + OWNER,
    ],
)
async def test_missing_or_bad_tokens_get_401(app, authorization):
    headers = rpc_headers()
    if authorization is not None:
        headers["Authorization"] = authorization
    resp = await _post(app, headers, json=rpc_body())
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


async def test_guest_token_gets_401(app, monkeypatch):
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "guest-signing-secret")
    guest = issue_guest_token(guest_id="g1", device_id="d1", display_name="Kid")
    resp = await _post(app, rpc_headers(guest), json=rpc_body())
    assert resp.status_code == 401


async def test_revoked_client_token_gets_401(app):
    identity, token = await new_client("gone")
    assert (await _post(app, rpc_headers(token), json=rpc_body())).status_code == 200
    assert await clients.revoke_client(get_db(), identity.id)
    assert (await _post(app, rpc_headers(token), json=rpc_body())).status_code == 401


@pytest.mark.parametrize("param", ["token", "access_token", "authorization"])
async def test_query_string_token_is_ignored(app, param):
    _, token = await new_client("query")
    resp = await _post(app, rpc_headers(), path=f"/mcp/?{param}={token}", json=rpc_body())
    assert resp.status_code == 401


async def test_eleventh_failure_gets_429_and_valid_token_still_works(app):
    _, token = await new_client("survivor")
    statuses = []
    for i in range(11):
        # Reason: forwarding headers are client-controlled and must not split the count.
        headers = {**rpc_headers("nope"), "X-Forwarded-For": f"10.0.0.{i}", "X-Real-IP": str(i)}
        statuses.append((await _post(app, headers, json=rpc_body())).status_code)
    assert statuses == [401] * 10 + [429]
    assert (await _post(app, rpc_headers(token), json=rpc_body())).status_code == 200
    assert (await _post(app, rpc_headers("nope"), json=rpc_body())).status_code == 429


async def test_verify_token_exception_refuses_with_5xx(app, monkeypatch):
    _, token = await new_client("unlucky")

    async def broken(db, presented):
        raise RuntimeError("database is locked " + presented)

    monkeypatch.setattr(clients, "verify_token", broken)
    resp = await _post(app, rpc_headers(token), json=rpc_body())
    assert 500 <= resp.status_code < 600
    assert token not in resp.text and "locked" not in resp.text


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
async def test_both_paths_work_with_the_mcp_client(app, path):
    _, token = await new_client("pathy")
    async with mcp_session(app, token, path=path) as session:
        assert len((await session.list_tools()).tools) == 5


async def test_existing_get_mcp_page_still_served(app):
    async with raw_http(app) as http:
        resp = await http.get("/mcp")
    assert resp.status_code == 200 and "settings page" in resp.text


async def test_post_mcp_without_slash_is_gated(app):
    assert (await _post(app, rpc_headers(), path="/mcp", json=rpc_body())).status_code == 401


async def test_host_outside_allowlist_refused(app):
    _, token = await new_client("hosty")
    resp = await _post(app, rpc_headers(token), base_url="http://evil.example", json=rpc_body())
    assert resp.status_code == 421
    resp = await _post(app, rpc_headers(token), base_url="http://localhost:8000", json=rpc_body())
    assert resp.status_code == 200


async def test_configured_tailnet_host_allowed(monkeypatch, tmp_path):
    hosts = ["localhost", "wits.tail1234.ts.net"]
    async with running(make_app(monkeypatch, tmp_path, allowed_hosts=hosts)) as application:
        _, token = await new_client("tailnet")
        resp = await _post(
            application,
            rpc_headers(token),
            base_url="http://wits.tail1234.ts.net:8000",
            json=rpc_body(),
        )
        assert resp.status_code == 200
        resp = await _post(
            application, rpc_headers(token), base_url="http://127.0.0.1", json=rpc_body()
        )
        assert resp.status_code == 421


async def test_browser_origin_refused(app):
    _, token = await new_client("browser")
    headers = {**rpc_headers(token), "Origin": "http://evil.example"}
    assert (await _post(app, headers, json=rpc_body())).status_code == 403


async def test_body_over_256kb_gets_413(app):
    _, token = await new_client("big")
    big = b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"x":"' + b"a" * (256 * 1024)
    big += b'"}}'
    assert (await _post(app, rpc_headers(token), content=big)).status_code == 413

    async def chunks():
        for _ in range(5):
            yield b"a" * (64 * 1024)
        yield b"a"

    assert (await _post(app, rpc_headers(token), content=chunks())).status_code == 413
    # Reason: auth comes first, so an unauthenticated caller learns nothing about limits.
    assert (await _post(app, rpc_headers(), content=big)).status_code == 401


async def test_handler_refuses_without_identity(app):
    tool = next(t for t in workspace_mcp.build_tools() if t.name == "get_project_state")
    result = await tool.fn(raw={"project": "p"})
    assert result.is_error is True
    assert result.structured_content["code"] == "forbidden_scope"


async def test_disabled_workspace_mounts_nothing(monkeypatch, tmp_path):
    application = make_app(monkeypatch, tmp_path, enabled=False)
    assert not any(isinstance(r, Mount) for r in application.routes)
    assert [r.path for r in application.routes if r.path.startswith("/mcp")] == ["/mcp"]
    async with raw_http(application) as http:
        # Reason: only the GET page exists; the slash form is redirected to it, never served.
        assert (await http.post("/mcp/", json=rpc_body())).status_code == 307
        assert (await http.post("/mcp", json=rpc_body())).status_code == 405
