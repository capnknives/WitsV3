"""Tests for the workspace MCP endpoint (WP 1.4b): the auth gate, limits and mounting.

Shares the started-app helpers with ``test_workspace_mcp.py``. Gate checks run on both
protocol paths (``BOTH_PATHS``): with ``MCP-Protocol-Version: 2026-07-28`` and without it.
"""

import logging

import pytest
from fastapi.routing import Mount

from core.guest_access import issue_guest_token
from core.workspace import clients
from tests.web.test_workspace_mcp import (  # noqa: F401 (fixtures are used by name)
    BOTH_PATHS,
    OWNER,
    _isolate,
    make_app,
    mcp_session,
    new_client,
    post,
    raw_http,
    rpc_body,
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


@BOTH_PATHS
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
async def test_missing_or_bad_tokens_get_401(app, authorization, modern):
    headers = {} if authorization is None else {"Authorization": authorization}
    resp = await post(app, modern=modern, headers=headers)
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


async def test_guest_token_gets_401(app, monkeypatch):
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "guest-signing-secret")
    guest = issue_guest_token(guest_id="g1", device_id="d1", display_name="Kid")
    assert (await post(app, token=guest)).status_code == 401


async def test_owner_token_shaped_like_a_client_token_never_reaches_verify(app, monkeypatch):
    _, client_shaped = await new_client("lookalike")
    monkeypatch.setenv("WITSV3_WEB_TOKEN", client_shaped)
    _, other = await new_client("normal")
    seen: list[str] = []
    real = clients.verify_token

    async def spy(db, presented):
        seen.append(presented)
        return await real(db, presented)

    monkeypatch.setattr(clients, "verify_token", spy)
    assert (await post(app, token=client_shaped)).status_code == 401
    assert seen == []
    assert (await post(app, token=other)).status_code == 200
    assert seen == [other]


@pytest.mark.parametrize("second", ["valid", "junk"])
async def test_duplicate_authorization_headers_get_401(app, second):
    _, token = await new_client("twice")
    other = token if second == "valid" else "junk"
    headers = [
        ("Content-Type", "application/json"),
        ("Accept", "application/json"),
        ("Authorization", f"Bearer {token}"),
        ("Authorization", f"Bearer {other}"),
    ]
    async with raw_http(app) as http:
        resp = await http.post("/mcp/", headers=headers, json=rpc_body())
    assert resp.status_code == 401


async def test_revoked_client_token_gets_401(app):
    identity, token = await new_client("gone")
    assert (await post(app, token=token)).status_code == 200
    assert await clients.revoke_client(get_db(), identity.id)
    assert (await post(app, token=token)).status_code == 401


@pytest.mark.parametrize("param", ["token", "access_token", "authorization"])
async def test_query_string_token_is_ignored(app, param):
    _, token = await new_client("query")
    assert (await post(app, path=f"/mcp/?{param}={token}")).status_code == 401


async def test_eleventh_failure_gets_429_and_valid_token_still_works(app):
    _, token = await new_client("survivor")
    statuses = []
    for i in range(11):
        # Reason: forwarding headers are client-controlled and must not split the count.
        spoof = {"X-Forwarded-For": f"10.0.0.{i}", "X-Real-IP": str(i)}
        statuses.append((await post(app, token="nope", headers=spoof)).status_code)
    assert statuses == [401] * 10 + [429]
    assert (await post(app, token=token)).status_code == 200
    assert (await post(app, token="nope")).status_code == 429


async def test_verify_token_exception_refuses_with_5xx(app, monkeypatch):
    _, token = await new_client("unlucky")

    async def broken(db, presented):
        raise RuntimeError("database is locked " + presented)

    monkeypatch.setattr(clients, "verify_token", broken)
    resp = await post(app, token=token)
    assert 500 <= resp.status_code < 600
    assert token not in resp.text and "locked" not in resp.text


async def test_http_bucket_limits_everything_a_client_sends(app):
    _, token = await new_client("chatty", rate=1)
    statuses = [(await post(app, token=token)).status_code for _ in range(4)]
    fifth = await post(app, token=token)
    assert statuses == [200] * 4
    assert fifth.status_code == 429 and fifth.headers["retry-after"] == "60"


@pytest.mark.parametrize("method", ["GET", "DELETE", "PUT", "OPTIONS", "HEAD"])
async def test_only_post_is_served(app, method):
    _, token = await new_client("poster")
    async with raw_http(app) as http:
        authed = await http.request(method, "/mcp/", headers={"Authorization": f"Bearer {token}"})
        anonymous = await http.request(method, "/mcp/")
    assert authed.status_code == 405 and authed.headers["allow"] == "POST"
    assert anonymous.status_code == 401


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
    assert (await post(app, path="/mcp")).status_code == 401


@BOTH_PATHS
async def test_hostile_host_and_origin_refused_without_logging_them(app, caplog, modern):
    _, token = await new_client("hosty")
    host, origin = "evil-host-marker.example", "http://evil-origin-marker.example"
    with caplog.at_level(logging.DEBUG):
        bad_host = await post(app, token=token, modern=modern, headers={"Host": host})
        bad_origin = await post(app, token=token, modern=modern, headers={"Origin": origin})
    assert bad_host.status_code == 421 and bad_origin.status_code == 403
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "marker" not in logged


@pytest.mark.parametrize(
    "host, status",
    [
        ("localhost", 200),
        ("localhost:8000", 200),
        ("localhost:8000.evil.example", 421),
        ("localhost.evil.example", 421),
        ("wits.ts.net:8000", 200),
        ("wits.ts.net", 421),
        ("wits.ts.net:9000", 421),
        ("10.0.0.5", 200),
        ("10.0.0.5:8000", 200),
        ("[::1]", 200),
        ("[::1]:8000", 200),
        ("[::2]:8000", 421),
        ("evil.example", 421),
    ],
)
async def test_allowed_hosts_matching(monkeypatch, tmp_path, host, status):
    hosts = ["localhost", "wits.ts.net:8000", "10.0.0.5", "::1"]
    async with running(make_app(monkeypatch, tmp_path, allowed_hosts=hosts)) as application:
        _, token = await new_client("tailnet")
        assert (await post(application, token=token, headers={"Host": host})).status_code == status


@BOTH_PATHS
async def test_body_over_256kb_gets_413(app, modern):
    _, token = await new_client("big")
    big = b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"x":"' + b"a" * (256 * 1024)
    big += b'"}}'
    assert (await post(app, token=token, modern=modern, content=big)).status_code == 413

    async def chunks():
        for _ in range(5):
            yield b"a" * (64 * 1024)
        yield b"a"

    resp = await post(app, token=token, modern=modern, content=chunks())
    assert resp.status_code == 413
    # Reason: auth comes first, so an unauthenticated caller learns nothing about limits.
    assert (await post(app, modern=modern, content=big)).status_code == 401


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
