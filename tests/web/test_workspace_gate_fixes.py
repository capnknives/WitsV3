"""Tests for the Stage 1 gate review fixes (WP 1.8) on the web side.

Covers: unknown tool calls answered by the gate and no client text in the log (item 1), the
``web_ui.require_auth`` refusal (item 4), importing without aiosqlite (item 5c) and the
event-stream ``GET /mcp`` route (item 6). MCP checks run on the real mounted app, on both
protocol paths, with the helpers from ``test_workspace_mcp.py``.
"""

import importlib
import json
import logging
import sys

import pytest
from fastapi import FastAPI

import web
from core.config import WebUISettings, WitsV3Config, WorkspaceSettings
from tests.web.test_workspace_mcp import (  # noqa: F401 (fixtures are used by name)
    BOTH_PATHS,
    MODERN,
    _isolate,
    make_app,
    new_client,
    raw_http,
    rpc,
    running,
)
from web import workspace_routes
from web.workspace_routes import register_workspace

MARKER = "zq-client-marker"


@pytest.fixture
async def app(monkeypatch, tmp_path):
    """A started app with a GET /mcp settings page registered before the workspace."""
    async with running(make_app(monkeypatch, tmp_path)) as application:
        yield application


async def _send(app, token, body, modern, method="tools/call", name=None):
    """POST a raw body; ``body`` is a dict (sent as JSON) or bytes (sent as-is)."""
    headers, _ = rpc("tools/list", None, token, modern)
    if modern:
        headers["Mcp-Method"] = method
        if name is not None:
            headers["Mcp-Name"] = name
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    async with raw_http(app) as http:
        return await http.post("/mcp/", headers=headers, content=content)


def _call(name, req_id=1, modern=True, params=None) -> dict:
    params = {"name": name, "arguments": {}} if params is None else params
    if modern and isinstance(params, dict):
        params["_meta"] = {
            "io.modelcontextprotocol/protocolVersion": MODERN,
            "io.modelcontextprotocol/clientCapabilities": {},
        }
    return {"jsonrpc": "2.0", "id": req_id, "method": "tools/call", "params": params}


# ----------------------------------------------------------------- item 1: unknown tools


@BOTH_PATHS
@pytest.mark.parametrize(
    "req_id, echoed",
    [(7, 7), ("abc", "abc"), ("x" * 64, "x" * 64), ("x" * 65, None), (True, None), (1.5, None)],
)
async def test_unknown_tool_answered_by_gate(app, modern, req_id, echoed):
    _, token = await new_client("caller")
    resp = await _send(app, token, _call(MARKER, req_id, modern), modern, name=MARKER)
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "jsonrpc": "2.0",
        "id": echoed,
        "error": {"code": -32602, "message": "Unknown tool"},
    }
    assert MARKER not in resp.text


@BOTH_PATHS
@pytest.mark.parametrize("params", [["a", "list"], "text", {"name": 5}, {}])
async def test_tools_call_with_malformed_params_is_unknown_tool(app, modern, params):
    _, token = await new_client("bad-params")
    resp = await _send(app, token, _call(None, modern=modern, params=params), modern)
    assert resp.json()["error"] == {"code": -32602, "message": "Unknown tool"}


@BOTH_PATHS
async def test_known_tool_and_non_object_bodies_still_reach_the_sdk(app, modern):
    _, token = await new_client("known")
    known = _call("get_project_state", modern=modern)
    known["params"]["arguments"] = {"project": "demo"}
    resp = await _send(app, token, known, modern, name="get_project_state")
    assert resp.status_code == 200 and "result" in resp.json()
    for raw in (b"not json", b"[1, 2]", b'"tools/call"'):
        resp = await _send(app, token, raw, modern)
        assert "Unknown tool" not in resp.text


@BOTH_PATHS
async def test_no_client_text_reaches_the_log(app, caplog, modern):
    _, token = await new_client("logger")
    long_name = MARKER + "y" * 5000
    bodies = [
        (_call(MARKER, modern=modern), "tools/call", MARKER),
        (_call(long_name, modern=modern), "tools/call", None),
        (_call("get_project_state", modern=modern, params=[MARKER]), "tools/call", None),
        (
            _call(
                "get_project_state",
                modern=modern,
                params={"name": "get_project_state", "arguments": MARKER},
            ),
            "tools/call",
            "get_project_state",
        ),
    ]
    unknown_method = _call("x", modern=modern)
    unknown_method["method"] = MARKER + "/method"
    bodies.append((unknown_method, MARKER + "/method", None))
    caplog.set_level(logging.DEBUG)
    for body, method, name in bodies:
        await _send(app, token, body, modern, method=method, name=name)
    leaked = [r for r in caplog.records if MARKER in r.getMessage()]
    assert leaked == []
    assert logging.getLogger("mcp").level == logging.WARNING


# ----------------------------------------------------------------- item 6: GET /mcp stream


async def test_event_stream_get_without_token_gets_401(app):
    async with raw_http(app) as http:
        resp = await http.get("/mcp", headers={"Accept": "text/event-stream"})
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


async def test_event_stream_get_with_token_gets_405(app):
    _, token = await new_client("streamer")
    headers = {"Accept": "application/json, Text/Event-Stream", "Authorization": f"Bearer {token}"}
    async with raw_http(app) as http:
        resp = await http.get("/mcp", headers=headers)
    assert resp.status_code == 405
    assert resp.headers["allow"] == "POST"


async def test_browser_get_still_reaches_settings_page(app):
    async with raw_http(app) as http:
        resp = await http.get("/mcp", headers={"Accept": "text/html,application/xhtml+xml"})
        bare = await http.get("/mcp")
    assert resp.status_code == 200 and "settings page" in resp.text
    assert bare.status_code == 200 and "settings page" in bare.text


# ----------------------------------------------------------------- item 4: require_auth


def _config(monkeypatch, tmp_path, module, enabled: bool, require_auth: bool) -> None:
    config = WitsV3Config(
        workspace=WorkspaceSettings(db_path=str(tmp_path / "ws" / "w.db"), enabled=enabled),
        web_ui=WebUISettings(require_auth=require_auth),
    )
    monkeypatch.setattr(module, "load_config", lambda *a, **k: config)


def test_enabled_with_require_auth_off_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace_routes, "_OPTIONAL_MODULES", ())
    _config(monkeypatch, tmp_path, workspace_routes, enabled=True, require_auth=False)
    with pytest.raises(RuntimeError) as info:
        register_workspace(FastAPI())
    assert "workspace.enabled" in str(info.value)
    assert "web_ui.require_auth" in str(info.value)


@pytest.mark.parametrize("enabled, require_auth", [(True, True), (False, False), (False, True)])
def test_other_combinations_start(monkeypatch, tmp_path, enabled, require_auth):
    monkeypatch.setattr(workspace_routes, "_OPTIONAL_MODULES", ())
    _config(monkeypatch, tmp_path, workspace_routes, enabled, require_auth)
    register_workspace(FastAPI())


# ----------------------------------------------------------------- item 5c: lazy aiosqlite


def test_disabled_workspace_imports_and_registers_without_aiosqlite(monkeypatch, tmp_path):
    # Reason: a None entry makes ``import aiosqlite`` raise, as in an environment without it.
    monkeypatch.setitem(sys.modules, "aiosqlite", None)
    monkeypatch.delitem(sys.modules, "core.workspace.db", raising=False)
    monkeypatch.delitem(sys.modules, "web.workspace_routes")
    # Reason: importing rebinds the package attribute; monkeypatch restores the original.
    monkeypatch.setattr(web, "workspace_routes", workspace_routes)
    fresh = importlib.import_module("web.workspace_routes")
    assert fresh is not workspace_routes
    _config(monkeypatch, tmp_path, fresh, enabled=False, require_auth=True)
    app = FastAPI()
    before = len(app.routes)
    fresh.register_workspace(app)
    assert len(app.routes) == before
    assert "core.workspace.db" not in sys.modules
