"""Tests for the Inbox link on the main page (script, markup and serving)."""

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.config import WitsV3Config, WorkspaceSettings
from web import workspace_routes
from web.workspace_routes import register_workspace

STATIC = Path(__file__).resolve().parents[2] / "web" / "static"
SCRIPT = STATIC / "inbox-link.js"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(workspace_routes, "_db", None)
    monkeypatch.setattr(workspace_routes, "_lifespan_factories", [])
    monkeypatch.setattr(workspace_routes, "_OPTIONAL_MODULES", ("web.workspace_approvals_api",))
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "owner-secret")


def _client(monkeypatch, tmp_path, enabled):
    settings = WorkspaceSettings(db_path=str(tmp_path / "ws" / "workspace.db"), enabled=enabled)
    config = WitsV3Config(workspace=settings)
    monkeypatch.setattr(workspace_routes, "load_config", lambda *a, **k: config)
    app = FastAPI()
    register_workspace(app)
    return TestClient(app)


def test_index_has_hidden_inbox_link_and_script():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    tag = re.search(r'<a id="inbox-link"[^>]*>', html)
    assert tag is not None
    assert 'href="/inbox"' in tag.group(0)
    assert re.search(r"\shidden[\s>]", tag.group(0))
    assert '<script src="/inbox-link.js"></script>' in html
    assert html.index("/app.js") < html.index("/inbox-link.js")


def test_script_served_when_workspace_enabled(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, True) as c:
        res = c.get("/inbox-link.js")
    assert res.status_code == 200
    assert "javascript" in res.headers["content-type"]
    assert res.headers["cache-control"] == "no-store"


def test_script_not_served_when_workspace_disabled(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, False) as c:
        assert c.get("/inbox-link.js").status_code == 404


@pytest.mark.parametrize("sink", ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"])
def test_script_has_no_html_sinks(sink):
    assert sink not in SCRIPT.read_text(encoding="utf-8")


def test_script_uses_api_and_token_key():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "/api/workspace/approvals" in src
    assert "wits_token" in src
