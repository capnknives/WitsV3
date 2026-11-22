"""Tests for the Decision Inbox owner routes and page (WP 1.5)."""

import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.config import WitsV3Config, WorkspaceSettings
from core.workspace import approvals, clients, folder_access
from core.workspace.store import WorkspaceError
from web import workspace_routes
from web.workspace_routes import get_db, register_workspace

OWNER = "owner-secret"
STATIC = Path(__file__).resolve().parents[2] / "web" / "static"
LIST = "/api/workspace/approvals"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(workspace_routes, "_db", None)
    monkeypatch.setattr(workspace_routes, "_lifespan_factories", [])
    monkeypatch.setattr(workspace_routes, "_OPTIONAL_MODULES", ("web.workspace_approvals_api",))
    monkeypatch.setenv("WITSV3_WEB_TOKEN", OWNER)


def _make_app(monkeypatch, tmp_path, enabled=True) -> FastAPI:
    settings = WorkspaceSettings(db_path=str(tmp_path / "ws" / "workspace.db"), enabled=enabled)
    config = WitsV3Config(workspace=settings)
    monkeypatch.setattr(workspace_routes, "load_config", lambda *a, **k: config)
    app = FastAPI()
    register_workspace(app)
    return app


@pytest.fixture
def client(monkeypatch, tmp_path):
    with TestClient(_make_app(monkeypatch, tmp_path)) as c:
        yield c


def _run(client: TestClient, coro_fn):
    """Run an async DB helper on the app's own event loop."""
    return client.portal.call(coro_fn)


def _seed(client, options=None, ttl=3600, title="Deploy?", body="Ship it", flagged=0):
    async def go():
        db = get_db()
        # Reason: names are unique, so a second seed in one test reuses the first client.
        try:
            identity, token = await clients.create_client(
                db, "Cursor Agent", ["decide:request"], ["*"]
            )
        except WorkspaceError:
            identity, token = await clients.create_client(
                db, "Cursor Agent 2", ["decide:request"], ["*"]
            )
        res = await approvals.create(
            db, identity.id, "proj", None, title, body, options, ttl, flagged
        )
        return identity, token, res["approval_id"]

    return _run(client, go)


def _owner() -> dict:
    return {"Authorization": f"Bearer {OWNER}"}


def _resolve_url(approval_id: str) -> str:
    return f"{LIST}/{approval_id}/resolve"


@pytest.mark.parametrize("route", ["list", "resolve"])
def test_routes_refuse_non_owner_tokens(client, route):
    _, client_token, approval_id = _seed(client)
    headers_list = [
        {},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer guest_abcdef.0123456789"},
        {"Authorization": f"Bearer {client_token}"},
    ]
    for headers in headers_list:
        if route == "list":
            resp = client.get(LIST, headers=headers)
        else:
            resp = client.post(
                _resolve_url(approval_id), json={"choice": "approve"}, headers=headers
            )
        assert resp.status_code == 401, headers
    # Nothing was resolved by the refused attempts.
    assert len(client.get(LIST, headers=_owner()).json()["approvals"]) == 1


def test_owner_lists_pending_with_options_client_and_flag(client):
    _seed(client, options=["approve", "later"], flagged=1)
    items = client.get(LIST, headers=_owner()).json()["approvals"]
    assert len(items) == 1
    item = items[0]
    assert item["client"] == "Cursor Agent"
    assert item["options"] == ["approve", "later", "deny"]
    assert item["flagged"] == 1
    assert item["title"] == "Deploy?" and item["project"] == "proj"


@pytest.mark.parametrize(
    "choice,status",
    [("approve", "approved"), ("deny", "denied"), ("later", "answered")],
)
def test_resolve_statuses_and_second_resolve_conflicts(client, choice, status):
    _, _, approval_id = _seed(client, options=["approve", "later"])
    resp = client.post(_resolve_url(approval_id), json={"choice": choice}, headers=_owner())
    assert resp.status_code == 200
    assert resp.json()["status"] == status and resp.json()["choice"] == choice
    again = client.post(_resolve_url(approval_id), json={"choice": choice}, headers=_owner())
    assert again.status_code == 409
    assert client.get(LIST, headers=_owner()).json()["approvals"] == []


def test_unknown_id_and_expired_give_409(client):
    resp = client.post(_resolve_url("apr_nope"), json={"choice": "approve"}, headers=_owner())
    assert resp.status_code == 409
    _, _, approval_id = _seed(client)

    async def expire():
        async with get_db().write() as conn:
            await conn.execute(
                "UPDATE approvals SET expires_at = :t WHERE id = :id",
                {"t": int(time.time()) - 5, "id": approval_id},
            )

    _run(client, expire)
    resp = client.post(_resolve_url(approval_id), json={"choice": "approve"}, headers=_owner())
    assert resp.status_code == 409


def test_choice_outside_options_is_400_and_leaves_pending(client):
    _, _, approval_id = _seed(client)
    resp = client.post(_resolve_url(approval_id), json={"choice": "maybe"}, headers=_owner())
    assert resp.status_code == 400
    assert len(client.get(LIST, headers=_owner()).json()["approvals"]) == 1


def test_note_is_cleaned_and_over_limit_refused(client):
    _, _, first = _seed(client)
    resp = client.post(
        _resolve_url(first), json={"choice": "approve", "note": "ok​ay"}, headers=_owner()
    )
    assert resp.status_code == 200 and resp.json()["note"] == "okay"
    _, _, second = _seed(client)
    resp = client.post(
        _resolve_url(second), json={"choice": "approve", "note": "x" * 2001}, headers=_owner()
    )
    assert resp.status_code == 400
    assert any(a["id"] == second for a in client.get(LIST, headers=_owner()).json()["approvals"])


def test_page_and_script_served_no_store(client):
    for path, marker in (("/inbox", "Decision Inbox"), ("/inbox.js", "textContent")):
        resp = client.get(path)
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == "no-store"
        assert marker in resp.text


def test_page_not_served_when_workspace_disabled(monkeypatch, tmp_path):
    with TestClient(_make_app(monkeypatch, tmp_path, enabled=False)) as c:
        assert c.get("/inbox").status_code == 404
        assert c.get("/inbox.js").status_code == 404


def test_inbox_js_has_no_html_sinks():
    source = (STATIC / "inbox.js").read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in source, sink
    assert re.search(r"\bsetAttribute\b|\.style\b|\.href\b|\.src\b", source) is None


def test_page_has_folders_section_and_button_labels():
    source = (STATIC / "inbox.html").read_text(encoding="utf-8") + (STATIC / "inbox.js").read_text(
        encoding="utf-8"
    )
    for text in ("Folders WITS can use", "Approve 30 days", "One day", "Deny", "Revoke"):
        assert text in source, text
    for text in ("Read files in this folder", "Read and save files in this folder"):
        assert text in source


def test_no_html_sinks_in_any_inbox_script():
    for path in STATIC.glob("inbox*.js"):
        source = path.read_text(encoding="utf-8")
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML"):
            assert sink not in source, (path.name, sink)


def test_folder_request_carries_path_mode_reason(client, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    cfg = SimpleNamespace(folders=SimpleNamespace(base=str(tmp_path / "base")), security=None)
    _seed(client, title="Plain", body="other")

    async def ask():
        return await folder_access.request(
            get_db(), str(folder), "read_write", "to file", config=cfg
        )

    res = _run(client, ask)
    assert res["status"] == "pending"
    items = client.get(LIST, headers=_owner()).json()["approvals"]
    by_id = {i["id"]: i for i in items}
    folder_item = by_id[res["approval_id"]]
    assert folder_item["folder"] == {"path": str(folder), "mode": "read_write", "reason": "to file"}
    assert folder_item["options"] == ["approve", "one day", "deny"]
    plain = next(i for i in items if i["id"] != res["approval_id"])
    assert "folder" not in plain
