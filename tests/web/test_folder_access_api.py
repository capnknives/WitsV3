"""Tests for the owner folder-grant routes and start-up wiring (WP F3)."""

import hashlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import path_access
from core.config import WitsV3Config, WorkspaceSettings
from core.guest_access import issue_guest_token
from core.workspace import approvals, clients, folder_access, ratelimit
from tests.web.test_workspace_mcp import make_app, post, running
from web import workspace_routes
from web.workspace_routes import get_db, register_workspace

OWNER = "owner-secret"
LIST = "/api/workspace/folder-grants"
INTERNAL = "cli_internal_wits"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(workspace_routes, "_db", None)
    monkeypatch.setattr(workspace_routes, "_lifespan_factories", [])
    monkeypatch.setattr(
        workspace_routes,
        "_OPTIONAL_MODULES",
        ("web.workspace_clients_api", "web.workspace_folder_api"),
    )
    monkeypatch.setattr(path_access, "grants_provider", path_access._no_grants)
    monkeypatch.setattr(approvals, "_listeners", [])
    monkeypatch.setenv("WITSV3_WEB_TOKEN", OWNER)
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "guest-signing-secret")


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


def _owner() -> dict:
    return {"Authorization": f"Bearer {OWNER}"}


def _grant(client, tmp_path) -> str:
    """Request and approve a folder through the real code path; return the grant id."""
    folder = tmp_path / "outside"
    folder.mkdir(exist_ok=True)
    cfg = SimpleNamespace(folders=SimpleNamespace(base=str(tmp_path / "base")), security=None)

    async def go():
        db = get_db()
        req = await folder_access.request(db, str(folder), "read", "why", config=cfg)
        await approvals.resolve(db, req["approval_id"], "approve", None)
        return (await folder_access.list_grants(db))[0]["id"]

    return client.portal.call(go)


def _client_token(client) -> str:
    async def go():
        return (await clients.create_client(get_db(), "agent", ["read"], ["*"]))[1]

    return client.portal.call(go)


@pytest.mark.parametrize("route", ["list", "revoke"])
def test_routes_refuse_non_owner_tokens(client, tmp_path, route):
    grant_id = _grant(client, tmp_path)
    guest = issue_guest_token(guest_id="g1", device_id="d1", display_name="Kid")
    for headers in (
        {},
        {"Authorization": "Bearer "},
        {"Authorization": f"Bearer {guest}"},
        {"Authorization": f"Bearer {_client_token(client) if route == 'list' else 'wits_x'}"},
    ):
        if route == "list":
            resp = client.get(LIST, headers=headers)
        else:
            resp = client.post(f"{LIST}/{grant_id}/revoke", headers=headers)
        assert resp.status_code == 401
    assert client.get(LIST, headers=_owner()).json()["grants"][0]["live"] is True


def test_revoke_route_refuses_client_token(client, tmp_path):
    grant_id = _grant(client, tmp_path)
    token = _client_token(client)
    resp = client.post(f"{LIST}/{grant_id}/revoke", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_owner_lists_and_revokes(client, tmp_path):
    grant_id = _grant(client, tmp_path)
    assert client.post(f"{LIST}/{grant_id}/revoke", headers=_owner()).json()["revoked"] is True
    assert client.post(f"{LIST}/{grant_id}/revoke", headers=_owner()).status_code == 404
    assert client.get(LIST, headers=_owner()).json()["grants"][0]["live"] is False


def test_provider_is_live_while_running_and_reset_after(monkeypatch, tmp_path):
    with TestClient(_make_app(monkeypatch, tmp_path)) as c:
        assert path_access.grants_provider is not path_access._no_grants
        _grant(c, tmp_path)
        assert len(c.portal.call(path_access._load_grants)) == 1
    assert path_access.grants_provider is path_access._no_grants


def test_disabled_workspace_keeps_no_grants(monkeypatch, tmp_path):
    with TestClient(_make_app(monkeypatch, tmp_path, enabled=False)) as c:
        assert path_access.grants_provider is path_access._no_grants
        assert c.get(LIST, headers=_owner()).status_code == 404


def test_internal_client_hidden_from_list_and_not_revocable(client):
    listed = client.get("/api/workspace/clients", headers=_owner()).json()["clients"]
    assert all(c["id"] != INTERNAL for c in listed)
    resp = client.post(f"/api/workspace/clients/{INTERNAL}/revoke", headers=_owner())
    assert resp.status_code == 404


async def test_internal_client_cannot_authenticate_at_mcp(monkeypatch, tmp_path):
    # Reason: even if the placeholder row were given a real-looking prefix and hash, the
    # internal id must never verify.
    token = "wits_0badc0de_" + "A" * 43
    monkeypatch.setattr(workspace_routes, "_OPTIONAL_MODULES", ("web.workspace_mcp",))
    ratelimit.reset()
    async with running(make_app(monkeypatch, tmp_path)) as app:
        async with get_db().write() as conn:
            await conn.execute(
                "UPDATE clients SET token_prefix = '0badc0de', token_hash = :h WHERE id = :id",
                {"h": hashlib.sha256(token.encode()).hexdigest(), "id": INTERNAL},
            )
        for presented in (token, "internal", "x", "wits_internal_" + "A" * 43):
            assert (await post(app, token=presented)).status_code == 401
