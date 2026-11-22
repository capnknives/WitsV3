"""Tests for the workspace gateway shell (WP 1.1b)."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from core.config import WitsV3Config, WorkspaceSettings
from web import workspace_routes
from web.workspace_routes import add_lifespan, get_db, register_workspace, require_owner

OWNER = "owner-secret"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Keep module-level workspace state from leaking between tests."""
    monkeypatch.setattr(workspace_routes, "_db", None)
    monkeypatch.setattr(workspace_routes, "_lifespan_factories", [])
    monkeypatch.setattr(workspace_routes, "_OPTIONAL_MODULES", ())
    monkeypatch.setenv("WITSV3_WEB_TOKEN", OWNER)


def _use_settings(monkeypatch, tmp_path: Path, **overrides) -> Path:
    db_path = tmp_path / "ws" / "workspace.db"
    settings = WorkspaceSettings(db_path=str(db_path), **overrides)
    config = WitsV3Config(workspace=settings)
    monkeypatch.setattr(workspace_routes, "load_config", lambda *a, **k: config)
    return db_path


def _app_with_startup_marker(events: list) -> FastAPI:
    @asynccontextmanager
    async def original(_app):
        events.append("app-start")
        yield
        events.append("app-stop")

    return FastAPI(lifespan=original)


def test_config_default_is_disabled():
    assert WitsV3Config().workspace.enabled is False


def test_disabled_mounts_nothing_and_creates_no_db(monkeypatch, tmp_path):
    db_path = _use_settings(monkeypatch, tmp_path, enabled=False)
    events: list = []
    app = _app_with_startup_marker(events)
    original = app.router.lifespan_context
    before = len(app.routes)
    register_workspace(app)
    assert app.router.lifespan_context is original
    assert len(app.routes) == before
    with TestClient(app):
        pass
    assert events == ["app-start", "app-stop"]
    assert not db_path.exists() and not db_path.parent.exists()
    with pytest.raises(RuntimeError):
        get_db()


@pytest.mark.parametrize("unset", [True, False])
def test_enabled_without_token_refuses(monkeypatch, tmp_path, unset):
    _use_settings(monkeypatch, tmp_path, enabled=True)
    if unset:
        monkeypatch.delenv("WITSV3_WEB_TOKEN")
    else:
        monkeypatch.setenv("WITSV3_WEB_TOKEN", "  ")
    with pytest.raises(RuntimeError, match="WITSV3_WEB_TOKEN"):
        register_workspace(FastAPI())


def _owner_app() -> TestClient:
    app = FastAPI()

    @app.get("/owner-only", dependencies=[Depends(require_owner)])
    async def owner_only():
        return {"ok": True}

    return TestClient(app)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer guest_abcdef.0123456789"},
        {"Authorization": "Bearer wits_0123abcd_" + "A" * 43},
        {"Authorization": "Bearer " + OWNER + "x"},
        {"Authorization": "Basic " + OWNER},
    ],
)
def test_require_owner_refuses(headers):
    assert _owner_app().get("/owner-only", headers=headers).status_code == 401


def test_require_owner_accepts_owner_token():
    resp = _owner_app().get("/owner-only", headers={"Authorization": f"Bearer {OWNER}"})
    assert resp.status_code == 200


def test_require_owner_non_ascii_token_is_401_not_500():
    client = _owner_app()
    token = "töken-é".encode("latin-1")
    resp = client.get("/owner-only", headers={"Authorization": b"Bearer " + token})
    assert resp.status_code == 401


def test_require_owner_fails_closed_when_server_token_unset(monkeypatch):
    monkeypatch.delenv("WITSV3_WEB_TOKEN")
    assert _owner_app().get("/owner-only").status_code == 401
    resp = _owner_app().get("/owner-only", headers={"Authorization": "Bearer "})
    assert resp.status_code == 401


def _write_module(tmp_path, monkeypatch, name: str, source: str) -> None:
    (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)


def test_missing_module_skipped_and_stub_register_called(monkeypatch, tmp_path):
    _use_settings(monkeypatch, tmp_path, enabled=True)
    _write_module(
        tmp_path,
        monkeypatch,
        "ws_stub_ok",
        "def register(app):\n    app.state.stub_registered = True\n",
    )
    monkeypatch.setattr(
        workspace_routes, "_OPTIONAL_MODULES", ("ws_stub_missing_xyz", "ws_stub_ok")
    )
    app = FastAPI()
    register_workspace(app)
    assert app.state.stub_registered is True


@pytest.mark.parametrize(
    "source,exc",
    [
        ("import ws_not_a_real_dependency_xyz\n", ModuleNotFoundError),
        ("raise ImportError('boom inside module')\n", ImportError),
    ],
)
def test_import_error_inside_existing_module_propagates(monkeypatch, tmp_path, source, exc):
    _use_settings(monkeypatch, tmp_path, enabled=True)
    _write_module(tmp_path, monkeypatch, "ws_stub_broken", source)
    monkeypatch.setattr(workspace_routes, "_OPTIONAL_MODULES", ("ws_stub_broken",))
    with pytest.raises(exc):
        register_workspace(FastAPI())


def test_lifespan_wraps_existing_and_runs_added_contexts(monkeypatch, tmp_path):
    db_path = _use_settings(monkeypatch, tmp_path, enabled=True)
    events: list = []

    @asynccontextmanager
    async def extra():
        events.append("extra-enter")
        yield
        events.append("extra-exit")

    _write_module(
        tmp_path,
        monkeypatch,
        "ws_stub_lifespan",
        "from web.workspace_routes import add_lifespan\n"
        "import ws_stub_hook\n"
        "def register(app):\n    add_lifespan(ws_stub_hook.factory)\n",
    )
    _write_module(tmp_path, monkeypatch, "ws_stub_hook", "factory = None\n")
    import ws_stub_hook  # noqa: E402  (stub written just above)

    ws_stub_hook.factory = extra
    monkeypatch.setattr(workspace_routes, "_OPTIONAL_MODULES", ("ws_stub_lifespan",))
    app = _app_with_startup_marker(events)
    register_workspace(app)
    with TestClient(app):
        assert events == ["app-start", "extra-enter"]
        assert get_db().path == db_path
        assert db_path.exists()
    assert events == ["app-start", "extra-enter", "extra-exit", "app-stop"]
    with pytest.raises(RuntimeError):
        get_db()


def test_add_lifespan_direct(monkeypatch, tmp_path):
    _use_settings(monkeypatch, tmp_path, enabled=True)
    events: list = []

    @asynccontextmanager
    async def extra():
        events.append("in")
        yield
        events.append("out")

    app = FastAPI()
    register_workspace(app)
    add_lifespan(extra)
    with TestClient(app):
        assert events == ["in"]
    assert events == ["in", "out"]
