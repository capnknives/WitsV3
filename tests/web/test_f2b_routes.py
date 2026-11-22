"""WP F2b: export, document upload and delete meet the folder-access check."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.web.test_web_server import FakeSystem, _parse_sse
from web.server import create_app


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "owner-sekrit")
    monkeypatch.setenv("WITSV3_GUEST_INVITE", "family-code")
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "guest-signing-secret-xyz")
    monkeypatch.delenv("WITSV3_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var" / "data").mkdir(parents=True)
    system = FakeSystem(tmp_path)
    system.config.web_ui.guest_access.enabled = True
    system.config.web_ui.guest_access.allow_lan_only = True
    return TestClient(create_app(system)), tmp_path


OWNER = {"Authorization": "Bearer owner-sekrit"}


def _guest_headers(client):
    reg = client.post(
        "/api/guest/register",
        json={
            "invite_code": "family-code",
            "display_name": "Alex",
            "device_id": "device-abc-12345",
            "age_band": "teen",
        },
    )
    return {"Authorization": f"Bearer {reg.json()['guest_token']}"}


def _session(client, headers):
    res = client.post("/api/chat", json={"message": "hello"}, headers=headers)
    return _parse_sse(res.text)[0][1]["session_id"]


def _all_files(root):
    return sorted(p.name for p in root.rglob("*") if p.is_file())


def test_owner_export_lands_in_outbox(env):
    client, tmp = env
    res = client.post("/api/export", json={"session_id": _session(client, OWNER)}, headers=OWNER)
    assert res.status_code == 200
    out = tmp / "wits" / "outbox"
    assert res.json()["file_path"].startswith(str(out).replace("\\", "/"))
    assert len(list(out.iterdir())) == 1
    assert not (tmp / "var" / "exports").exists()


NEVER = [".env", "secrets.txt", "client_secret_x.json", "id_rsa"]
ALIASES = [".env.", ".env ", "notes.txt:secret"]
WHO = ["owner", "guest"]


def _headers(client, who):
    return OWNER if who == "owner" else _guest_headers(client)


@pytest.mark.parametrize("who", WHO)
@pytest.mark.parametrize("name", NEVER)
def test_never_listed_export_name_refused(env, who, name):
    client, tmp = env
    headers = _headers(client, who)
    sid = _session(client, headers)
    before = _all_files(tmp)
    res = client.post("/api/export", json={"session_id": sid, "file_path": name}, headers=headers)
    assert res.status_code == 403
    assert _all_files(tmp) == before


def test_guest_export_works_without_absolute_path(env):
    client, tmp = env
    headers = _guest_headers(client)
    sid = _session(client, headers)
    res = client.post("/api/export", json={"session_id": sid}, headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert "/" not in body["file_path"] and str(tmp) not in body["message"]
    assert (tmp / "var" / "exports" / body["file_path"]).is_file()


@pytest.mark.parametrize("who", WHO)
def test_export_traversal_still_refused(env, who):
    client, tmp = env
    headers = _headers(client, who)
    sid = _session(client, headers)
    before = _all_files(tmp)
    res = client.post(
        "/api/export", json={"session_id": sid, "file_path": "../../evil.txt"}, headers=headers
    )
    assert res.status_code == 400
    assert _all_files(tmp) == before


def test_guest_export_symlink_cannot_redirect_write(env):
    client, tmp = env
    headers = _guest_headers(client)
    sid = _session(client, headers)
    exports = tmp / "var" / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    outside = tmp / "outside.txt"
    try:
        (exports / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    res = client.post(
        "/api/export", json={"session_id": sid, "file_path": "link.txt"}, headers=headers
    )
    assert res.status_code == 403
    assert not outside.exists()


@pytest.mark.parametrize("name", NEVER + ALIASES)
def test_upload_of_secret_names_refused(env, name):
    client, tmp = env
    before = _all_files(tmp)
    res = client.post(
        "/api/documents/upload", files={"file": (name, b"x", "text/plain")}, headers=OWNER
    )
    assert res.status_code in (400, 403)
    assert res.status_code == 403 or name in ALIASES
    assert _all_files(tmp) == before


def test_upload_of_plain_file_works(env):
    client, tmp = env
    res = client.post(
        "/api/documents/upload", files={"file": ("notes.txt", b"hi", "text/plain")}, headers=OWNER
    )
    assert res.status_code == 200 and res.json()["saved"] == "notes.txt"
    assert (tmp / "documents" / "notes.txt").read_bytes() == b"hi"


@pytest.mark.parametrize("name", NEVER + ALIASES)
def test_delete_of_never_listed_name_refused(env, name):
    client, tmp = env
    docs = tmp / "documents"
    docs.mkdir()
    (docs / name).write_text("KEY=1")
    res = client.post("/api/documents/delete", json={"name": name}, headers=OWNER)
    assert res.status_code in (400, 403)
    assert res.status_code == 403 or name in ALIASES
    assert (docs / name).read_text() == "KEY=1"


def test_delete_outside_folder_still_refused(env):
    client, tmp = env
    (tmp / "documents").mkdir()
    (tmp / "keep.txt").write_text("x")
    res = client.post("/api/documents/delete", json={"name": "../keep.txt"}, headers=OWNER)
    assert res.status_code == 400
    assert (tmp / "keep.txt").exists()
