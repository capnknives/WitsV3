"""Tests for folder access requests and grants (WP F3, folder access contract section 5)."""

import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import path_access
from core.tool_context import ToolContext, use_tool_context
from core.workspace import approvals, clients, folder_access
from core.workspace.db import open_db
from tools.folder_access_tool import FolderAccessRequestTool

DAY = 24 * 60 * 60
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(approvals, "_listeners", [])
    monkeypatch.setattr(path_access, "grants_provider", path_access._no_grants)
    yield
    folder_access.deactivate()


@pytest.fixture
async def db(tmp_path):
    database = await open_db(tmp_path / "workspace.db")
    yield database
    await database.close()


@pytest.fixture
def cfg(tmp_path):
    return SimpleNamespace(folders=SimpleNamespace(base=str(tmp_path / "base")), security=None)


@pytest.fixture
def outside(tmp_path):
    folder = tmp_path / "outside"
    folder.mkdir()
    (folder / "notes.txt").write_text("hi")
    return folder


async def _count(db, table):
    async with db.read().execute(f"SELECT COUNT(*) FROM {table}") as cur:
        return (await cur.fetchone())[0]


async def _audits(db, action):
    async with db.read().execute(
        "SELECT COUNT(*) FROM audit_log WHERE action = ?", (action,)
    ) as cur:
        return (await cur.fetchone())[0]


async def _request(db, cfg, path, mode="read", now=None):
    return await folder_access.request(db, str(path), mode, "need the notes", config=cfg, now=now)


async def test_never_path_creates_nothing(db, cfg, outside):
    (outside / ".env").write_text("SECRET=1")
    for path in (outside / ".env", outside / ".ssh", "/"):
        result = await _request(db, cfg, path)
        assert result["status"] == "refused" and result["code"] == "never"
    assert await _count(db, "approvals") == 0
    assert await _count(db, "folder_requests") == 0


@pytest.mark.parametrize("path", ["relative/dir", ".", "", "/tmp/a​b", 42])
async def test_bad_paths_are_refused_without_crashing(db, cfg, path):
    result = await folder_access.request(db, path, "read", "why", config=cfg)
    assert result["status"] == "refused"
    assert await _count(db, "approvals") == 0


async def test_unresolvable_path_is_a_refusal(db, cfg, monkeypatch):
    def boom(*a, **k):
        raise PermissionError("nope")

    monkeypatch.setattr(path_access, "roots_for", boom)
    result = await _request(db, cfg, "/tmp/x")
    assert result["status"] == "refused"


async def test_already_allowed_creates_nothing(db, cfg, tmp_path):
    for path, mode in ((tmp_path / "base" / "outbox", "read_write"), (REPO / "docs", "read")):
        assert (await _request(db, cfg, path, mode))["status"] == "already_allowed"
    # Reason: repo writes go through the verified-edit pipeline, so they are not requestable.
    assert (await _request(db, cfg, REPO / "docs", "read_write"))["code"] == "never"
    assert await _count(db, "approvals") == 0


async def test_request_creates_one_approval_and_row_and_repeat_returns_it(db, cfg, outside):
    first = await _request(db, cfg, outside)
    assert first["status"] == "pending" and first["existing"] is False
    again = await _request(db, cfg, outside)
    assert again["approval_id"] == first["approval_id"] and again["existing"] is True
    assert await _count(db, "approvals") == 1
    assert await _count(db, "folder_requests") == 1
    pending = await approvals.list_pending(db)
    assert pending[0]["title"] == f"Folder access: read {outside.resolve()}"
    assert pending[0]["options"] == ["approve", "one day", "deny"]
    assert pending[0]["client"] == "WITS" and pending[0]["body"] == "need the notes"
    assert await _audits(db, "folder.request") == 1


async def test_file_request_is_recorded_against_parent(db, cfg, outside):
    result = await _request(db, cfg, outside / "notes.txt")
    assert result["path"] == str(outside.resolve())


async def test_at_most_three_pending(db, cfg, tmp_path):
    for i in range(3):
        assert (await _request(db, cfg, tmp_path / f"d{i}"))["status"] == "pending"
    assert (await _request(db, cfg, tmp_path / "d9"))["code"] == "too_many_pending"


async def test_reason_is_required_and_cleaned(db, cfg, outside):
    empty = await folder_access.request(db, str(outside), "read", "​ ", config=cfg)
    assert empty["code"] == "invalid_argument"
    long = await folder_access.request(db, str(outside), "read", "x" * 8001, config=cfg)
    assert long["code"] == "invalid_argument"
    bad_mode = await folder_access.request(db, str(outside), "write", "why", config=cfg)
    assert bad_mode["code"] == "invalid_argument"
    await folder_access.request(db, str(outside), "read", "a‮b", config=cfg)
    assert (await approvals.list_pending(db))[0]["body"] == "ab"


@pytest.mark.parametrize("choice,days", [("approve", 30), ("one day", 1)])
async def test_approve_and_one_day_give_grants(db, cfg, outside, choice, days):
    req = await _request(db, cfg, outside, "read_write", now=1000)
    resolved = await approvals.resolve(db, req["approval_id"], choice, None, now=2000)
    assert resolved is not None
    grants = await folder_access.list_grants(db, now=2000)
    assert len(grants) == 1
    g = grants[0]
    assert (g["path"], g["mode"], g["created_at"]) == (str(outside.resolve()), "read_write", 2000)
    assert g["expires_at"] == 2000 + days * DAY and re.fullmatch(r"grt_[0-9a-f]{16}", g["id"])
    assert await _audits(db, "folder.grant") == 1
    # Reason: resolving twice is a 409 (None) and cannot create a second grant.
    assert await approvals.resolve(db, req["approval_id"], choice, None, now=2001) is None
    assert await _count(db, "folder_grants") == 1


async def test_deny_and_ordinary_approvals_give_nothing(db, cfg, outside):
    req = await _request(db, cfg, outside)
    await approvals.resolve(db, req["approval_id"], "deny", None)
    async with db.write() as conn:
        await conn.execute(
            "INSERT INTO clients (id, name, token_prefix, token_hash, scopes, created_at)"
            " VALUES ('cli_a', 'a', 'aaaaaaaa', 'h', 'decide:request', 0)"
        )
    other = await approvals.create(db, "cli_a", "p", None, "t", "", ["approve"], 3600, 0)
    await approvals.resolve(db, other["approval_id"], "approve", None)
    assert await _count(db, "folder_grants") == 0


async def test_grant_from_resolution_refuses_unresolved(db, cfg, outside):
    req = await _request(db, cfg, outside, now=1000)
    async with db.write() as conn:
        for choice in ("approve", "one day", "deny"):
            assert (
                await folder_access.grant_from_resolution(conn, req["approval_id"], choice, 1000)
                is None
            )
    await approvals.resolve(db, req["approval_id"], "one day", None, now=1500)
    async with db.write() as conn:
        assert (
            await folder_access.grant_from_resolution(conn, req["approval_id"], "approve", 1500)
            is None
        )
    assert await _count(db, "folder_grants") == 1


async def test_check_uses_live_grants_and_stops_on_revoke_or_expiry(db, cfg, outside):
    target = outside / "notes.txt"
    folder_access.activate(db)
    with pytest.raises(PermissionError):
        await path_access.check(target, "read", config=cfg)
    req = await _request(db, cfg, outside)
    await approvals.resolve(db, req["approval_id"], "approve", None)
    assert await path_access.check(target, "read", config=cfg) == target.resolve()
    with pytest.raises(PermissionError):
        await path_access.check(target, "write", config=cfg)
    grant_id = (await folder_access.list_grants(db))[0]["id"]
    assert await folder_access.revoke(db, grant_id) is True
    with pytest.raises(PermissionError):
        await path_access.check(target, "read", config=cfg)
    assert await folder_access.revoke(db, grant_id) is False
    assert await _audits(db, "folder.revoke") == 2

    old = int(time.time()) - 2 * DAY
    req = await _request(db, cfg, outside, now=old)
    await approvals.resolve(db, req["approval_id"], "one day", None, now=old + 10)
    assert await _count(db, "folder_grants") == 2
    with pytest.raises(PermissionError):
        await path_access.check(target, "read", config=cfg)


async def test_deactivate_restores_no_grants(db, cfg, outside):
    req = await _request(db, cfg, outside)
    await approvals.resolve(db, req["approval_id"], "approve", None)
    folder_access.activate(db)
    assert len(await path_access._load_grants()) == 1
    folder_access.deactivate()
    assert await path_access._load_grants() == ()
    assert folder_access.current_db() is None


async def test_internal_client_is_not_a_credential(db):
    async with db.read().execute(
        "SELECT token_prefix, token_hash FROM clients WHERE id = 'cli_internal_wits'"
    ) as cur:
        row = await cur.fetchone()
    assert not re.fullmatch(r"[0-9a-f]{64}", row["token_hash"])
    for token in ("internal", "x", "wits_internal_" + "A" * 43, "wits_00000000_" + "x" * 43):
        assert await clients.verify_token(db, token) is None
    assert all(c["id"] != "cli_internal_wits" for c in await clients.list_clients(db))
    assert await clients.revoke_client(db, "cli_internal_wits") is False


async def test_tool_is_owner_only_and_does_not_wait(db, cfg, outside, monkeypatch):
    monkeypatch.setattr("core.config.load_config", lambda *a, **k: _tool_config(cfg))
    tool = FolderAccessRequestTool()
    assert "workspace" in await tool.execute(str(outside), "read", "why")
    folder_access.activate(db)
    guest = ToolContext(session_id=None, user_role="guest", conversation_history=None, turn_id=None)
    with use_tool_context(guest):
        assert "only the owner" in await tool.execute(str(outside), "read", "why")
    assert "only the owner" in await tool.execute(str(outside), "read", "why", user_role="guest")
    assert await _count(db, "approvals") == 0
    text = await tool.execute(str(outside), "read", "why")
    assert "Inbox" in text and re.search(r"apr_[0-9a-f]{16}", text)
    assert "Error" in await tool.execute(str(outside / ".ssh"), "read", "why")
    assert "Error" in await tool.execute(str(outside), "read", "   ")


def _tool_config(cfg):
    return SimpleNamespace(
        folders=cfg.folders, security=None, workspace=SimpleNamespace(approval_ttl_minutes=60)
    )
