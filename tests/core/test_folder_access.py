"""Tests for folder access requests and grants (WP F3, folder access contract section 5)."""

import re
import shutil
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import path_access
from core.tool_context import ToolContext, use_tool_context
from core.workspace import approvals, clients, folder_access
from core.workspace import db as dbmod
from core.workspace.db import open_db
from tools.folder_access_tool import FolderAccessRequestTool

DAY = 24 * 60 * 60
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(approvals, "_listeners", [])
    monkeypatch.setattr(path_access, "grants_provider", path_access._no_grants)
    monkeypatch.setattr(folder_access, "_calls", folder_access.deque())
    monkeypatch.setattr(folder_access, "_provider_warned", False)
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


async def _outcomes(db):
    async with db.read().execute(
        "SELECT outcome FROM audit_log WHERE action = 'folder.request' ORDER BY id"
    ) as cur:
        return [r[0] for r in await cur.fetchall()]


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
    assert pending[0]["title"] == f"Folder access: read {outside}"
    assert pending[0]["options"] == ["approve", "one day", "deny"]
    assert pending[0]["client"] == "WITS" and pending[0]["body"] == "need the notes"
    assert await _outcomes(db) == ["ok", "duplicate"]


async def test_file_request_is_recorded_against_parent(db, cfg, outside):
    result = await _request(db, cfg, outside / "notes.txt")
    assert result["path"] == str(outside)


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
    assert (g["path"], g["mode"], g["created_at"]) == (str(outside), "read_write", 2000)
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


def _ctx(role):
    return ToolContext(session_id=None, user_role=role, conversation_history=None, turn_id=None)


async def test_tool_is_owner_only_fails_closed_and_does_not_wait(db, cfg, outside, monkeypatch):
    monkeypatch.setattr("core.config.load_config", lambda *a, **k: _tool_config(cfg))
    tool = FolderAccessRequestTool()
    with use_tool_context(_ctx("owner")):
        assert "workspace" in await tool.execute(str(outside), "read", "why")
    folder_access.activate(db)
    # Reason: no context means no trusted role, even when the argument claims owner.
    assert "only the owner" in await tool.execute(str(outside), "read", "why", user_role="owner")
    for role in ("guest", "", "Owner"):
        with use_tool_context(_ctx(role)):
            assert "only the owner" in await tool.execute(str(outside), "read", "why")
    assert await _count(db, "approvals") == 0
    with use_tool_context(_ctx("owner")):
        text = await tool.execute(str(outside), "read", "why")
        assert "Inbox" in text and re.search(r"apr_[0-9a-f]{16}", text)
        assert "Error" in await tool.execute(str(outside / ".ssh"), "read", "why")
        assert "Error" in await tool.execute(str(outside), "read", "   ")


def _tool_config(cfg):
    return SimpleNamespace(
        folders=cfg.folders, security=None, workspace=SimpleNamespace(approval_ttl_minutes=60)
    )


async def test_unc_and_never_paths_touch_no_filesystem(db, cfg, monkeypatch):
    seen = []
    for name in ("resolve", "is_file", "exists", "expanduser", "stat", "is_dir"):
        real = getattr(Path, name)

        def spy(self, *a, _real=real, **k):
            seen.append(str(self))
            return _real(self, *a, **k)

        monkeypatch.setattr(Path, name, spy)
    for path in ("\\\\host\\share\\x", "//host/share/x", "/srv/data/.env", "/srv/.ssh/k"):
        result = await _request(db, cfg, path)
        assert result["code"] == "never", path
    assert seen, "spy is not live"
    assert not [p for p in seen if "host" in p or "/srv" in p]


async def test_answers_do_not_reveal_whether_paths_exist(db, cfg, outside, tmp_path):
    real = await _request(db, cfg, outside / "notes.txt")
    ghost = await _request(db, cfg, tmp_path / "ghost" / "nothing.txt")
    assert real.keys() == ghost.keys() and real["status"] == ghost["status"] == "pending"
    assert ghost["path"] == str(tmp_path / "ghost")
    # Reason: no suffix means a folder, existing or not; nothing is resolved or expanded.
    assert (await _request(db, cfg, f"{tmp_path}/a/../noext"))["path"] == f"{tmp_path}/noext"
    # Reason: with the queue full, a repeat and a new path get the same answer.
    assert (await _request(db, cfg, outside / "notes.txt"))["code"] == "too_many_pending"
    assert (await _request(db, cfg, tmp_path / "other"))["code"] == "too_many_pending"


async def test_every_call_is_audited_and_rate_limited(db, cfg, outside, tmp_path):
    await _request(db, cfg, "/srv/.ssh", now=100)
    await _request(db, cfg, tmp_path / "base" / "inbox", now=100)
    await folder_access.request(db, "rel", "read", "why", config=cfg, now=100)
    for _ in range(7):
        await _request(db, cfg, outside, now=100)
    over = await _request(db, cfg, outside, now=200)
    assert over["code"] == "rate_limited" and "ask the owner" in over["message"]
    outcomes = await _outcomes(db)
    assert outcomes[:4] == ["refused", "already_allowed", "refused", "ok"]
    assert outcomes[4:] == ["duplicate"] * 6 + ["refused"]
    assert (await _request(db, cfg, outside, now=100 + 3601))["status"] == "pending"


async def test_spelling_variants_share_one_request(db, cfg, outside):
    first = await _request(db, cfg, outside)
    for variant in (f"{outside}/", f"{outside}/./", str(outside).upper()):
        assert (await _request(db, cfg, variant))["approval_id"] == first["approval_id"]
    assert await _count(db, "folder_requests") == 1


async def test_deny_starts_a_one_day_cooldown(db, cfg, outside):
    req = await _request(db, cfg, outside, now=1000)
    await approvals.resolve(db, req["approval_id"], "deny", None, now=1100)
    assert (await _request(db, cfg, f"{outside}/", now=1200))["code"] == "cooldown"
    assert (await _request(db, cfg, outside, "index", now=1200))["status"] == "pending"
    assert (await _request(db, cfg, outside, now=1100 + DAY + 1))["status"] == "pending"


async def test_expired_request_cannot_be_granted(db, cfg, outside):
    req = await _request(db, cfg, outside, now=1000)
    assert await approvals.resolve(db, req["approval_id"], "approve", None, now=1000 + DAY) is None
    assert await _count(db, "folder_grants") == 0


async def test_never_children_stay_refused_under_a_grant(db, cfg, outside):
    folder_access.activate(db)
    req = await _request(db, cfg, outside, "read_write")
    await approvals.resolve(db, req["approval_id"], "approve", None)
    (outside / ".git").mkdir()
    assert await path_access.check(outside / "notes.txt", "write", config=cfg)
    for target, mode in ((outside / ".env", "read"), (outside / ".git" / "config", "write")):
        with pytest.raises(PermissionError):
            await path_access.check(target, mode, config=cfg)


async def test_grant_rows_only_change_by_revoking_once(db, cfg, outside):
    req = await _request(db, cfg, outside)
    await approvals.resolve(db, req["approval_id"], "approve", None)
    for sql in (
        "UPDATE folder_grants SET path = '/'",
        "UPDATE folder_grants SET expires_at = expires_at + 1",
        "UPDATE folder_grants SET revoked_at = 1, mode = 'read_write'",
        "DELETE FROM folder_grants",
        "INSERT OR REPLACE INTO folder_grants SELECT * FROM folder_grants",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            async with db.write() as conn:
                await conn.execute(sql)
    grant_id = (await folder_access.list_grants(db))[0]["id"]
    assert await folder_access.revoke(db, grant_id) is True
    with pytest.raises(sqlite3.IntegrityError):
        async with db.write() as conn:
            await conn.execute("UPDATE folder_grants SET revoked_at = NULL")


async def test_provider_fails_closed_and_logs_once(tmp_path, cfg, outside, caplog):
    database = await open_db(tmp_path / "closed.db")
    req = await _request(database, cfg, outside)
    await approvals.resolve(database, req["approval_id"], "approve", None)
    folder_access.activate(database)
    assert len(await path_access.grants_provider()) == 1
    await database.close()
    with caplog.at_level("WARNING", logger=folder_access.logger.name):
        assert await path_access.grants_provider() == ()
        assert await path_access.grants_provider() == ()
    assert sum("unreadable" in r.message for r in caplog.records) == 1


async def test_migration_survives_an_existing_wits_client(tmp_path, monkeypatch):
    mig = tmp_path / "mig"
    mig.mkdir()
    shutil.copy(dbmod.MIGRATIONS_DIR / "0001_init.sql", mig)
    real_dir = dbmod.MIGRATIONS_DIR
    monkeypatch.setattr(dbmod, "MIGRATIONS_DIR", mig)
    path = tmp_path / "old.db"
    old = await open_db(path)
    async with old.write() as conn:
        await conn.execute(
            "INSERT INTO clients (id, name, token_prefix, token_hash, scopes, created_at)"
            " VALUES ('cli_x', 'WITS', 'abcd1234', 'h', 'read', 0)"
        )
    await old.close()
    monkeypatch.setattr(dbmod, "MIGRATIONS_DIR", real_dir)
    upgraded = await open_db(path)
    async with upgraded.read().execute(
        "SELECT name, token_prefix FROM clients WHERE id = 'cli_internal_wits'"
    ) as cur:
        assert tuple(await cur.fetchone()) == ("WITS (internal_wits)", "internal")
    await upgraded.close()
