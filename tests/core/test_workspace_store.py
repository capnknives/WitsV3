"""Tests for the workspace store (WP 1.1a)."""

import asyncio
import shutil
import sqlite3

import pytest

from core.workspace import db as dbmod
from core.workspace import store
from core.workspace.db import open_db
from core.workspace.store import WorkspaceError
from core.workspace.text import clean_text

T0 = 1_000_000


@pytest.fixture
async def db(tmp_path):
    database = await open_db(tmp_path / "workspace.db")
    async with database.write() as conn:
        for cid, name in (("cli_a", "alice"), ("cli_b", "bob")):
            await conn.execute(
                "INSERT INTO clients (id, name, token_prefix, token_hash, scopes, created_at)"
                " VALUES (?, ?, ?, 'h', 'read write', 0)",
                (cid, name, cid),
            )
    yield database
    await database.close()


async def _scalar(database, sql, *args):
    async with database.read().execute(sql, args) as cur:
        return (await cur.fetchone())[0]


async def _new_task(database, client="cli_a", project="proj", now=T0, lease=600):
    return await store.claim_task(database, client, project, None, "t", "b", lease, 0, now=now)


async def expect(code, coro):
    with pytest.raises(WorkspaceError) as exc:
        await coro
    assert exc.value.code == code
    return exc.value


class TestMigrations:
    async def test_fresh_then_reopen(self, tmp_path):
        path = tmp_path / "w.db"
        first = await open_db(path)
        assert await _scalar(first, "SELECT MAX(version) FROM schema_version") == 1
        await first.close()
        second = await open_db(path)
        assert await _scalar(second, "SELECT COUNT(*) FROM schema_version") == 1
        await second.close()
        assert list(tmp_path.glob("*.bak")) == []

    async def test_failed_migration_leaves_no_objects(self, tmp_path, monkeypatch):
        mig = tmp_path / "mig"
        mig.mkdir()
        shutil.copy(dbmod.MIGRATIONS_DIR / "0001_init.sql", mig)
        (mig / "0002_bad.sql").write_text(
            "CREATE TABLE ok_table (x INTEGER);\nCREATE TABLE ok_table (y INTEGER);\n"
        )
        monkeypatch.setattr(dbmod, "MIGRATIONS_DIR", mig)
        path = tmp_path / "w.db"
        with pytest.raises(sqlite3.OperationalError):
            await open_db(path)
        con = sqlite3.connect(path)
        assert con.execute("SELECT name FROM sqlite_master WHERE name='ok_table'").fetchall() == []
        assert con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1
        con.close()

    async def test_backup_only_when_migrating_existing(self, tmp_path, monkeypatch):
        path = tmp_path / "w.db"
        await (await open_db(path)).close()
        assert list(tmp_path.glob("*.bak")) == []
        mig = tmp_path / "mig"
        mig.mkdir()
        shutil.copy(dbmod.MIGRATIONS_DIR / "0001_init.sql", mig)
        (mig / "0002_extra.sql").write_text("CREATE TABLE extra (x INTEGER);\n")
        monkeypatch.setattr(dbmod, "MIGRATIONS_DIR", mig)
        upgraded = await open_db(path)
        assert await _scalar(upgraded, "SELECT MAX(version) FROM schema_version") == 2
        await upgraded.close()
        backups = list(tmp_path.glob("w.db.*.bak"))
        assert len(backups) == 1
        con = sqlite3.connect(backups[0])
        assert con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1
        con.close()


class TestIntegrity:
    @pytest.mark.parametrize("table", ["handoffs", "audit_log"])
    async def test_append_only(self, db, table):
        await store.audit(db, "owner", "x", None, "ok", now=T0)
        await store.append_handoff(db, "cli_a", "proj", None, "note", "hi", 0, now=T0)
        cols = "(id, project, client_id, kind, body, created_at)"
        replace = (
            f"INSERT OR REPLACE INTO handoffs {cols} VALUES (1, 'p', 'cli_a', 'note', 'z', 0)"
            if table == "handoffs"
            else "INSERT OR REPLACE INTO audit_log (id, ts, actor, action, outcome)"
            " VALUES (1, 0, 'a', 'b', 'c')"
        )
        for sql in (f"UPDATE {table} SET id = id", f"DELETE FROM {table}", replace):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                async with db.write() as conn:
                    await conn.execute(sql)
        assert await _scalar(db, f"SELECT COUNT(*) FROM {table}") == 1

    async def test_foreign_keys_enforced(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            async with db.write() as conn:
                await conn.execute(
                    "INSERT INTO tasks (id, project, title, created_by, created_at, updated_at)"
                    " VALUES ('tsk_x', 'p', 't', 'cli_missing', 0, 0)"
                )


class TestClaim:
    async def test_create_and_claim(self, db):
        res = await _new_task(db)
        assert res["created"] is True and res["lease_expires_at"] == T0 + 600
        assert await _scalar(db, "SELECT COUNT(*) FROM task_locks") == 1

    async def test_failure_after_task_insert_rolls_back(self, db):
        # Reason: SQLite stores NaN as NULL, so the lock insert violates NOT NULL after the task
        # insert has already succeeded in the same transaction.
        with pytest.raises(sqlite3.IntegrityError):
            await store.claim_task(db, "cli_a", "proj", None, "t", "b", float("nan"), 0, now=T0)
        assert await _scalar(db, "SELECT COUNT(*) FROM tasks") == 0
        assert await _scalar(db, "SELECT COUNT(*) FROM task_locks") == 0

    async def test_lock_lifecycle(self, db):
        tid = (await _new_task(db))["task_id"]
        err = await expect(
            "task_locked", store.claim_task(db, "cli_b", "proj", tid, None, "", 600, 0, now=T0 + 10)
        )
        assert err.extra == {"holder": "alice", "expires_at": T0 + 600}
        renewed = await store.claim_task(db, "cli_a", "proj", tid, None, "", 900, 0, now=T0 + 100)
        assert renewed["created"] is False and renewed["lease_expires_at"] == T0 + 1000
        assert await _scalar(db, "SELECT claimed_at FROM task_locks") == T0
        takeover = await store.claim_task(db, "cli_b", "proj", tid, None, "", 600, 0, now=T0 + 1000)
        assert takeover["lease_expires_at"] == T0 + 1600
        assert await _scalar(db, "SELECT client_id FROM task_locks") == "cli_b"
        assert await _scalar(db, "SELECT claimed_at FROM task_locks") == T0 + 1000

    async def test_lapsed_holder_cannot_finish(self, db):
        tid = (await _new_task(db))["task_id"]
        await expect(
            "not_lock_holder",
            store.append_handoff(db, "cli_a", "proj", tid, "done", "x", 0, now=T0 + 600),
        )

    async def test_needs_exactly_one_of_id_or_title(self, db):
        await expect(
            "invalid_argument", store.claim_task(db, "cli_a", "proj", None, None, "", 60, 0)
        )


class TestHandoffs:
    async def test_note_keeps_lock_and_never_shortens(self, db):
        tid = (await _new_task(db, lease=600))["task_id"]
        res = await store.append_handoff(db, "cli_a", "proj", tid, "note", "n", 0, now=T0 + 500)
        assert res["lock_released"] is False
        assert await _scalar(db, "SELECT expires_at FROM task_locks") == T0 + 1100
        await store.append_handoff(db, "cli_a", "proj", tid, "note", "n", 0, now=T0 + 100)
        assert await _scalar(db, "SELECT expires_at FROM task_locks") == T0 + 1100
        await store.append_handoff(db, "cli_b", "proj", tid, "note", "other", 0, now=T0 + 900)
        assert await _scalar(db, "SELECT expires_at FROM task_locks") == T0 + 1100

    @pytest.mark.parametrize("kind", ["handoff", "blocked", "done"])
    async def test_releasing_kinds(self, db, kind):
        tid = (await _new_task(db))["task_id"]
        res = await store.append_handoff(db, "cli_a", "proj", tid, kind, "x", 0, now=T0 + 1)
        assert res["lock_released"] is True
        assert await _scalar(db, "SELECT COUNT(*) FROM task_locks") == 0
        status = await _scalar(db, "SELECT status FROM tasks WHERE id = ?", tid)
        assert status == ("done" if kind == "done" else "open")

    async def test_closed_task(self, db):
        tid = (await _new_task(db))["task_id"]
        await store.append_handoff(db, "cli_a", "proj", tid, "done", "x", 0, now=T0 + 1)
        await expect("task_closed", store.append_handoff(db, "cli_a", "proj", tid, "note", "x", 0))
        await expect("task_closed", store.claim_task(db, "cli_a", "proj", tid, None, "", 60, 0))

    async def test_non_holder_cannot_release(self, db):
        tid = (await _new_task(db))["task_id"]
        await expect(
            "not_lock_holder",
            store.append_handoff(db, "cli_b", "proj", tid, "handoff", "x", 0, now=T0 + 1),
        )


class TestProjectBinding:
    async def test_other_project_is_not_found(self, db):
        tid = (await _new_task(db, project="proj"))["task_id"]
        await expect("not_found", store.claim_task(db, "cli_a", "other", tid, None, "", 60, 0))
        await expect("not_found", store.append_handoff(db, "cli_a", "other", tid, "note", "x", 0))
        await expect(
            "not_found", store.link_artifact(db, "cli_a", "other", tid, "url", "https://x/y", "")
        )
        assert await _scalar(db, "SELECT COUNT(*) FROM handoffs") == 0
        assert await _scalar(db, "SELECT COUNT(*) FROM artifacts") == 0


class TestArtifacts:
    async def test_repeat_uri_is_idempotent(self, db):
        a = await store.link_artifact(db, "cli_a", "proj", None, "url", "https://x/y", "t", now=T0)
        b = await store.link_artifact(db, "cli_b", "proj", None, "url", "https://x/y", "u", now=T0)
        assert a["created"] is True and b["created"] is False
        assert a["artifact_id"] == b["artifact_id"]
        assert await _scalar(db, "SELECT COUNT(*) FROM artifacts") == 1


class TestProjectState:
    async def test_caps_ordering_and_names(self, db):
        for i in range(52):
            await store.claim_task(db, "cli_a", "proj", None, f"t{i}", "", 600, 0, now=T0 + i)
        for i in range(5):
            await store.append_handoff(db, "cli_b", "proj", None, "note", f"h{i}", 0, now=T0 + i)
        await store.append_handoff(db, "cli_b", "other", None, "note", "elsewhere", 0, now=T0)
        for i in range(51):
            uri = f"docs/{i}.md"
            await store.link_artifact(db, "cli_a", "proj", None, "file", uri, "", now=T0 + i)
        state = await store.project_state(db, "proj", 0, 3, now=T0 + 10)
        assert [t["title"] for t in state["tasks"]][:2] == ["t51", "t50"]
        assert len(state["tasks"]) == 50
        assert state["tasks"][0]["lock"] == {"client": "alice", "expires_at": T0 + 51 + 600}
        assert [h["body"] for h in state["handoffs"]] == ["h2", "h3", "h4"]
        assert state["handoffs"][0]["client"] == "bob" and "client_id" not in state["handoffs"][0]
        assert len(state["artifacts"]) == 50 and state["artifacts"][0]["uri"] == "docs/50.md"
        assert state["truncated"] is True and "approvals" not in state

    async def test_since_and_untruncated(self, db):
        for i in range(3):
            await store.append_handoff(db, "cli_a", "proj", None, "note", f"h{i}", 0, now=T0)
        state = await store.project_state(db, "proj", 1, 20, now=T0)
        assert [h["id"] for h in state["handoffs"]] == [2, 3]
        assert state["truncated"] is False

    async def test_expired_lock_hidden_and_closed_tasks_excluded(self, db):
        tid = (await _new_task(db))["task_id"]
        done = (await _new_task(db))["task_id"]
        await store.append_handoff(db, "cli_a", "proj", done, "done", "x", 0, now=T0 + 1)
        state = await store.project_state(db, "proj", 0, 20, now=T0 + 600)
        assert [t["id"] for t in state["tasks"]] == [tid] and state["tasks"][0]["lock"] is None


class TestWriters:
    async def test_concurrent_writes_all_land(self, db, tmp_path):
        other = await open_db(db.path)

        async def add(database, n):
            for _ in range(n):
                async with database.write() as conn:
                    await conn.execute(
                        "INSERT INTO audit_log (ts, actor, action, outcome)"
                        " VALUES (0, 'a', 'x', 'ok')"
                    )
                    await asyncio.sleep(0)

        await asyncio.gather(add(db, 25), add(db, 25), add(other, 25))
        assert await _scalar(db, "SELECT COUNT(*) FROM audit_log") == 75
        await other.close()


class TestAudit:
    async def test_appends_one_row(self, db):
        await store.audit(db, "cli_a", "claim_task", "tsk_1", "ok", "d", now=T0)
        assert await _scalar(db, "SELECT COUNT(*) FROM audit_log") == 1
        assert await _scalar(db, "SELECT detail FROM audit_log") == "d"


class TestCleanText:
    def test_drops_invisible_keeps_whitespace(self):
        raw = "a\u200bb\u202ec\U000e0041d\U000e0100e\u2028f\x00g\x07h\ni\tj"
        assert clean_text(raw, 100) == "abcdefgh\ni\tj"

    def test_keeps_emoji_presentation_selector(self):
        # Reason: the contract lists only U+E0100 to U+E01EF; emoji need U+FE0F.
        assert clean_text("\u2764\ufe0f", 10) == "\u2764\ufe0f"

    def test_rejects_over_limit_without_truncating(self):
        assert clean_text("x" * 10, 10) == "x" * 10
        with pytest.raises(WorkspaceError) as exc:
            clean_text("x" * 11, 10)
        assert exc.value.code == "invalid_argument"

    def test_length_counts_after_cleaning(self):
        assert clean_text("x​" * 10, 10) == "x" * 10

    def test_required_and_optional(self):
        with pytest.raises(WorkspaceError):
            clean_text("​  ", 10)
        assert clean_text("​", 10, required=False) == ""
