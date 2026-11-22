"""Tests for the workspace approval queue (WP 1.3)."""

import asyncio
import json
import logging

import pytest

from core.workspace import approvals
from core.workspace.db import open_db
from core.workspace.store import WorkspaceError

T0 = 1_000_000
TTL = 3600
DAY = 24 * 60 * 60


async def _add_clients(database):
    # Reason: clients.py belongs to WP 1.2, built in parallel, so rows are inserted directly.
    async with database.write() as conn:
        for cid, name in (("cli_a", "alice"), ("cli_b", "bob")):
            await conn.execute(
                "INSERT INTO clients (id, name, token_prefix, token_hash, scopes, created_at)"
                " VALUES (?, ?, ?, 'h', 'decide:request', 0)",
                (cid, name, cid),
            )


@pytest.fixture
async def db(tmp_path):
    database = await open_db(tmp_path / "workspace.db")
    await _add_clients(database)
    yield database
    await database.close()


@pytest.fixture(autouse=True)
def isolated_listeners(monkeypatch):
    monkeypatch.setattr(approvals, "_listeners", [])


async def _row(database, approval_id):
    async with database.read().execute(
        "SELECT * FROM approvals WHERE id = ?", (approval_id,)
    ) as cur:
        return await cur.fetchone()


async def _scalar(database, sql, *args):
    async with database.read().execute(sql, args) as cur:
        return (await cur.fetchone())[0]


async def _create(database, client="cli_a", project="proj", options=None, now=T0, ttl=TTL, **kw):
    return await approvals.create(
        database,
        client,
        project,
        kw.get("task_id"),
        kw.get("title", "Ship it?"),
        "body",
        options,
        ttl,
        0,
        now=now,
    )


async def expect(code, coro):
    with pytest.raises(WorkspaceError) as exc:
        await coro
    assert exc.value.code == code
    return exc.value


class TestCreate:
    async def test_pending_with_expiry_and_default_options(self, db):
        result = await _create(db)
        assert result["status"] == "pending"
        assert result["expires_at"] == T0 + TTL
        assert result["approval_id"].startswith("apr_")
        row = await _row(db, result["approval_id"])
        assert row["status"] == "pending"
        assert json.loads(row["options"]) == ["approve", "deny"]

    async def test_custom_option_kept_and_deny_appended(self, db):
        result = await _create(db, options=["  ship tonight ", "wait"])
        row = await _row(db, result["approval_id"])
        assert json.loads(row["options"]) == ["ship tonight", "wait", "deny"]

    async def test_deny_not_duplicated(self, db):
        result = await _create(db, options=["deny", "approve"])
        assert json.loads((await _row(db, result["approval_id"]))["options"]) == [
            "deny",
            "approve",
        ]

    @pytest.mark.parametrize(
        "options",
        [
            ["Yes", "yes"],
            ["ship", "SHIP"],
            ["Ａ", "a"],  # fullwidth A folds to a under NFKC
            ["Straße", "strasse"],  # casefold of sharp s
            ["Approve"],
            ["DENY"],
            ["ａpprove"],  # fullwidth a: folds to approve but is not exactly it
            [],
            ["a", "b", "c", "d"],
            [""],
            ["   "],
            ["x" * 41],
            "approve",
            [1],
        ],
    )
    async def test_bad_options_refused(self, db, options):
        await expect("invalid_argument", _create(db, options=options))
        assert await _scalar(db, "SELECT COUNT(*) FROM approvals") == 0

    @pytest.mark.parametrize("ttl", [0, 299, 10080 * 60 + 1, True, 60.5])
    async def test_bad_ttl_refused(self, db, ttl):
        await expect("invalid_argument", _create(db, ttl=ttl))

    async def test_task_from_other_project_not_found(self, db):
        async with db.write() as conn:
            await conn.execute(
                "INSERT INTO tasks (id, project, title, created_by, created_at, updated_at)"
                " VALUES ('tsk_x', 'other', 't', 'cli_a', 0, 0)"
            )
        await expect("not_found", _create(db, task_id="tsk_x"))
        await expect("not_found", _create(db, task_id="tsk_missing"))
        ok = await _create(db, project="other", task_id="tsk_x")
        assert (await _row(db, ok["approval_id"]))["task_id"] == "tsk_x"

    async def test_unknown_client_not_found(self, db):
        await expect("not_found", _create(db, client="cli_nobody"))


class TestLimits:
    async def test_sixth_live_pending_refused(self, db):
        for i in range(5):
            await _create(db, now=T0 + i)
        await expect("too_many_pending", _create(db, now=T0 + 10))
        # Another client is unaffected.
        await _create(db, client="cli_b", now=T0 + 10)

    async def test_expired_frees_a_slot(self, db):
        await _create(db, ttl=300)
        for i in range(4):
            await _create(db, now=T0 + i)
        await expect("too_many_pending", _create(db, now=T0 + 10))
        await _create(db, now=T0 + 300)

    async def test_resolved_frees_a_slot(self, db):
        ids = [(await _create(db, now=T0 + i))["approval_id"] for i in range(5)]
        await approvals.resolve(db, ids[0], "deny", None, now=T0 + 10)
        await _create(db, now=T0 + 11)

    async def test_twenty_first_in_24_hours_refused(self, db):
        for i in range(20):
            created = await _create(db, now=T0 + i)
            await approvals.resolve(db, created["approval_id"], "approve", None, now=T0 + i)
        await expect("too_many_pending", _create(db, now=T0 + 100))
        # The window rolls: the first creation drops out after 24 hours.
        await _create(db, now=T0 + DAY)

    async def test_racing_creates_at_limit_same_db(self, db):
        for i in range(4):
            await _create(db, now=T0 + i)
        results = await asyncio.gather(
            _create(db, now=T0 + 10), _create(db, now=T0 + 10), return_exceptions=True
        )
        assert sum(isinstance(r, dict) for r in results) == 1
        assert sum(isinstance(r, WorkspaceError) for r in results) == 1
        assert await _scalar(db, "SELECT COUNT(*) FROM approvals") == 5

    async def test_racing_creates_across_connections(self, db, tmp_path):
        # Reason: two Database objects have separate locks, so only BEGIN IMMEDIATE serializes.
        other = await open_db(tmp_path / "workspace.db")
        try:
            for i in range(4):
                await _create(db, now=T0 + i)
            results = await asyncio.gather(
                _create(db, now=T0 + 10), _create(other, now=T0 + 10), return_exceptions=True
            )
        finally:
            await other.close()
        assert sum(isinstance(r, dict) for r in results) == 1
        errors = [r for r in results if isinstance(r, WorkspaceError)]
        assert [e.code for e in errors] == ["too_many_pending"]
        assert await _scalar(db, "SELECT COUNT(*) FROM approvals") == 5


class TestResolve:
    @pytest.mark.parametrize(
        "choice,status", [("approve", "approved"), ("deny", "denied"), ("later", "answered")]
    )
    async def test_status_from_choice(self, db, choice, status):
        aid = (await _create(db, options=["approve", "later"]))["approval_id"]
        result = await approvals.resolve(db, aid, choice, "ok", now=T0 + 5)
        assert result["status"] == status
        assert result["choice"] == choice
        row = await _row(db, aid)
        assert (row["status"], row["choice"], row["note"]) == (status, choice, "ok")
        assert (row["resolved_at"], row["resolved_by"]) == (T0 + 5, "owner")

    async def test_choice_outside_options_refused(self, db):
        aid = (await _create(db))["approval_id"]
        await expect("invalid_argument", approvals.resolve(db, aid, "Approve", None, now=T0))
        await expect("invalid_argument", approvals.resolve(db, aid, "maybe", None, now=T0))
        assert (await _row(db, aid))["status"] == "pending"

    async def test_second_resolve_returns_none(self, db):
        aid = (await _create(db))["approval_id"]
        await approvals.resolve(db, aid, "approve", "first", now=T0 + 1)
        before = dict(await _row(db, aid))
        assert await approvals.resolve(db, aid, "deny", "second", now=T0 + 2) is None
        assert dict(await _row(db, aid)) == before

    async def test_expired_returns_none(self, db):
        aid = (await _create(db))["approval_id"]
        before = dict(await _row(db, aid))
        assert await approvals.resolve(db, aid, "approve", None, now=T0 + TTL) is None
        assert dict(await _row(db, aid)) == before

    async def test_cancelled_returns_none(self, db):
        aid = (await _create(db))["approval_id"]
        async with db.write() as conn:
            await conn.execute(
                "UPDATE approvals SET status = 'cancelled', resolved_at = ? WHERE id = ?",
                (T0 + 1, aid),
            )
        before = dict(await _row(db, aid))
        assert await approvals.resolve(db, aid, "approve", None, now=T0 + 2) is None
        assert dict(await _row(db, aid)) == before

    async def test_missing_returns_none(self, db):
        assert await approvals.resolve(db, "apr_missing", "approve", None, now=T0) is None


class TestLists:
    async def test_list_for_project(self, db):
        live = (await _create(db, title="live"))["approval_id"]
        expired = (await _create(db, title="expired", ttl=300))["approval_id"]
        done = (await _create(db, client="cli_b", title="done"))["approval_id"]
        old = (await _create(db, title="old", now=T0 - 8 * DAY))["approval_id"]
        await approvals.resolve(db, old, "deny", None, now=T0 - 8 * DAY + 1)
        await _create(db, project="elsewhere", title="other")
        now = T0 + 600
        await approvals.resolve(db, done, "approve", "go", now=now)
        items = {i["id"]: i for i in await approvals.list_for_project(db, "proj", now=now)}
        assert set(items) == {live, expired, done}
        assert items[live]["status"] == "pending"
        assert items[expired]["status"] == "expired"
        assert items[done] == {
            "id": done,
            "client": "bob",
            "title": "done",
            "status": "approved",
            "choice": "approve",
            "note": "go",
            "expires_at": T0 + TTL,
        }
        assert items[live]["client"] == "alice"
        # Expired pending drops out seven days after its expiry.
        later = await approvals.list_for_project(db, "proj", now=T0 + 300 + 7 * DAY)
        assert expired not in {i["id"] for i in later}

    async def test_list_pending_live_only(self, db):
        live = (await _create(db, options=["ship"]))["approval_id"]
        await _create(db, ttl=300)
        resolved = (await _create(db))["approval_id"]
        await approvals.resolve(db, resolved, "approve", None, now=T0 + 1)
        other = (await _create(db, client="cli_b", project="elsewhere", now=T0 + 2))["approval_id"]
        items = await approvals.list_pending(db, now=T0 + 300)
        assert [i["id"] for i in items] == [live, other]
        assert items[0]["client"] == "alice"
        assert items[0]["options"] == ["ship", "deny"]
        assert items[1]["project"] == "elsewhere"


class TestListener:
    async def test_called_once_after_commit(self, db):
        calls = []

        async def cb(approval_id, client_name):
            # The row is committed and the writer lock is free when the listener runs.
            assert not db._lock.locked()
            calls.append((approval_id, client_name, await _row(db, approval_id) is not None))

        approvals.add_created_listener(cb)
        result = await _create(db, client="cli_b")
        await asyncio.gather(*approvals._listener_tasks)
        assert calls == [(result["approval_id"], "bob", True)]

    async def test_no_call_when_create_fails(self, db):
        calls = []

        async def cb(approval_id, client_name):
            calls.append(approval_id)

        approvals.add_created_listener(cb)
        await expect("invalid_argument", _create(db, options=["Approve"]))
        await asyncio.sleep(0)
        assert calls == []

    async def test_raising_and_hanging_listeners_do_not_affect_create(
        self, db, monkeypatch, caplog
    ):
        monkeypatch.setattr(approvals, "LISTENER_TIMEOUT_SECONDS", 0.05)
        reached = []

        async def boom(approval_id, client_name):
            raise RuntimeError("listener broke")

        async def hang(approval_id, client_name):
            await asyncio.Event().wait()

        async def good(approval_id, client_name):
            reached.append(approval_id)

        for cb in (boom, hang, good):
            approvals.add_created_listener(cb)
        caplog.set_level(logging.WARNING, logger=approvals.__name__)
        result = await _create(db)
        # Create returned while the listeners are still scheduled, so it never waited on them.
        assert result["status"] == "pending"
        assert len(approvals._listener_tasks) == 3
        assert not any(t.done() for t in approvals._listener_tasks)
        await asyncio.wait_for(asyncio.gather(*approvals._listener_tasks), 2)
        assert reached == [result["approval_id"]]
        assert "failed" in caplog.text and "timed out" in caplog.text
        assert not approvals._listener_tasks


class TestAudit:
    async def test_create_and_resolve_rows(self, db):
        aid = (await _create(db))["approval_id"]
        await approvals.resolve(db, aid, "deny", None, now=T0 + 1)
        async with db.read().execute(
            "SELECT actor, action, target, outcome FROM audit_log ORDER BY id"
        ) as cur:
            rows = [tuple(r) for r in await cur.fetchall()]
        assert rows == [
            ("cli_a", "approval.create", aid, "ok"),
            ("owner", "approval.resolve", aid, "denied"),
        ]

    async def test_refused_create_writes_nothing(self, db):
        for i in range(5):
            await _create(db, now=T0 + i)
        await expect("too_many_pending", _create(db, now=T0 + 10))
        assert await _scalar(db, "SELECT COUNT(*) FROM audit_log") == 5
