"""Safety tests for the workspace database (WP 1.8): interrupted BEGIN and statement logging."""

import asyncio
import logging
import sqlite3

import pytest

from core.workspace.db import open_db


@pytest.fixture
async def db(tmp_path):
    database = await open_db(tmp_path / "ws.db")
    try:
        yield database
    finally:
        await database.close()


async def _insert_note(db, body: str) -> None:
    async with db.write() as conn:
        await conn.execute(
            "INSERT INTO audit_log (ts, actor, action, target, outcome, detail) "
            "VALUES (0, 'test', 'note', NULL, 'ok', ?)",
            (body,),
        )


async def test_cancel_during_begin_leaves_connection_usable(db, monkeypatch):
    # Reason: another connection holding the write lock keeps BEGIN IMMEDIATE waiting on
    # aiosqlite's thread, so the cancel lands while the statement is still running.
    blocker = sqlite3.connect(db.path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    conn = db._write_conn
    real_execute = conn.execute
    at_begin = asyncio.Event()

    def spy(sql, *args, **kwargs):
        if sql == "BEGIN IMMEDIATE":
            at_begin.set()
        return real_execute(sql, *args, **kwargs)

    monkeypatch.setattr(conn, "execute", spy)
    task = asyncio.ensure_future(_insert_note(db, "never written"))
    await at_begin.wait()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0)
    # The BEGIN now completes on the thread after the caller was cancelled.
    blocker.execute("COMMIT")
    blocker.close()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not db._lock.locked()
    assert not conn.in_transaction
    await _insert_note(db, "written after cancel")
    async with db.read().execute("SELECT detail FROM audit_log") as cur:
        rows = [r[0] for r in await cur.fetchall()]
    assert rows == ["written after cancel"]


async def test_failed_begin_releases_lock_and_next_write_succeeds(db, monkeypatch):
    conn = db._write_conn
    real_execute = conn.execute
    calls = {"n": 0}

    def flaky(sql, *args, **kwargs):
        if sql == "BEGIN IMMEDIATE" and calls["n"] == 0:
            calls["n"] += 1
            return real_execute("BEGIN NONSENSE")
        return real_execute(sql, *args, **kwargs)

    monkeypatch.setattr(conn, "execute", flaky)
    with pytest.raises(sqlite3.OperationalError):
        await _insert_note(db, "x")
    assert not db._lock.locked() and not conn.in_transaction
    await _insert_note(db, "y")


async def test_statement_parameters_never_reach_the_log(db, caplog):
    marker = "MARKER-7f3a-handoff-body"
    with caplog.at_level(logging.DEBUG):
        await _insert_note(db, marker)
        async with db.read().execute("SELECT detail FROM audit_log") as cur:
            assert (await cur.fetchone())[0] == marker
    assert not [r for r in caplog.records if marker in r.getMessage()]
    assert logging.getLogger("aiosqlite").getEffectiveLevel() >= logging.INFO
