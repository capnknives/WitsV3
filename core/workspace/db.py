"""SQLite connections, migrations and the single-writer gate (contract sections 1 and 2)."""

import asyncio
import re
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

# Reason: module-level so tests can point migrations at a temporary directory without widening
# the ``open_db(path)`` signature other packages import.
MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_MIGRATION_NAME = re.compile(r"^(\d{4})_(.+)\.sql$")
_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA recursive_triggers=ON",
    "PRAGMA busy_timeout=5000",
)


class Database:
    """A write connection behind one lock, plus a read connection.

    Attributes:
        path: Absolute path of the database file.
    """

    def __init__(
        self, path: Path, write_conn: aiosqlite.Connection, read_conn: aiosqlite.Connection
    ) -> None:
        self.path = path
        self._write_conn = write_conn
        self._read_conn = read_conn
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def write(self) -> AsyncIterator[aiosqlite.Connection]:
        """Hold the writer lock inside ``BEGIN IMMEDIATE``; commit on exit, roll back on error."""
        async with self._lock:
            conn = self._write_conn
            await conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                await conn.execute("ROLLBACK")
                raise
            await conn.execute("COMMIT")

    def read(self) -> aiosqlite.Connection:
        """Return the read connection (for owner list routes; never write through it)."""
        return self._read_conn

    async def close(self) -> None:
        """Close both connections."""
        await self._read_conn.close()
        await self._write_conn.close()


async def _connect(path: Path) -> aiosqlite.Connection:
    # Reason: isolation_level=None keeps Python from opening implicit transactions, so the
    # explicit BEGIN IMMEDIATE in Database.write() is the only transaction control.
    conn = await aiosqlite.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    for pragma in _PRAGMAS:
        await (await conn.execute(pragma)).close()
    return conn


def _list_migrations() -> list[tuple[int, str, str]]:
    found = []
    for file in MIGRATIONS_DIR.iterdir():
        match = _MIGRATION_NAME.match(file.name)
        if match:
            found.append((int(match.group(1)), match.group(2), file.read_text(encoding="utf-8")))
    return sorted(found)


def _split_statements(sql: str) -> list[str]:
    # Reason: executescript() commits half a failed file, and splitting on ';' breaks trigger
    # bodies; complete_statement knows where a statement really ends.
    statements, buf = [], ""
    for line in sql.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            statements.append(buf)
            buf = ""
    if any(ln.strip() and not ln.strip().startswith("--") for ln in buf.splitlines()):
        raise ValueError("migration ends with an incomplete statement")
    return statements


async def _current_version(conn: aiosqlite.Connection) -> int:
    async with conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ) as cur:
        if await cur.fetchone() is None:
            return 0
    async with conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version") as cur:
        row = await cur.fetchone()
    return int(row[0])


async def _backup(conn: aiosqlite.Connection, path: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.parent / f"{path.name}.{stamp}.bak"
    await (await conn.execute("VACUUM INTO ?", (str(target),))).close()


async def _migrate(conn: aiosqlite.Connection, path: Path) -> None:
    migrations = await asyncio.to_thread(_list_migrations)
    current = await _current_version(conn)
    pending = [m for m in migrations if m[0] > current]
    if not pending:
        return
    if current > 0:
        await _backup(conn, path)
    for version, name, sql in pending:
        statements = _split_statements(sql)
        await conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in statements:
                await conn.execute(statement)
            await conn.execute(
                "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                (version, int(time.time()), name),
            )
        except BaseException:
            await conn.execute("ROLLBACK")
            raise
        await conn.execute("COMMIT")


async def open_db(path: Path) -> Database:
    """Open (creating if needed) the workspace database and apply pending migrations.

    Args:
        path: Database file location; parent directories are created.

    Returns:
        A ready ``Database`` with pragmas set and the schema at the latest version.
    """
    path = Path(path).resolve()
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    write_conn = await _connect(path)
    try:
        await _migrate(write_conn, path)
        read_conn = await _connect(path)
    except BaseException:
        await write_conn.close()
        raise
    return Database(path, write_conn, read_conn)
