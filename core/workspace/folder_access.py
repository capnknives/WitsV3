"""Folder access requests and owner grants (docs/architecture/folder-access.md, section 5).

A request is an approval in the existing queue plus a ``folder_requests`` row naming the folder.
A grant is created in exactly one place, ``grant_from_resolution``, which ``approvals.resolve``
calls inside the owner's resolve transaction. ``path_access.check`` reads live grants through the
provider that ``activate`` installs at workspace start-up; nothing is cached, so an expired or
revoked grant stops allowing access on the very next check.

``request`` never touches the filesystem for the supplied path: it works on the string alone,
so its answers cannot tell a model whether a file or folder exists.
"""

from __future__ import annotations

import logging
import ntpath
import os
import posixpath
import secrets
import time
from collections import deque
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from core import path_access
from core.injection_guard import scan_text

from . import approvals
from .clients import INTERNAL_CLIENT_ID
from .store import WorkspaceError
from .text import clean_text

if TYPE_CHECKING:
    from .db import Database

logger = logging.getLogger(__name__)

INTERNAL_CLIENT = INTERNAL_CLIENT_ID
PROJECT = "wits"
OPTIONS = ("approve", "one day")
_DAY = 24 * 60 * 60
DURATIONS = {"approve": 30 * _DAY, "one day": _DAY}
MAX_PENDING_REQUESTS = 3
DENY_COOLDOWN = _DAY
RATE_LIMIT = 10
RATE_WINDOW = 60 * 60
REASON_MAX = 8_000  # Reason: the reason is the approval body, so it gets the body limit.
PATH_MAX = 1_000
_IMPLIED: dict[str, tuple[path_access.Mode, ...]] = {
    "read": ("read",),
    "read_write": ("read", "write"),
    "index": ("read", "index"),
}

_db: Database | None = None
# Reason: per process, in memory; bounds how fast a model can probe even refused requests.
_calls: deque[int] = deque()
_provider_warned = False


def _refused(code: str, message: str) -> dict:
    return {"status": "refused", "code": code, "message": message}


def _key(path: str) -> str:
    """Duplicate key: spelling variants of one folder (case, separators) map to one request."""
    return path.replace("\\", "/").rstrip("/").casefold()


def _rate_ok(ts: int) -> bool:
    while _calls and _calls[0] <= ts - RATE_WINDOW:
        _calls.popleft()
    if len(_calls) >= RATE_LIMIT:
        return False
    _calls.append(ts)
    return True


def _pure(path: str, windows: bool) -> PurePath:
    if windows:
        return PureWindowsPath(ntpath.normpath(path))
    return PurePosixPath(posixpath.normpath(path))


def _target(path: Any, mode: str, config: Any | None) -> tuple[str | None, dict]:
    """Name the folder a grant would cover, or say why it cannot be requested.

    Works on the supplied string only (no resolve, stat or ``~`` expansion). A final component
    with a suffix is taken as a file and its parent is requested. Section 3 runs with no grants
    for every action the mode implies, on the path as named and on the folder.

    Returns:
        ``(folder, {})`` to request, or ``(None, result)`` with a refusal or
        ``already_allowed``.
    """
    if not isinstance(path, str) or not path.strip() or len(path) > PATH_MAX:
        return None, _refused("invalid_argument", f"path must be 1 to {PATH_MAX} characters")
    try:
        # Reason: a path with invisible characters would show the owner a different folder
        # from the one granted, so it is refused, never cleaned.
        if clean_text(path, PATH_MAX) != path:
            return None, _refused("invalid_argument", "path contains invisible characters")
    except WorkspaceError:
        return None, _refused("invalid_argument", "path contains invisible characters")
    windows = os.name == "nt"
    alias = path_access.alias_reason(path, windows)
    if alias:
        return None, _refused("never", f"never allowed: {alias}")
    if path.startswith("~"):
        return None, _refused("invalid_argument", "give the full path; '~' is not expanded")
    pure = _pure(path, windows)
    if not pure.is_absolute():
        return None, _refused("invalid_argument", "path must be absolute")
    folder = pure.parent if pure.suffix else pure
    try:
        roots = path_access.roots_for(config)
        decisions = [
            path_access.decide(target, action, "owner", (), roots)
            for target in (pure, folder)
            for action in _IMPLIED[mode]
        ]
    except (PermissionError, ValueError, RuntimeError, OSError, TypeError):
        return None, _refused("invalid_argument", "unusable path")
    for decision in decisions:
        if not decision.allowed and not decision.requestable:
            return None, _refused("never", decision.reason)
    if all(d.allowed for d in decisions):
        return None, {"status": "already_allowed", "path": str(folder)}
    return str(folder), {}


async def _audit(conn, actor: str, action: str, target: str, outcome: str, detail: str, ts: int):
    # Reason: on the caller's connection, so the row commits or rolls back with its change.
    await conn.execute(
        "INSERT INTO audit_log (ts, actor, action, target, outcome, detail)"
        " VALUES (:ts, :actor, :action, :target, :outcome, :detail)",
        dict(ts=ts, actor=actor, action=action, target=target, outcome=outcome, detail=detail),
    )


async def _log_request(db: Database, target: Any, outcome: str, detail: str, ts: int) -> None:
    text = target if isinstance(target, str) else repr(target)
    async with db.write() as conn:
        await _audit(conn, INTERNAL_CLIENT, "folder.request", text[:PATH_MAX], outcome, detail, ts)


async def request(
    db: Database,
    path: str,
    mode: str,
    reason: str,
    *,
    ttl_seconds: int = _DAY,
    config: Any | None = None,
    now: int | None = None,
) -> dict:
    """Queue a folder access request for the owner; never waits for the answer.

    Every call writes one ``folder.request`` audit row and counts against a per-process limit
    of ``RATE_LIMIT`` calls an hour.

    Args:
        db: Open workspace database.
        path: Absolute folder path; a path whose last part has a suffix is taken as a file and
            recorded against its parent folder.
        mode: ``read``, ``read_write`` or ``index``.
        reason: Why access is needed, shown to the owner as the approval body.
        ttl_seconds: How long the request stays answerable.
        config: WitsV3Config for the base folders, or None to load it.
        now: Unix seconds override for tests.

    Returns:
        ``{status: "pending", approval_id, path, mode, expires_at, existing}``,
        ``{status: "already_allowed", path}`` or ``{status: "refused", code, message}``.
        Only a new pending request creates approval rows.
    """
    ts = int(time.time()) if now is None else now
    if not _rate_ok(ts):
        await _log_request(db, path, "refused", "rate limited", ts)
        return _refused("rate_limited", "Too many folder requests; ask the owner directly.")
    folder, result = None, _refused("invalid_argument", "mode must be read, read_write or index")
    if mode in _IMPLIED:
        try:
            clean_reason = clean_text(reason, REASON_MAX)
            folder, result = _target(path, mode, config)
        except WorkspaceError as exc:
            result = _refused(exc.code, f"reason: {exc.message}")
    if folder is None:
        outcome = "already_allowed" if result["status"] == "already_allowed" else "refused"
        await _log_request(db, path, outcome, result.get("message", ""), ts)
        return result
    key = _key(folder)

    async def add_request(conn, approval_id: str, now_ts: int) -> None:
        # Reason: runs inside the approval's own BEGIN IMMEDIATE, so the checks and both
        # inserts are one atomic step even with racing requests.
        async with conn.execute(
            "SELECT r.approval_id, r.path, r.mode, a.status, a.expires_at FROM folder_requests r"
            " JOIN approvals a ON a.id = r.approval_id WHERE a.id != :id AND"
            " ((a.status = 'pending' AND a.expires_at > :ts)"
            " OR (a.status = 'denied' AND a.resolved_at > :since))",
            {"id": approval_id, "ts": now_ts, "since": now_ts - DENY_COOLDOWN},
        ) as cur:
            rows = await cur.fetchall()
        live = [r for r in rows if r["status"] == "pending"]
        same = [r for r in rows if _key(r["path"]) == key and r["mode"] == mode]
        # Reason: the cap is checked first so a full queue answers the same for every path.
        if len(live) >= MAX_PENDING_REQUESTS:
            raise WorkspaceError(
                "too_many_pending", f"at most {MAX_PENDING_REQUESTS} folder requests may wait"
            )
        for row in same:
            if row["status"] == "pending":
                raise WorkspaceError("duplicate", "already requested", dict(row))
        if same:
            raise WorkspaceError("cooldown", "the owner denied this within the last 24 hours")
        await conn.execute(
            "INSERT INTO folder_requests (approval_id, path, mode, reason, created_at)"
            " VALUES (:id, :path, :mode, :reason, :ts)",
            {"id": approval_id, "path": folder, "mode": mode, "reason": clean_reason, "ts": now_ts},
        )
        await _audit(conn, INTERNAL_CLIENT, "folder.request", folder, "ok", approval_id, now_ts)

    try:
        created = await approvals.create(
            db,
            INTERNAL_CLIENT,
            PROJECT,
            None,
            f"Folder access: {mode} {folder}",
            clean_reason,
            list(OPTIONS),
            ttl_seconds,
            1 if scan_text(clean_reason) else 0,
            now=ts,
            on_insert=add_request,
        )
    except WorkspaceError as exc:
        outcome = exc.code if exc.code in ("duplicate", "too_many_pending") else "refused"
        await _log_request(db, folder, outcome, exc.message, ts)
        if exc.code != "duplicate":
            return _refused(exc.code, exc.message)
        return {
            "status": "pending",
            "approval_id": exc.extra["approval_id"],
            "path": folder,
            "mode": mode,
            "expires_at": exc.extra["expires_at"],
            "existing": True,
        }
    return {**created, "path": folder, "mode": mode, "existing": False}


async def grant_from_resolution(conn, approval_id: str, choice: str, now: int) -> str | None:
    """Create the grant for a folder request the owner has just approved; the only grant writer.

    Called by ``approvals.resolve`` inside its transaction, after the status update. The folder
    and mode come from ``folder_requests`` (fixed at request time), never from the caller.

    Args:
        conn: The resolve transaction's connection.
        approval_id: The approval just resolved.
        choice: The owner's choice; ``approve`` gives 30 days, ``one day`` 24 hours.
        now: The resolve time, which must equal the approval's ``resolved_at``.

    Returns:
        The new grant id, or None when nothing is granted.
    """
    duration = DURATIONS.get(choice)
    if duration is None:
        return None
    async with conn.execute(
        "SELECT r.path, r.mode FROM folder_requests r JOIN approvals a ON a.id = r.approval_id"
        " WHERE r.approval_id = :id AND a.status IN ('approved', 'answered')"
        " AND a.choice = :choice AND a.resolved_at = :now AND a.resolved_by = 'owner'"
        " AND NOT EXISTS (SELECT 1 FROM folder_grants g WHERE g.approval_id = r.approval_id)",
        {"id": approval_id, "choice": choice, "now": now},
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    grant_id = "grt_" + secrets.token_hex(8)
    expires = now + duration
    await conn.execute(
        "INSERT INTO folder_grants (id, approval_id, path, mode, created_at, expires_at)"
        " VALUES (:id, :approval, :path, :mode, :now, :exp)",
        dict(
            id=grant_id,
            approval=approval_id,
            path=row["path"],
            mode=row["mode"],
            now=now,
            exp=expires,
        ),
    )
    detail = f"{row['mode']} {row['path']} until {expires}"
    await _audit(conn, "owner", "folder.grant", grant_id, "ok", detail, now)
    return grant_id


async def live_grants(db: Database, *, now: int | None = None) -> tuple[path_access.Grant, ...]:
    """Return the grants that allow access right now (unexpired and not revoked)."""
    ts = int(time.time()) if now is None else now
    async with db.read().execute(
        "SELECT path, mode, expires_at, revoked_at FROM folder_grants"
        " WHERE revoked_at IS NULL AND expires_at > :ts",
        {"ts": ts},
    ) as cur:
        rows = await cur.fetchall()
    return tuple(
        path_access.Grant(r["path"], r["mode"], int(r["expires_at"]), r["revoked_at"]) for r in rows
    )


async def list_grants(db: Database, *, now: int | None = None) -> list[dict]:
    """List grants for the owner, newest first, each with a ``live`` flag."""
    ts = int(time.time()) if now is None else now
    async with db.read().execute(
        "SELECT id, approval_id, path, mode, created_at, expires_at, revoked_at,"
        " (revoked_at IS NULL AND expires_at > :ts) AS live FROM folder_grants"
        " ORDER BY created_at DESC, id LIMIT 200",
        {"ts": ts},
    ) as cur:
        return [{**dict(r), "live": bool(r["live"])} for r in await cur.fetchall()]


async def revoke(db: Database, grant_id: str, *, now: int | None = None) -> bool:
    """Revoke a live grant. Returns False when it is unknown, already revoked or expired."""
    ts = int(time.time()) if now is None else now
    async with db.write() as conn:
        cur = await conn.execute(
            "UPDATE folder_grants SET revoked_at = :ts"
            " WHERE id = :id AND revoked_at IS NULL AND expires_at > :ts",
            {"ts": ts, "id": grant_id},
        )
        outcome = "ok" if cur.rowcount == 1 else "not_found"
        await _audit(conn, "owner", "folder.revoke", str(grant_id)[:100], outcome, "", ts)
    return outcome == "ok"


def current_db() -> Database | None:
    """The workspace database while the workspace runs, else None (requests are refused)."""
    return _db


def activate(db: Database) -> None:
    """Point ``path_access`` at the live grants in *db* (workspace start-up)."""
    global _db
    _db = db

    async def provider() -> tuple[path_access.Grant, ...]:
        global _provider_warned
        try:
            return await live_grants(db)
        except Exception as err:  # Reason: a failed read must narrow access, never widen it.
            if not _provider_warned:
                _provider_warned = True
                logger.warning("folder grants unreadable, using no grants: %s", err)
            return ()

    path_access.grants_provider = provider


def deactivate() -> None:
    """Back to "no grants" (workspace shutdown)."""
    global _db
    _db = None
    path_access.grants_provider = path_access._no_grants
