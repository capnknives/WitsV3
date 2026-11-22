"""Workspace store: tasks, locks, handoffs and artifacts (contract sections 3, 4 and 9).

Functions here enforce project binding and lock rules. They do not clean text, check scopes or
validate URIs; ``service.py`` (WP 1.4) does that before calling in.
"""

import secrets
import time
from typing import Any

from .db import Database

HANDOFF_KINDS = ("note", "handoff", "blocked", "done")
ARTIFACT_KINDS = ("pr", "commit", "branch", "file", "url", "doc")
_RELEASING_KINDS = ("handoff", "blocked", "done")
_TASK_LIMIT = 50
_ARTIFACT_LIMIT = 50


class WorkspaceError(Exception):
    """A failure with a stable code, per the section 3 error shapes.

    Attributes:
        code: Machine-readable code such as ``not_found`` or ``task_locked``.
        message: Human-readable description.
        extra: Optional extra fields (for example ``holder`` and ``expires_at``).
    """

    def __init__(self, code: str, message: str, extra: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.extra = extra or {}


def _now(now: int | None) -> int:
    return int(time.time()) if now is None else now


async def _get_task(conn, project: str, task_id: str):
    # Reason: project binding is checked before any write, so another project's task id is
    # indistinguishable from a missing one.
    async with conn.execute(
        "SELECT id, title, body, status FROM tasks WHERE id = :id AND project = :project",
        {"id": task_id, "project": project},
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        raise WorkspaceError("not_found", "task not found in this project")
    return row


def _require_open(task) -> None:
    if task["status"] != "open":
        raise WorkspaceError("task_closed", f"task is {task['status']}")


async def claim_task(
    db: Database,
    client_id: str,
    project: str,
    task_id: str | None,
    title: str | None,
    body: str,
    lease_seconds: int,
    flagged: int,
    *,
    now: int | None = None,
) -> dict:
    """Claim an existing task, or create a task and claim it in one transaction.

    Args:
        db: Open database.
        client_id: The caller's client id.
        project: Project slug the task must belong to.
        task_id: Existing task to claim; exclusive with ``title``.
        title: Title of a new task to create and claim; exclusive with ``task_id``.
        body: Body of the new task (used only with ``title``).
        lease_seconds: Lease length.
        flagged: Injection-guard flag for new task text.
        now: Unix seconds override for tests.

    Returns:
        ``{task_id, title, body, created, lease_expires_at}``.

    Raises:
        WorkspaceError: ``invalid_argument``, ``not_found``, ``task_closed`` or ``task_locked``
            (extra: ``holder``, ``expires_at``).
    """
    if (task_id is None) == (title is None):
        raise WorkspaceError("invalid_argument", "give exactly one of task_id or title")
    ts = _now(now)
    expires = ts + lease_seconds
    async with db.write() as conn:
        if title is not None:
            task_id = "tsk_" + secrets.token_hex(8)
            await conn.execute(
                "INSERT INTO tasks (id, project, title, body, created_by, flagged, created_at,"
                " updated_at) VALUES (:id, :project, :title, :body, :me, :flagged, :ts, :ts)",
                {
                    "id": task_id,
                    "project": project,
                    "title": title,
                    "body": body,
                    "me": client_id,
                    "flagged": flagged,
                    "ts": ts,
                },
            )
            task_title, task_body, created = title, body, True
        else:
            task = await _get_task(conn, project, task_id)
            _require_open(task)
            task_title, task_body, created = task["title"], task["body"], False
        # Reason: one upsert is the whole lock rule. It only writes when the lock lapsed or the
        # caller already holds it; claimed_at survives a live renewal.
        cur = await conn.execute(
            "INSERT INTO task_locks (task_id, client_id, lease_seconds, claimed_at, expires_at)"
            " VALUES (:task, :me, :lease, :ts, :exp)"
            " ON CONFLICT(task_id) DO UPDATE SET client_id = excluded.client_id,"
            " lease_seconds = excluded.lease_seconds, expires_at = excluded.expires_at,"
            " claimed_at = CASE WHEN task_locks.client_id = :me AND task_locks.expires_at > :ts"
            " THEN task_locks.claimed_at ELSE excluded.claimed_at END"
            " WHERE task_locks.expires_at <= :ts OR task_locks.client_id = :me",
            {"task": task_id, "me": client_id, "lease": lease_seconds, "ts": ts, "exp": expires},
        )
        if cur.rowcount == 0:
            async with conn.execute(
                "SELECT c.name, l.expires_at FROM task_locks l JOIN clients c ON c.id = l.client_id"
                " WHERE l.task_id = :task",
                {"task": task_id},
            ) as held:
                row = await held.fetchone()
            raise WorkspaceError(
                "task_locked",
                "task is claimed by another client",
                {"holder": row["name"], "expires_at": row["expires_at"]},
            )
        await conn.execute(
            "UPDATE tasks SET updated_at = :ts WHERE id = :id", {"ts": ts, "id": task_id}
        )
    return {
        "task_id": task_id,
        "title": task_title,
        "body": task_body,
        "created": created,
        "lease_expires_at": expires,
    }


async def append_handoff(
    db: Database,
    client_id: str,
    project: str,
    task_id: str | None,
    kind: str,
    body: str,
    flagged: int,
    *,
    now: int | None = None,
) -> dict:
    """Append a handoff record, applying the section 4 lock rules for task-bound entries.

    Args:
        db: Open database.
        client_id: The caller's client id.
        project: Project slug.
        task_id: Task the entry concerns, or None for a project-level note.
        kind: ``note``, ``handoff``, ``blocked`` or ``done``.
        body: Entry text.
        flagged: Injection-guard flag.
        now: Unix seconds override for tests.

    Returns:
        ``{handoff_id, lock_released}``.

    Raises:
        WorkspaceError: ``invalid_argument``, ``not_found``, ``task_closed`` or
            ``not_lock_holder``.
    """
    if kind not in HANDOFF_KINDS:
        raise WorkspaceError("invalid_argument", f"kind must be one of {', '.join(HANDOFF_KINDS)}")
    ts = _now(now)
    released = False
    async with db.write() as conn:
        if task_id is not None:
            _require_open(await _get_task(conn, project, task_id))
            async with conn.execute(
                "SELECT lease_seconds FROM task_locks WHERE task_id = :task AND client_id = :me"
                " AND expires_at > :ts",
                {"task": task_id, "me": client_id, "ts": ts},
            ) as cur:
                lock = await cur.fetchone()
            if kind in _RELEASING_KINDS:
                if lock is None:
                    raise WorkspaceError("not_lock_holder", "caller does not hold this task's lock")
                await conn.execute(
                    "DELETE FROM task_locks WHERE task_id = :task", {"task": task_id}
                )
                released = True
                if kind == "done":
                    await conn.execute(
                        "UPDATE tasks SET status = 'done' WHERE id = :id", {"id": task_id}
                    )
            elif lock is not None:
                # Reason: a note proves the holder is alive, but must never shorten a lease.
                await conn.execute(
                    "UPDATE task_locks SET expires_at = MAX(expires_at, :new) WHERE task_id = :task",
                    {"new": ts + lock["lease_seconds"], "task": task_id},
                )
            await conn.execute(
                "UPDATE tasks SET updated_at = :ts WHERE id = :id", {"ts": ts, "id": task_id}
            )
        cur = await conn.execute(
            "INSERT INTO handoffs (project, task_id, client_id, kind, body, flagged, created_at)"
            " VALUES (:project, :task, :me, :kind, :body, :flagged, :ts)",
            {
                "project": project,
                "task": task_id,
                "me": client_id,
                "kind": kind,
                "body": body,
                "flagged": flagged,
                "ts": ts,
            },
        )
        handoff_id = cur.lastrowid
    return {"handoff_id": handoff_id, "lock_released": released}


async def link_artifact(
    db: Database,
    client_id: str,
    project: str,
    task_id: str | None,
    kind: str,
    uri: str,
    title: str,
    *,
    now: int | None = None,
) -> dict:
    """Record an artifact reference; a repeated URI in a project returns the existing row.

    Args:
        db: Open database.
        client_id: The caller's client id.
        project: Project slug.
        task_id: Optional task the artifact belongs to.
        kind: One of ``pr``, ``commit``, ``branch``, ``file``, ``url``, ``doc``.
        uri: Already-validated reference; never fetched.
        title: Optional display title.
        now: Unix seconds override for tests.

    Returns:
        ``{artifact_id, created}``.

    Raises:
        WorkspaceError: ``invalid_argument`` or ``not_found``.
    """
    if kind not in ARTIFACT_KINDS:
        raise WorkspaceError("invalid_argument", f"kind must be one of {', '.join(ARTIFACT_KINDS)}")
    ts = _now(now)
    async with db.write() as conn:
        if task_id is not None:
            await _get_task(conn, project, task_id)
        artifact_id = "art_" + secrets.token_hex(8)
        cur = await conn.execute(
            "INSERT INTO artifacts (id, project, task_id, client_id, kind, uri, title, created_at)"
            " VALUES (:id, :project, :task, :me, :kind, :uri, :title, :ts)"
            " ON CONFLICT(project, uri) DO NOTHING",
            {
                "id": artifact_id,
                "project": project,
                "task": task_id,
                "me": client_id,
                "kind": kind,
                "uri": uri,
                "title": title,
                "ts": ts,
            },
        )
        created = cur.rowcount == 1
        if not created:
            async with conn.execute(
                "SELECT id FROM artifacts WHERE project = :project AND uri = :uri",
                {"project": project, "uri": uri},
            ) as existing:
                artifact_id = (await existing.fetchone())["id"]
    return {"artifact_id": artifact_id, "created": created}


async def project_state(
    db: Database,
    project: str,
    since_handoff_id: int,
    limit: int,
    *,
    now: int | None = None,
) -> dict:
    """Read a project's open tasks, recent handoffs and artifacts (approvals are merged later).

    Args:
        db: Open database.
        project: Project slug.
        since_handoff_id: Only handoffs with a larger id are returned.
        limit: Maximum handoffs returned (the newest ones, in ascending order).
        now: Unix seconds override for tests.

    Returns:
        ``{project, tasks, handoffs, artifacts, truncated}``; client fields are names.
    """
    ts = _now(now)
    conn = db.read()
    params = {"project": project, "ts": ts}
    async with conn.execute(
        "SELECT t.id, t.title, t.flagged, c.name AS holder, l.expires_at FROM tasks t"
        " LEFT JOIN task_locks l ON l.task_id = t.id AND l.expires_at > :ts"
        " LEFT JOIN clients c ON c.id = l.client_id"
        " WHERE t.project = :project AND t.status = 'open'"
        " ORDER BY t.updated_at DESC, t.id LIMIT :n",
        {**params, "n": _TASK_LIMIT + 1},
    ) as cur:
        task_rows = await cur.fetchall()
    async with conn.execute(
        "SELECT * FROM (SELECT h.id, h.task_id, c.name AS client, h.kind, h.body, h.flagged,"
        " h.created_at FROM handoffs h JOIN clients c ON c.id = h.client_id"
        " WHERE h.project = :project AND h.id > :since ORDER BY h.id DESC LIMIT :n) ORDER BY id",
        {"project": project, "since": since_handoff_id, "n": limit + 1},
    ) as cur:
        handoff_rows = await cur.fetchall()
    async with conn.execute(
        "SELECT a.id, a.task_id, c.name AS client, a.kind, a.uri, a.title, a.created_at"
        " FROM artifacts a JOIN clients c ON c.id = a.client_id WHERE a.project = :project"
        " ORDER BY a.created_at DESC, a.rowid DESC LIMIT :n",
        {"project": project, "n": _ARTIFACT_LIMIT + 1},
    ) as cur:
        artifact_rows = await cur.fetchall()
    # Reason: each query fetches one row past its cap so truncation is detected, not guessed.
    # The handoff query is newest-first, so its extra row is the oldest and is dropped.
    handoffs_cut = len(handoff_rows) > limit
    if handoffs_cut:
        handoff_rows = handoff_rows[1:]
    truncated = handoffs_cut or len(task_rows) > _TASK_LIMIT or len(artifact_rows) > _ARTIFACT_LIMIT
    return {
        "project": project,
        "tasks": [
            {
                "id": r["id"],
                "title": r["title"],
                "flagged": r["flagged"],
                "lock": (
                    {"client": r["holder"], "expires_at": r["expires_at"]} if r["holder"] else None
                ),
            }
            for r in task_rows[:_TASK_LIMIT]
        ],
        "handoffs": [dict(r) for r in handoff_rows],
        "artifacts": [dict(r) for r in artifact_rows[:_ARTIFACT_LIMIT]],
        "truncated": truncated,
    }


async def audit(
    db: Database,
    actor: str,
    action: str,
    target: str | None,
    outcome: str,
    detail: str | None = None,
    *,
    now: int | None = None,
) -> None:
    """Append one audit_log row.

    Args:
        db: Open database.
        actor: Client id, or ``owner``.
        action: Tool name or admin action.
        target: What the action touched, if anything.
        outcome: ``ok`` or an error code.
        detail: Optional extra text.
        now: Unix seconds override for tests.
    """
    async with db.write() as conn:
        await conn.execute(
            "INSERT INTO audit_log (ts, actor, action, target, outcome, detail)"
            " VALUES (:ts, :actor, :action, :target, :outcome, :detail)",
            {
                "ts": _now(now),
                "actor": actor,
                "action": action,
                "target": target,
                "outcome": outcome,
                "detail": detail,
            },
        )
