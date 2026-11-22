"""Approval queue: agents ask, the owner decides (contract sections 1, 3, 6 and 9).

Agents call ``create`` (through ``service.py``, WP 1.4) and poll ``list_for_project``; the owner
resolves through the Inbox route (WP 1.5). Approving records a choice and runs nothing.

Text arrives already cleaned and scanned by the caller; this module only trims and checks
options, enforces limits and records state. "Expired" is never stored: an approval is expired
while ``status='pending' AND expires_at <= now`` (section 1).
"""

import asyncio
import json
import logging
import secrets
import time
import unicodedata
from collections.abc import Awaitable, Callable, Sequence

from .db import Database
from .store import WorkspaceError

logger = logging.getLogger(__name__)

CreatedListener = Callable[[str, str], Awaitable[None]]

MAX_OPTIONS = 3
OPTION_MAX_LEN = 40
DEFAULT_OPTIONS = ("approve",)
MAX_LIVE_PENDING = 5
MAX_CREATED_PER_DAY = 20
MIN_TTL_SECONDS = 5 * 60
MAX_TTL_SECONDS = 10080 * 60
_DAY = 24 * 60 * 60
_RECENT_WINDOW = 7 * _DAY
_RESERVED = ("approve", "deny")

# Reason: module-level so tests can shorten it; the contract fixes it at 10 seconds.
LISTENER_TIMEOUT_SECONDS: float = 10.0

_listeners: list[CreatedListener] = []
# Reason: the event loop keeps only weak references to tasks, so a fire-and-forget listener
# task could be garbage-collected mid-run without a strong reference held here.
_listener_tasks: set[asyncio.Task] = set()


def _now(now: int | None) -> int:
    return int(time.time()) if now is None else now


def _fold(text: str) -> str:
    # Reason: casefold can produce non-NFKC output (e.g. compatibility characters), so normalise
    # on both sides to make the comparison stable.
    return unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text).casefold())


def _final_options(options: Sequence[str] | None) -> list[str]:
    """Validate the agent's options and return the list stored and shown to the owner.

    Args:
        options: One to three strings, or None for the default ``["approve"]``.

    Returns:
        The trimmed options with ``deny`` appended when absent.

    Raises:
        WorkspaceError: ``invalid_argument`` for a bad count, type, length, a duplicate under
            NFKC plus casefold, or a near-miss of ``approve``/``deny`` such as ``"Approve"``.
    """
    if options is None:
        options = DEFAULT_OPTIONS
    if isinstance(options, str) or not 1 <= len(options) <= MAX_OPTIONS:
        raise WorkspaceError("invalid_argument", f"options must be 1 to {MAX_OPTIONS} strings")
    final: list[str] = []
    seen: set[str] = set()
    for raw in options:
        if not isinstance(raw, str):
            raise WorkspaceError("invalid_argument", "each option must be a string")
        option = raw.strip()
        if not option or len(option) > OPTION_MAX_LEN:
            raise WorkspaceError(
                "invalid_argument", f"each option must be 1 to {OPTION_MAX_LEN} characters"
            )
        folded = _fold(option)
        if folded in seen:
            raise WorkspaceError("invalid_argument", "options must be unique")
        # Reason: status is derived from the exact strings approve/deny, so a look-alike such as
        # "Approve" would show as approval to the owner yet record "answered".
        if folded in _RESERVED and option != folded:
            raise WorkspaceError("invalid_argument", f"option must be exactly '{folded}'")
        seen.add(folded)
        final.append(option)
    if "deny" not in final:
        final.append("deny")
    return final


async def _audit(conn, actor: str, action: str, target: str, outcome: str, ts: int) -> None:
    # Reason: written on the caller's connection so the audit row commits or rolls back with
    # the change it records; store.audit opens its own transaction and cannot be nested.
    await conn.execute(
        "INSERT INTO audit_log (ts, actor, action, target, outcome)"
        " VALUES (:ts, :actor, :action, :target, :outcome)",
        {"ts": ts, "actor": actor, "action": action, "target": target, "outcome": outcome},
    )


async def _scalar(conn, sql: str, params: dict) -> int:
    async with conn.execute(sql, params) as cur:
        return int((await cur.fetchone())[0])


async def create(
    db: Database,
    client_id: str,
    project: str,
    task_id: str | None,
    title: str,
    body: str,
    options: Sequence[str] | None,
    ttl_seconds: int,
    flagged: int,
    *,
    now: int | None = None,
) -> dict:
    """Queue a decision for the owner and notify created-listeners after commit.

    Args:
        db: Open database.
        client_id: The requesting client's id.
        project: Project slug.
        task_id: Optional task the decision concerns; must belong to ``project``.
        title: Already-cleaned title.
        body: Already-cleaned body.
        options: One to three option strings, or None for the default.
        ttl_seconds: Seconds until the request expires (5 minutes to 7 days).
        flagged: Injection-guard flag for the text.
        now: Unix seconds override for tests.

    Returns:
        ``{approval_id, status: "pending", expires_at}``.

    Raises:
        WorkspaceError: ``invalid_argument``, ``not_found`` or ``too_many_pending``.
    """
    final = _final_options(options)
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int)
        or not MIN_TTL_SECONDS <= ttl_seconds <= MAX_TTL_SECONDS
    ):
        raise WorkspaceError(
            "invalid_argument",
            f"ttl_seconds must be {MIN_TTL_SECONDS} to {MAX_TTL_SECONDS}",
        )
    ts = _now(now)
    expires = ts + ttl_seconds
    approval_id = "apr_" + secrets.token_hex(8)
    async with db.write() as conn:
        async with conn.execute(
            "SELECT name FROM clients WHERE id = :id", {"id": client_id}
        ) as cur:
            client = await cur.fetchone()
        if client is None:
            raise WorkspaceError("not_found", "client not found")
        if task_id is not None:
            # Reason: project binding (section 3): another project's task looks missing.
            async with conn.execute(
                "SELECT 1 FROM tasks WHERE id = :id AND project = :project",
                {"id": task_id, "project": project},
            ) as cur:
                if await cur.fetchone() is None:
                    raise WorkspaceError("not_found", "task not found in this project")
        # Reason: counting inside the same BEGIN IMMEDIATE as the insert means two racing
        # creates, even from separate connections, cannot both pass the limit.
        live = await _scalar(
            conn,
            "SELECT COUNT(*) FROM approvals WHERE client_id = :me AND status = 'pending'"
            " AND expires_at > :ts",
            {"me": client_id, "ts": ts},
        )
        recent = await _scalar(
            conn,
            "SELECT COUNT(*) FROM approvals WHERE client_id = :me AND created_at > :since",
            {"me": client_id, "since": ts - _DAY},
        )
        if live >= MAX_LIVE_PENDING:
            raise WorkspaceError(
                "too_many_pending", f"at most {MAX_LIVE_PENDING} pending decisions per client"
            )
        if recent >= MAX_CREATED_PER_DAY:
            raise WorkspaceError(
                "too_many_pending",
                f"at most {MAX_CREATED_PER_DAY} decisions per client in 24 hours",
            )
        await conn.execute(
            "INSERT INTO approvals (id, project, task_id, client_id, title, body, options,"
            " flagged, created_at, expires_at) VALUES (:id, :project, :task, :me, :title, :body,"
            " :options, :flagged, :ts, :exp)",
            {
                "id": approval_id,
                "project": project,
                "task": task_id,
                "me": client_id,
                "title": title,
                "body": body,
                "options": json.dumps(final),
                "flagged": flagged,
                "ts": ts,
                "exp": expires,
            },
        )
        await _audit(conn, client_id, "approval.create", approval_id, "ok", ts)
    # Reason: after the context exits the row is committed and the write lock released, so a
    # listener can read the row and even write without deadlocking against this caller.
    _notify_created(approval_id, client["name"])
    return {"approval_id": approval_id, "status": "pending", "expires_at": expires}


async def resolve(
    db: Database,
    approval_id: str,
    choice: str,
    note: str | None,
    *,
    now: int | None = None,
) -> dict | None:
    """Record the owner's choice on a live pending approval.

    Args:
        db: Open database.
        approval_id: The approval to resolve.
        choice: One of the stored options, compared exactly.
        note: Owner's note, already cleaned by the route, or None.
        now: Unix seconds override for tests.

    Returns:
        ``{id, project, task_id, status, choice, note, resolved_at}``, or None when the approval
        is missing, already final or expired (the route answers 409). Nothing changes on None.

    Raises:
        WorkspaceError: ``invalid_argument`` when ``choice`` is not in the stored options.
    """
    ts = _now(now)
    async with db.write() as conn:
        async with conn.execute(
            "SELECT project, task_id, options FROM approvals WHERE id = :id"
            " AND status = 'pending' AND expires_at > :ts",
            {"id": approval_id, "ts": ts},
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        if choice not in json.loads(row["options"]):
            raise WorkspaceError("invalid_argument", "choice is not one of the options")
        status = {"approve": "approved", "deny": "denied"}.get(choice, "answered")
        # Reason: the WHERE guard repeats the liveness check so the state change is a single
        # conditional statement (section 6), safe even if a future caller skips the SELECT.
        cur = await conn.execute(
            "UPDATE approvals SET status = :status, choice = :choice, note = :note,"
            " resolved_at = :ts, resolved_by = 'owner'"
            " WHERE id = :id AND status = 'pending' AND expires_at > :ts",
            {"status": status, "choice": choice, "note": note, "ts": ts, "id": approval_id},
        )
        if cur.rowcount == 0:
            return None
        await _audit(conn, "owner", "approval.resolve", approval_id, status, ts)
    return {
        "id": approval_id,
        "project": row["project"],
        "task_id": row["task_id"],
        "status": status,
        "choice": choice,
        "note": note,
        "resolved_at": ts,
    }


async def list_for_project(db: Database, project: str, *, now: int | None = None) -> list[dict]:
    """List a project's live pending approvals plus those finished in the last seven days.

    Args:
        db: Open database.
        project: Project slug.
        now: Unix seconds override for tests.

    Returns:
        Newest first, ``{id, client, title, status, choice, note, expires_at}`` with ``client``
        a name and ``status`` reported as ``expired`` for pending past expiry.
    """
    ts = _now(now)
    # Reason: a live pending row has expires_at > now, so the single COALESCE window covers
    # live pending and everything finished (resolved, cancelled or expired) within seven days.
    async with db.read().execute(
        "SELECT a.id, c.name AS client, a.title,"
        " CASE WHEN a.status = 'pending' AND a.expires_at <= :ts THEN 'expired'"
        " ELSE a.status END AS status, a.choice, a.note, a.expires_at"
        " FROM approvals a JOIN clients c ON c.id = a.client_id"
        " WHERE a.project = :project AND COALESCE(a.resolved_at, a.expires_at) > :cutoff"
        " ORDER BY a.created_at DESC, a.id",
        {"ts": ts, "project": project, "cutoff": ts - _RECENT_WINDOW},
    ) as cur:
        return [dict(r) for r in await cur.fetchall()]


async def list_pending(db: Database, *, now: int | None = None) -> list[dict]:
    """List live pending approvals across all projects, oldest first (for the owner's Inbox).

    Args:
        db: Open database.
        now: Unix seconds override for tests.

    Returns:
        ``{id, project, task_id, client, title, body, options, flagged, created_at,
        expires_at}`` with ``client`` a name and ``options`` a list.
    """
    ts = _now(now)
    async with db.read().execute(
        "SELECT a.id, a.project, a.task_id, c.name AS client, a.title, a.body, a.options,"
        " a.flagged, a.created_at, a.expires_at FROM approvals a"
        " JOIN clients c ON c.id = a.client_id"
        " WHERE a.status = 'pending' AND a.expires_at > :ts ORDER BY a.created_at, a.id",
        {"ts": ts},
    ) as cur:
        rows = await cur.fetchall()
    return [{**dict(r), "options": json.loads(r["options"])} for r in rows]


def add_created_listener(cb: CreatedListener) -> None:
    """Register ``async def cb(approval_id, client_name)`` to run after each create commits.

    Args:
        cb: Coroutine function; failures and timeouts are logged and never reach the creator.
    """
    _listeners.append(cb)


def _notify_created(approval_id: str, client_name: str) -> None:
    for cb in list(_listeners):
        task = asyncio.get_running_loop().create_task(_run_listener(cb, approval_id, client_name))
        _listener_tasks.add(task)
        task.add_done_callback(_listener_tasks.discard)


async def _run_listener(cb: CreatedListener, approval_id: str, client_name: str) -> None:
    # Reason: asyncio.timeout() is 3.11+; wait_for works on 3.10 and cancels the listener.
    try:
        await asyncio.wait_for(cb(approval_id, client_name), LISTENER_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("approval listener %r timed out for %s", cb, approval_id)
    except Exception:
        logger.exception("approval listener %r failed for %s", cb, approval_id)
