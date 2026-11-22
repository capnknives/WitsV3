"""Network client identities and bearer tokens (contract sections 1, 5 and 9).

A token is ``wits_<8 hex prefix>_<token_urlsafe(32)>``. Only the prefix (the lookup key) and the
sha256 hex of the whole token are stored. The full token exists in memory once, at creation, and
is returned to the owner; it is never logged, audited or put into an exception message.
"""

import hashlib
import hmac
import logging
import re
import secrets
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .db import Database
from .store import WorkspaceError
from .text import clean_text

logger = logging.getLogger("WitsV3.Workspace.Clients")

# Reason: fullmatch with re.ASCII; a plain "$" would also accept a trailing newline.
TOKEN_RE = re.compile(r"wits_([0-9a-f]{8})_([A-Za-z0-9_-]{43})", re.ASCII)
PROJECT_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}", re.ASCII)
VALID_SCOPES = frozenset({"read", "write", "decide:request"})
NAME_MAX = 100
_LAST_SEEN_INTERVAL = 60
_PREFIX_ATTEMPTS = 8
# Reason: inserted by migration 0002 so folder requests can own approvals; never a credential.
INTERNAL_CLIENT_ID = "cli_internal_wits"


@dataclass(frozen=True)
class ClientIdentity:
    """A verified client: id, unique name, scopes, project slugs or ``{"*"}``, and rate limit."""

    id: str
    name: str
    scopes: frozenset
    projects: frozenset
    rate_per_min: int


def _now(now: int | None) -> int:
    return int(time.time()) if now is None else now


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _as_set(values: Iterable[str]) -> frozenset:
    return frozenset(values.split() if isinstance(values, str) else values)


def _normalize_scopes(scopes: Iterable[str]) -> frozenset:
    result = _as_set(scopes)
    if not result or not result <= VALID_SCOPES:
        raise WorkspaceError(
            "invalid_argument", f"scopes must be a non-empty subset of {sorted(VALID_SCOPES)}"
        )
    return result


def _normalize_projects(projects: Iterable[str]) -> frozenset:
    result = _as_set(projects)
    if not result:
        raise WorkspaceError("invalid_argument", "projects must not be empty (use '*' for all)")
    if "*" in result:
        # Reason: '*' already means every project; storing it alone keeps the row unambiguous.
        return frozenset({"*"})
    bad = [p for p in result if not isinstance(p, str) or not PROJECT_RE.fullmatch(p)]
    if bad:
        raise WorkspaceError("invalid_argument", "project slugs must match ^[a-z0-9][a-z0-9-]*$")
    return result


def _identity(row: Any) -> ClientIdentity:
    return ClientIdentity(
        id=row["id"],
        name=row["name"],
        scopes=frozenset(row["scopes"].split()),
        projects=frozenset(row["projects"].split()),
        rate_per_min=row["rate_per_min"],
    )


async def _audit_row(conn, ts: int, action: str, target: str, outcome: str, detail: str) -> None:
    # Reason: written on the caller's connection so the audit row commits or rolls back with
    # the change it records (store.audit would take the writer lock a second time).
    await conn.execute(
        "INSERT INTO audit_log (ts, actor, action, target, outcome, detail)"
        " VALUES (:ts, 'owner', :action, :target, :outcome, :detail)",
        {"ts": ts, "action": action, "target": target, "outcome": outcome, "detail": detail},
    )


async def create_client(
    db: Database,
    name: str,
    scopes: Iterable[str],
    projects: Iterable[str],
    rate_per_min: int = 60,
    expires_at: int | None = None,
    *,
    now: int | None = None,
) -> tuple[ClientIdentity, str]:
    """Create a client and mint its token.

    Args:
        db: Open database.
        name: Unique display name (cleaned, at most ``NAME_MAX`` characters).
        scopes: Scopes to grant, a non-empty subset of ``VALID_SCOPES``.
        projects: Project slugs the client may touch, or ``["*"]``.
        rate_per_min: Token bucket size per minute, at least 1.
        expires_at: Optional Unix seconds after which the token stops verifying.
        now: Unix seconds override for tests.

    Returns:
        The new identity and the full token. The token is not recoverable afterwards.

    Raises:
        WorkspaceError: ``invalid_argument`` for bad input, ``name_taken`` for a duplicate name.
    """
    clean_name = clean_text(name, NAME_MAX)
    # Reason: the name is written to logs and audit detail; a line break could forge an entry.
    if "\n" in clean_name or "\t" in clean_name:
        raise WorkspaceError("invalid_argument", "name must be a single line without tabs")
    scope_set = _normalize_scopes(scopes)
    project_set = _normalize_projects(projects)
    if isinstance(rate_per_min, bool) or not isinstance(rate_per_min, int) or rate_per_min < 1:
        raise WorkspaceError("invalid_argument", "rate_per_min must be a positive integer")
    if expires_at is not None and (isinstance(expires_at, bool) or not isinstance(expires_at, int)):
        raise WorkspaceError("invalid_argument", "expires_at must be Unix seconds or null")
    ts = _now(now)
    client_id = "cli_" + secrets.token_hex(8)
    async with db.write() as conn:
        async with conn.execute(
            "SELECT 1 FROM clients WHERE name = :name", {"name": clean_name}
        ) as cur:
            if await cur.fetchone() is not None:
                raise WorkspaceError("name_taken", "a client with that name already exists")
        # Reason: the prefix column is UNIQUE; checking inside BEGIN IMMEDIATE under the writer
        # lock means nothing can take a free prefix between the check and the insert.
        for _ in range(_PREFIX_ATTEMPTS):
            prefix = secrets.token_hex(4)
            async with conn.execute(
                "SELECT 1 FROM clients WHERE token_prefix = :p", {"p": prefix}
            ) as cur:
                if await cur.fetchone() is None:
                    break
        else:
            raise WorkspaceError("prefix_exhausted", "could not find a free token prefix")
        token = f"wits_{prefix}_{secrets.token_urlsafe(32)}"
        scope_text, project_text = " ".join(sorted(scope_set)), " ".join(sorted(project_set))
        row = dict(id=client_id, name=clean_name, prefix=prefix, hash=_hash(token), ts=ts)
        row.update(scopes=scope_text, projects=project_text, rate=rate_per_min, exp=expires_at)
        await conn.execute(
            "INSERT INTO clients (id, name, token_prefix, token_hash, scopes, projects,"
            " rate_per_min, created_at, expires_at)"
            " VALUES (:id, :name, :prefix, :hash, :scopes, :projects, :rate, :ts, :exp)",
            row,
        )
        detail = f"name={clean_name} scopes={scope_text}"
        await _audit_row(conn, ts, "client.create", client_id, "ok", detail)
    logger.info("workspace client created: %s (%s)", client_id, clean_name)
    identity = ClientIdentity(client_id, clean_name, scope_set, project_set, rate_per_min)
    return identity, token


async def verify_token(
    db: Database, token: str, *, now: int | None = None
) -> ClientIdentity | None:
    """Return the identity a bearer token belongs to, or None.

    Checked on every call with no cache: shape, row by prefix, constant-time hash comparison,
    then revocation and expiry. Never raises for bad input.

    Returns:
        The identity, or None for any malformed, unknown, wrong, revoked or expired token.
    """
    if not isinstance(token, str) or not token.isascii():
        return None
    match = TOKEN_RE.fullmatch(token)
    if match is None:
        return None
    async with db.read().execute(
        "SELECT id, name, token_hash, scopes, projects, rate_per_min, expires_at, last_seen_at,"
        " revoked_at FROM clients WHERE token_prefix = :p",
        {"p": match.group(1)},
    ) as cur:
        row = await cur.fetchone()
    # Reason: the internal WITS row (migration 0002) only anchors foreign keys; it is never a
    # credential, even if its placeholder hash or prefix were ever changed.
    if row is None or row["id"] == INTERNAL_CLIENT_ID:
        return None
    if not hmac.compare_digest(_hash(token), row["token_hash"]):
        return None
    ts = _now(now)
    if row["revoked_at"] is not None:
        return None
    if row["expires_at"] is not None and row["expires_at"] <= ts:
        return None
    last_seen = row["last_seen_at"]
    if last_seen is None or ts - last_seen >= _LAST_SEEN_INTERVAL:
        # Reason: the WHERE repeats the throttle so concurrent requests write at most once.
        async with db.write() as conn:
            await conn.execute(
                "UPDATE clients SET last_seen_at = :ts WHERE id = :id"
                " AND (last_seen_at IS NULL OR last_seen_at <= :cutoff)",
                {"ts": ts, "id": row["id"], "cutoff": ts - _LAST_SEEN_INTERVAL},
            )
    return _identity(row)


async def revoke_client(db: Database, client_id: str, *, now: int | None = None) -> bool:
    """Revoke a client, free its locks and cancel its live pending approvals, in one transaction.

    Rows are kept. Returns True if the client was active, False if unknown or already revoked.
    """
    ts = _now(now)
    params = {"id": client_id, "ts": ts, "internal": INTERNAL_CLIENT_ID}
    async with db.write() as conn:
        # Reason: revoking the internal row would cancel the owner's pending folder requests.
        cur = await conn.execute(
            "UPDATE clients SET revoked_at = :ts WHERE id = :id AND revoked_at IS NULL"
            " AND id != :internal",
            params,
        )
        if cur.rowcount != 1:
            await _audit_row(conn, ts, "client.revoke", str(client_id)[:100], "not_found", "")
            return False
        await conn.execute("DELETE FROM task_locks WHERE client_id = :id", params)
        cancelled = await conn.execute(
            "UPDATE approvals SET status = 'cancelled', resolved_at = :ts"
            " WHERE client_id = :id AND status = 'pending' AND expires_at > :ts",
            params,
        )
        detail = f"approvals_cancelled={cancelled.rowcount}"
        await _audit_row(conn, ts, "client.revoke", client_id, "ok", detail)
    logger.info("workspace client revoked: %s", client_id)
    return True


async def enforce_not_revoked(db: Database, client_id: str, *, now: int | None = None) -> None:
    """Undo what a write made in the revoke window, then refuse (contract section 5).

    A call verified just before a revoke committed can still write once. Called after every
    successful write tool: if the client is revoked by now, its locks and live pending
    approvals are cleared again and the caller gets ``revoked`` instead of the result. An
    unknown client id is treated as revoked, so a missing row never passes.

    Args:
        db: Open database. Must not be called inside ``db.write()`` (it is not re-entrant).
        client_id: The caller's client id.
        now: Unix seconds override for tests.

    Raises:
        WorkspaceError: ``revoked`` when the client is revoked or missing.
    """
    ts = _now(now)
    params = {"id": client_id, "ts": ts}
    async with db.write() as conn:
        async with conn.execute("SELECT revoked_at FROM clients WHERE id = :id", params) as cur:
            row = await cur.fetchone()
        if row is not None and row["revoked_at"] is None:
            return
        # Reason: the same two cleanup statements as revoke_client, so nothing the late write
        # created (a lock or a pending approval) survives the revoke.
        await conn.execute("DELETE FROM task_locks WHERE client_id = :id", params)
        await conn.execute(
            "UPDATE approvals SET status = 'cancelled', resolved_at = :ts"
            " WHERE client_id = :id AND status = 'pending' AND expires_at > :ts",
            params,
        )
    raise WorkspaceError("revoked", "this client has been revoked")


async def list_clients(db: Database) -> list[dict]:
    """List every client for the owner, newest first; no token, hash or prefix is selected.

    The internal WITS row is left out: it is not a client the owner created or can revoke.
    """
    async with db.read().execute(
        "SELECT id, name, scopes, projects, rate_per_min, created_at, expires_at,"
        " last_seen_at, revoked_at FROM clients WHERE id != :internal"
        " ORDER BY created_at DESC, id",
        {"internal": INTERNAL_CLIENT_ID},
    ) as cur:
        rows = await cur.fetchall()
    return [
        {**dict(r), "scopes": r["scopes"].split(), "projects": r["projects"].split()} for r in rows
    ]
