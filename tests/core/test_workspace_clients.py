"""Tests for workspace client tokens, permissions and rate limits (WP 1.2)."""

import hashlib
import logging
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.config import WitsV3Config, WorkspaceSettings
from core.guest_access import issue_guest_token
from core.workspace import clients, ratelimit
from core.workspace.clients import ClientIdentity, create_client, revoke_client, verify_token
from core.workspace.db import open_db
from core.workspace.permissions import check
from core.workspace.ratelimit import AuthFailureLimiter, TokenBuckets
from core.workspace.store import WorkspaceError
from web import workspace_routes

T0 = 1_000_000
OWNER = "owner-secret"
TOKEN_RE = re.compile(r"^wits_[0-9a-f]{8}_[A-Za-z0-9_-]{43}$")
_TOKENS: list[str] = []


@pytest.fixture(autouse=True)
def _no_token_in_logs(caplog):
    """Fail any test whose log output contains a full token minted during it."""
    caplog.set_level(logging.DEBUG)
    _TOKENS.clear()
    yield
    # Reason: after the yield, caplog.records holds only the teardown phase, so the test body's
    # records must be read per phase or a leak in the call phase would go unseen.
    records = [r for phase in ("setup", "call", "teardown") for r in caplog.get_records(phase)]
    text = "\n".join(r.getMessage() for r in records)
    for token in _TOKENS:
        assert token not in text


@pytest.fixture
async def db(tmp_path):
    database = await open_db(tmp_path / "workspace.db")
    yield database
    await database.close()


async def _new(db, name="agent", scopes=("read",), projects=("*",), **kw):
    identity, token = await create_client(db, name, scopes, projects, **kw)
    _TOKENS.append(token)
    return identity, token


async def _rows(db, sql, *args):
    async with db.read().execute(sql, args) as cur:
        return [dict(r) for r in await cur.fetchall()]


class TestCreate:
    async def test_token_shape_and_storage(self, db):
        identity, token = await _new(db, scopes=["read", "write"], projects=["proj-a"], now=T0)
        assert TOKEN_RE.match(token)
        assert identity.scopes == frozenset({"read", "write"})
        (row,) = await _rows(db, "SELECT * FROM clients")
        assert row["token_prefix"] == token[5:13]
        assert row["token_hash"] == hashlib.sha256(token.encode()).hexdigest()
        assert token[14:] not in " ".join(str(v) for v in row.values())
        (audit,) = await _rows(db, "SELECT * FROM audit_log")
        assert audit["action"] == "client.create" and audit["actor"] == "owner"
        assert audit["target"] == identity.id and token not in str(audit)

    @pytest.mark.parametrize(
        "kw",
        [
            {"scopes": []},
            {"scopes": ["admin"]},
            {"scopes": ["decide:resolve"]},
            {"projects": []},
            {"projects": ["Bad_Slug"]},
            {"rate_per_min": 0},
            {"name": "   "},
            {"name": "bot\nINFO forged line"},
            {"name": "bot\tx"},
        ],
    )
    async def test_invalid_input(self, db, kw):
        with pytest.raises(WorkspaceError) as exc:
            await _new(db, **kw)
        assert exc.value.code == "invalid_argument"

    async def test_duplicate_name(self, db):
        await _new(db)
        with pytest.raises(WorkspaceError) as exc:
            await _new(db)
        assert exc.value.code == "name_taken"

    async def test_prefix_collision_retries(self, db, monkeypatch):
        _, first = await _new(db, name="one")
        taken = first[5:13]
        prefixes = iter([taken, taken, "0badc0de"])
        real = clients.secrets.token_hex
        monkeypatch.setattr(
            clients.secrets, "token_hex", lambda n: next(prefixes) if n == 4 else real(n)
        )
        _, second = await _new(db, name="two")
        assert second.startswith("wits_0badc0de_")
        assert await verify_token(db, first) is not None
        assert await verify_token(db, second) is not None


class TestVerify:
    async def test_right_token_passes(self, db):
        identity, token = await _new(db)
        assert await verify_token(db, token) == identity

    async def test_rejections_return_none(self, db):
        _, token = await _new(db, name="live")
        _, expired = await _new(db, name="old", expires_at=T0)
        revoked_id, revoked = await _new(db, name="gone")
        assert await revoke_client(db, revoked_id.id)
        wrong = token[:14] + ("A" if token[14] != "A" else "B") + token[15:]
        bad = [
            wrong,
            "wits_00000000_" + "A" * 43,
            token + "\n",
            token[:-1],
            "wits_" + token[5:13].upper() + token[13:],
            "",
            "wits_" + token[5:13] + "_" + "é" * 43,
            "töken",
            None,
            12345,
        ]
        for candidate in bad:
            assert await verify_token(db, candidate) is None
        assert await verify_token(db, expired, now=T0) is None
        assert await verify_token(db, expired, now=T0 - 1) is not None
        assert await verify_token(db, revoked) is None

    async def test_last_seen_throttled(self, db):
        identity, token = await _new(db)
        sql = "SELECT last_seen_at FROM clients"
        assert (await _rows(db, sql))[0]["last_seen_at"] is None
        await verify_token(db, token, now=T0)
        assert (await _rows(db, sql))[0]["last_seen_at"] == T0
        await verify_token(db, token, now=T0 + 59)
        assert (await _rows(db, sql))[0]["last_seen_at"] == T0
        await verify_token(db, token, now=T0 + 60)
        assert (await _rows(db, sql))[0]["last_seen_at"] == T0 + 60


class TestRevoke:
    async def _seed(self, db, cid):
        async with db.write() as conn:
            await conn.execute(
                "INSERT INTO tasks (id, project, title, created_by, created_at, updated_at)"
                " VALUES ('tsk_1', 'p', 't', ?, 0, 0)",
                (cid,),
            )
            await conn.execute(
                "INSERT INTO task_locks VALUES ('tsk_1', ?, 600, ?, ?)", (cid, T0, T0 + 600)
            )
            for aid, status, exp, resolved in (
                ("apr_live", "pending", T0 + 100, None),
                ("apr_expired", "pending", T0 - 1, None),
                ("apr_done", "approved", T0 + 100, T0 - 5),
            ):
                await conn.execute(
                    "INSERT INTO approvals (id, project, client_id, title, options, status,"
                    " created_at, expires_at, resolved_at) VALUES (?, 'p', ?, 't', '[]', ?, 0, ?,"
                    " ?)",
                    (aid, cid, status, exp, resolved),
                )

    async def test_revoke_in_one_transaction(self, db):
        identity, token = await _new(db)
        other, _ = await _new(db, name="other")
        await self._seed(db, identity.id)
        assert await revoke_client(db, identity.id, now=T0) is True
        (row,) = await _rows(db, "SELECT revoked_at FROM clients WHERE id = ?", identity.id)
        assert row["revoked_at"] == T0
        assert await _rows(db, "SELECT * FROM task_locks") == []
        appr = {
            r["id"]: r for r in await _rows(db, "SELECT id, status, resolved_at FROM approvals")
        }
        assert appr["apr_live"] == {"id": "apr_live", "status": "cancelled", "resolved_at": T0}
        assert appr["apr_expired"]["status"] == "pending"
        assert appr["apr_done"] == {"id": "apr_done", "status": "approved", "resolved_at": T0 - 5}
        assert len(await _rows(db, "SELECT * FROM clients")) == 2
        assert await verify_token(db, token) is None
        assert await revoke_client(db, identity.id, now=T0 + 1) is False
        assert await revoke_client(db, "cli_missing") is False
        actions = await _rows(db, "SELECT action, target, outcome FROM audit_log")
        assert {"action": "client.revoke", "target": identity.id, "outcome": "ok"} in actions
        assert other.id not in [a["target"] for a in actions if a["action"] == "client.revoke"]

    async def test_failed_statement_rolls_back(self, db, monkeypatch):
        identity, _ = await _new(db)
        await self._seed(db, identity.id)

        async def boom(*_a, **_k):
            raise RuntimeError("audit failed")

        monkeypatch.setattr(clients, "_audit_row", boom)
        with pytest.raises(RuntimeError):
            await revoke_client(db, identity.id, now=T0)
        (row,) = await _rows(db, "SELECT revoked_at FROM clients")
        assert row["revoked_at"] is None
        assert len(await _rows(db, "SELECT * FROM task_locks")) == 1


def _ident(scopes, projects=("*",)):
    return ClientIdentity("cli_x", "x", frozenset(scopes), frozenset(projects))


class TestPermissions:
    TOOLS = {
        "read": {"get_project_state"},
        "write": {"claim_task", "append_handoff", "link_artifact"},
        "decide:request": {"request_decision"},
    }
    ALL = set().union(*TOOLS.values())

    @pytest.mark.parametrize("scope", ["read", "write", "decide:request"])
    def test_each_scope_reaches_only_its_tools(self, scope):
        for tool in self.ALL:
            if tool in self.TOOLS[scope]:
                check(_ident([scope]), tool, "p")
            else:
                with pytest.raises(WorkspaceError) as exc:
                    check(_ident([scope]), tool, "p")
                assert exc.value.code == "forbidden_scope"

    @pytest.mark.parametrize("tool", ["resolve_approval", "", "read", "unknown"])
    def test_unknown_tool_denied(self, tool):
        with pytest.raises(WorkspaceError) as exc:
            check(_ident(["read", "write", "decide:request", "admin"]), tool, "p")
        assert exc.value.code == "forbidden_scope"

    def test_projects(self):
        bound = _ident(["read"], ["alpha", "beta"])
        check(bound, "get_project_state", "beta")
        for project in ("gamma", "*", ""):
            with pytest.raises(WorkspaceError) as exc:
                check(bound, "get_project_state", project)
            assert exc.value.code == "forbidden_project"
        check(_ident(["read"], ["*"]), "get_project_state", "anything")


class TestRateLimit:
    def test_bucket_refuses_and_refills(self):
        now = [0.0]
        buckets = TokenBuckets(clock=lambda: now[0])
        assert all(buckets.allow("cli_a", 3) for _ in range(3))
        assert buckets.allow("cli_a", 3) is False
        assert buckets.allow("cli_b", 3) is True
        now[0] = 19.0
        assert buckets.allow("cli_a", 3) is False
        now[0] = 20.0
        assert buckets.allow("cli_a", 3) is True
        now[0] = 1000.0
        assert sum(buckets.allow("cli_a", 3) for _ in range(10)) == 3
        assert buckets.allow("cli_z", 0) is False

    def test_failures_blocked_after_ten(self):
        now = [0.0]
        limiter = AuthFailureLimiter(clock=lambda: now[0])
        assert [limiter.record_failure("1.2.3.4") for _ in range(10)] == [False] * 10
        assert limiter.record_failure("1.2.3.4") is True
        assert limiter.record_failure("5.6.7.8") is False
        now[0] = 61.0
        assert limiter.record_failure("1.2.3.4") is False

    def test_table_capped(self):
        now = [0.0]
        limiter = AuthFailureLimiter(clock=lambda: now[0])
        for i in range(3000):
            now[0] = i / 1000
            limiter.record_failure(f"10.0.{i // 256}.{i % 256}")
            assert len(limiter) <= 1024
        assert len(limiter) == 1024

    def test_module_functions(self):
        ratelimit.reset()
        assert ratelimit.allow_call("cli_m", 1) is True
        assert ratelimit.allow_call("cli_m", 1) is False
        assert not any(ratelimit.auth_failure_blocked("9.9.9.9") for _ in range(10))
        assert ratelimit.auth_failure_blocked("9.9.9.9") is True
        ratelimit.reset()

    async def test_success_never_blocked(self, db):
        # A verified token skips the failure limiter entirely, so a blocked address still
        # authenticates: verification does not consult it.
        ratelimit.reset()
        _, token = await _new(db)
        for _ in range(20):
            ratelimit.auth_failure_blocked("7.7.7.7")
        assert ratelimit.auth_failure_blocked("7.7.7.7") is True
        assert await verify_token(db, token) is not None
        ratelimit.reset()


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace_routes, "_db", None)
    monkeypatch.setattr(workspace_routes, "_lifespan_factories", [])
    monkeypatch.setattr(workspace_routes, "_OPTIONAL_MODULES", ("web.workspace_clients_api",))
    monkeypatch.setenv("WITSV3_WEB_TOKEN", OWNER)
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "guest-signing-secret")
    settings = WorkspaceSettings(enabled=True, db_path=str(tmp_path / "ws.db"))
    config = WitsV3Config(workspace=settings)
    monkeypatch.setattr(workspace_routes, "load_config", lambda *a, **k: config)
    app = FastAPI()
    workspace_routes.register_workspace(app)
    with TestClient(app) as client:
        yield client


def _owner():
    return {"Authorization": f"Bearer {OWNER}"}


def _create(api, name="bot", **extra):
    body = {"name": name, "scopes": ["read", "write"], "projects": ["proj"], **extra}
    resp = api.post("/api/workspace/clients", json=body, headers=_owner())
    if resp.status_code == 201:
        _TOKENS.append(resp.json()["token"])
    return resp


class TestRoutes:
    def test_refused_without_owner_token(self, api):
        created = _create(api, name="victim")
        assert created.status_code == 201
        client_token = created.json()["token"]
        cid = created.json()["client"]["id"]
        guest = issue_guest_token(guest_id="g1", device_id="d1", display_name="Guest")
        for headers in (
            {},
            {"Authorization": "Bearer "},
            {"Authorization": f"Bearer {guest}"},
            {"Authorization": f"Bearer {client_token}"},
        ):
            assert api.get("/api/workspace/clients", headers=headers).status_code == 401
            body = {"name": "x", "scopes": ["read"]}
            resp = api.post("/api/workspace/clients", json=body, headers=headers)
            assert resp.status_code == 401
            assert (
                api.post(f"/api/workspace/clients/{cid}/revoke", headers=headers).status_code == 401
            )
        listed = api.get("/api/workspace/clients", headers=_owner()).json()["clients"]
        assert [c["name"] for c in listed] == ["victim"]
        assert listed[0]["revoked_at"] is None

    def test_owner_create_list_revoke(self, api):
        created = _create(api)
        assert created.status_code == 201
        assert created.headers["cache-control"] == "no-store"
        data = created.json()
        assert TOKEN_RE.match(data["token"])
        assert data["client"]["scopes"] == ["read", "write"]
        token = data["token"]
        resp = api.get("/api/workspace/clients", headers=_owner())
        assert resp.status_code == 200
        digest = hashlib.sha256(token.encode()).hexdigest()
        for secret in (token, token[14:], digest, token[5:13]):
            assert secret not in resp.text
        (entry,) = resp.json()["clients"]
        assert set(entry) == {
            "id",
            "name",
            "scopes",
            "projects",
            "rate_per_min",
            "created_at",
            "expires_at",
            "last_seen_at",
            "revoked_at",
        }
        url = f"/api/workspace/clients/{entry['id']}/revoke"
        assert api.post(url, headers=_owner()).json() == {"id": entry["id"], "revoked": True}
        assert api.post(url, headers=_owner()).status_code == 404
        (entry,) = api.get("/api/workspace/clients", headers=_owner()).json()["clients"]
        assert entry["revoked_at"] is not None

    def test_create_errors(self, api):
        assert _create(api).status_code == 201
        dup = _create(api)
        assert dup.status_code == 409 and dup.json()["detail"]["code"] == "name_taken"
        assert _create(api, name="b", scopes=["admin"]).status_code == 400
        assert _create(api, name="c", token="x").status_code == 422


def test_create_requires_explicit_projects(api):
    body = {"name": "no-projects", "scopes": ["read"]}
    resp = api.post("/api/workspace/clients", json=body, headers=_owner())
    assert resp.status_code == 422
