"""RT-10b: role normalisation helper and the trusted entry points that must pass "owner".

A missing role is now a guest everywhere, so each trusted caller (auth-off web UI, CLI,
scheduled jobs, golden runner) must name the owner explicitly; guest tokens stay guest.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.config import WitsV3Config
from core.roles import is_owner, normalize_role
from core.schemas import StreamData
from tests.web.test_web_server import FakeSystem
from web.server import create_app

NOT_OWNER = [None, "", "guest", "anonymous", "Owner", "owner ", " owner", "OWNER", "admin", 1]


class RecordingControl:
    """Records the raw kwargs each turn was started with (no permissive defaults)."""

    def __init__(self):
        self.kwargs: list[dict] = []

    async def run(self, *args, **kwargs):
        self.kwargs.append(kwargs)
        yield StreamData(type="result", content="ok", source="fake")


# ------------------------------------------------------------ helper


def test_normalize_role_exact_owner_only():
    assert normalize_role("owner") == "owner"
    assert is_owner("owner") is True


@pytest.mark.parametrize("role", NOT_OWNER, ids=repr)
def test_normalize_role_everything_else_is_guest(role):
    assert normalize_role(role) == "guest"
    assert is_owner(role) is False


# ------------------------------------------------------------ web chat route


def _chat_role(client, system, **headers) -> object:
    system.control_center = RecordingControl()
    res = client.post("/api/chat", json={"message": "hi"}, headers=headers)
    assert res.status_code == 200
    return system.control_center.kwargs[-1]["user_role"]


def test_auth_off_local_mode_chat_is_owner(tmp_path, monkeypatch):
    monkeypatch.delenv("WITSV3_WEB_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    system = FakeSystem(tmp_path)
    system.config.web_ui.require_auth = False
    # Reason: the middleware sets auth_role="owner" itself in this mode; no fallback.
    assert _chat_role(TestClient(create_app(system)), system) == "owner"


def test_require_auth_without_token_chat_is_owner(tmp_path, monkeypatch):
    monkeypatch.delenv("WITSV3_WEB_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    system = FakeSystem(tmp_path)
    system.config.web_ui.require_auth = True
    # Reason: documented local setup (web/guest_auth.py resolve_auth, "no token configured"
    # branch; server.py auth middleware comment): with guests off and no WITSV3_WEB_TOKEN
    # set, require_auth has no token to check, so resolve_auth returns role "owner"
    # explicitly. Anyone who can reach the port is the owner; README says set a token.
    assert _chat_role(TestClient(create_app(system)), system) == "owner"


def test_owner_token_chat_is_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "sekrit")
    monkeypatch.chdir(tmp_path)
    system = FakeSystem(tmp_path)
    client = TestClient(create_app(system))
    assert _chat_role(client, system, Authorization="Bearer sekrit") == "owner"


def test_guest_token_chat_is_guest(tmp_path, monkeypatch):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "owner-sekrit")
    monkeypatch.setenv("WITSV3_GUEST_INVITE", "family-code")
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "guest-signing-secret-xyz")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var" / "data").mkdir(parents=True)
    system = FakeSystem(tmp_path)
    system.config.web_ui.guest_access.enabled = True
    client = TestClient(create_app(system))
    reg = client.post(
        "/api/guest/register",
        json={
            "invite_code": "family-code",
            "display_name": "Alex",
            "device_id": "device-abc-12345",
            "age_band": "teen",
        },
    )
    token = reg.json()["guest_token"]
    assert _chat_role(client, system, Authorization=f"Bearer {token}") == "guest"


# ------------------------------------------------------------ CLI and scheduled jobs


async def test_cli_turn_passes_owner():
    from run import WitsV3System

    system = WitsV3System(WitsV3Config())
    system.control_center = RecordingControl()
    await system.process_user_input("hello", session_id="cli-1")
    assert system.control_center.kwargs[-1]["user_role"] == "owner"


async def test_scheduled_self_repair_passes_owner():
    from run import WitsV3System

    system = WitsV3System(WitsV3Config())
    system.self_repair_agent = RecordingControl()
    await system._run_scheduled_self_repair()
    assert system.self_repair_agent.kwargs[-1]["user_role"] == "owner"


# ------------------------------------------------------------ golden runner


def test_golden_runner_passes_owner():
    from core.golden import GoldenCase

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("golden_run", root / "scripts/golden_run.py")
    golden_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden_run)
    control = RecordingControl()
    case = GoldenCase(id="hi", message="hello", tags=["chat"])
    asyncio.run(golden_run.run_once(control, case))
    assert control.kwargs[-1]["user_role"] == "owner"
