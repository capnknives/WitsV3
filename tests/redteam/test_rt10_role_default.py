"""Red-team RT-10: any role other than exactly "owner" must be treated as guest.

Today several mappings read "guest" as the only restricted role and everything else
(``"anonymous"``, ``""``, ``None``, ``"Owner"``, ``"owner "``, ``"admin"``...) as owner:

- agents/turn_context.py: TurnContext.for_request only skips global memory for "guest"
- core/guest_access.py: resolve_effective_role / tools_for_role lower-case, strip and
  default empty/None to owner
- agents/routing_classifier.py + wits_control_center_agent.py: ``role != "guest"`` gates
- agents/orchestrator_preflight.py: guest tool block only fires for ``== "guest"``
- web/server.py: /api/chat maps any role outside (owner, guest) to owner

Hole-proving tests are strict xfails asserting the CORRECT behaviour; variants that are
already safe are plain passing pins.
"""

import pytest
from fastapi.testclient import TestClient

from agents.orchestrator_preflight import OrchestratorPreflightMixin
from agents.routing_classifier import RoutingContext, classify_message
from agents.turn_context import TurnContext
from agents.wits_control_center_agent import WitsControlCenterAgent
from core.config import WitsV3Config
from core.guest_access import GUEST_ALLOWED_TOOLS, resolve_effective_role, tools_for_role
from core.tool_context import ToolContext, tool_user_role, use_tool_context
from tests.redteam.test_rt4_role_race import DummyLLM, GatedMemory
from tests.web.test_web_server import FakeSystem
from web.server import create_app

# Every value here is "not exactly owner", so none may get owner rights.
NOT_OWNER = ["anonymous", "", None, "Owner", "owner ", "admin", "OWNER", "totally-unknown"]
# Variants the current code already treats as non-owner at the given level.
SAFE_TOOLS_FOR_ROLE = ["anonymous", "admin", "totally-unknown"]
HOLE_TOOLS_FOR_ROLE = ["", None, "Owner", "owner ", "OWNER"]


def _id(value):
    return repr(value)


def _holes(values, reason):
    return [pytest.param(v, marks=pytest.mark.xfail(strict=True, reason=reason)) for v in values]


# ------------------------------------------------------------ pins: owner and guest


def test_owner_keeps_owner_rights():
    assert resolve_effective_role("owner") == "owner"
    assert tools_for_role("owner") is None
    turn = TurnContext.for_request(user_role="owner")
    assert turn.user_role == "owner" and turn.skip_global_memory_store is False
    ctx = ToolContext(None, "owner", None, None)
    with use_tool_context(ctx):
        assert tool_user_role("owner") == "owner"


def test_guest_stays_guest():
    assert resolve_effective_role("guest") != "owner"
    assert tools_for_role("guest") is not None
    turn = TurnContext.for_request(user_role="guest")
    assert turn.user_role == "guest" and turn.skip_global_memory_store is True


@pytest.mark.parametrize("role", NOT_OWNER, ids=_id)
def test_tool_context_fails_closed(role):
    # Already safe: tool_user_role collapses anything but "owner" to guest.
    with use_tool_context(ToolContext(None, role, None, None)):
        assert tool_user_role("owner") == "guest"


# ------------------------------------------------------------ core/guest_access.py


@pytest.mark.parametrize("role", SAFE_TOOLS_FOR_ROLE, ids=_id)
def test_tools_for_role_unknown_gets_guest_tools(role):
    # Already safe: unrecognised names fall back to the guest allow-list.
    assert tools_for_role(role) == GUEST_ALLOWED_TOOLS


@pytest.mark.parametrize(
    "role",
    _holes(HOLE_TOOLS_FOR_ROLE, "RT-10: tools_for_role lower-cases/strips/defaults to all tools"),
    ids=_id,
)
def test_tools_for_role_only_exact_owner_gets_all_tools(role):
    assert tools_for_role(role) is not None


@pytest.mark.parametrize(
    "role",
    _holes(NOT_OWNER, "RT-10: resolve_effective_role returns owner for any non-guest role"),
    ids=_id,
)
def test_effective_role_is_not_owner(role):
    assert resolve_effective_role(role) != "owner"


# ------------------------------------------------------------ agents/turn_context.py


@pytest.mark.parametrize(
    "role",
    _holes(NOT_OWNER, "RT-10: TurnContext only skips global memory for the literal guest"),
    ids=_id,
)
def test_turn_context_non_owner_is_guest(role):
    turn = TurnContext.for_request(user_role=role)
    assert turn.skip_global_memory_store is True
    assert turn.user_role == "guest"


# ------------------------------------------------------------ routing / orchestrator


@pytest.mark.parametrize(
    "role",
    _holes(NOT_OWNER, 'RT-10: routing_classifier gates guests with `role != "guest"`'),
    ids=_id,
)
def test_classifier_denies_direct_file_read(role):
    ctx = RoutingContext(message="please read the file secrets.txt", user_role=role)
    assert classify_message(ctx).destination != "read_file_direct"


def test_classifier_guest_denied_direct_file_read():
    ctx = RoutingContext(message="please read the file secrets.txt", user_role="guest")
    assert classify_message(ctx).destination != "read_file_direct"


@pytest.mark.parametrize("role", ["anonymous", "", None, "admin", "totally-unknown"], ids=_id)
def test_classifier_unknown_role_not_owner_only_route(role):
    # Already safe: owner-only routes use `role == "owner"`.
    ctx = RoutingContext(message="list the recurring bugs zebra", user_role=role)
    assert classify_message(ctx).destination not in (
        "guest_chat_history",
        "guest_profile",
        "knowledge_log",
    )


class _Preflight(OrchestratorPreflightMixin):
    config = None


@pytest.mark.parametrize(
    "role",
    _holes(NOT_OWNER, "RT-10: preflight blocks owner-only tools only for the literal guest"),
    ids=_id,
)
def test_preflight_blocks_owner_tool(role):
    state = {"user_role": role, "goal": "hello"}
    blocked = _Preflight()._preflight_tool_call("write_file", {"file_path": "x"}, state)
    assert blocked and "not available for guest" in blocked


def test_preflight_blocks_owner_tool_for_guest():
    state = {"user_role": "guest", "goal": "hello"}
    blocked = _Preflight()._preflight_tool_call("write_file", {"file_path": "x"}, state)
    assert blocked and "not available for guest" in blocked


# ------------------------------------------------------------ control center turn


@pytest.fixture
def agent_and_memory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mem = GatedMemory()
    agent = WitsControlCenterAgent(
        agent_name="RT10", config=WitsV3Config(), llm_interface=DummyLLM(), memory_manager=mem
    )
    agent.has_enhanced_capabilities = False
    agent.meta_reasoning = None
    return agent, mem


REMEMBER = "remember that my safe code is zebra-seven-seven"


async def _remembered(agent, mem, role):
    _ = [e async for e in agent.run(REMEMBER, session_id="s", user_role=role)]
    return any("zebra-seven-seven" in text for text in mem.added)


async def test_owner_turn_remembers(agent_and_memory):
    agent, mem = agent_and_memory
    assert await _remembered(agent, mem, "owner") is True


async def test_guest_turn_does_not_remember(agent_and_memory):
    agent, mem = agent_and_memory
    assert await _remembered(agent, mem, "guest") is False


@pytest.mark.parametrize(
    "role",
    _holes(NOT_OWNER, "RT-10: control center runs the remember rule for any non-guest role"),
    ids=_id,
)
async def test_non_owner_turn_does_not_write_global_memory(agent_and_memory, role):
    agent, mem = agent_and_memory
    assert await _remembered(agent, mem, role) is False


# ------------------------------------------------------------ web chat route


def _chat_client(tmp_path, monkeypatch, role):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "sekrit")
    monkeypatch.chdir(tmp_path)
    system = FakeSystem(tmp_path)
    system.config.web_ui.require_auth = True
    # Reason: with a token set, real auth only ever yields owner or guest for /api/chat, so
    # the stray role is injected at the resolve_auth seam to prove the chat call's mapping.
    import web.server as server

    monkeypatch.setattr(
        server,
        "resolve_auth",
        lambda request, config: {"role": role, "guest": None, "token": "t", "allow": True},
    )
    return TestClient(create_app(system)), system


def test_chat_route_owner_role_stays_owner(tmp_path, monkeypatch):
    client, system = _chat_client(tmp_path, monkeypatch, "owner")
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 200
    assert system.control_center.calls[-1]["user_role"] == "owner"


def test_chat_route_guest_role_stays_guest(tmp_path, monkeypatch):
    client, system = _chat_client(tmp_path, monkeypatch, "guest")
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 200
    assert system.control_center.calls[-1]["user_role"] == "guest"


@pytest.mark.parametrize(
    "role",
    _holes(
        ["anonymous", "", "Owner", "owner ", "admin", "OWNER", "totally-unknown"],
        "RT-10: web/server.py maps any role outside (owner, guest) to owner for chat",
    ),
    ids=_id,
)
def test_chat_route_non_owner_role_is_not_passed_as_owner(tmp_path, monkeypatch, role):
    client, system = _chat_client(tmp_path, monkeypatch, role)
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 200
    assert system.control_center.calls[-1]["user_role"] != "owner"
