"""RT-10 review rework: a guest from any source wins, and no owner default is left.

- explicit ``user_role="guest"`` inside a bound owner turn stays a guest (and vice versa)
- an abandoned owner generator never leaks its role into the next turn on the same agent
- concurrent owner and guest turns never cross
- PII helpers redact when the role is omitted
- the legacy orchestrator path gives tools the guest's role, not the tools' owner default
- a non-owner chat request without a guest record never shares the owner's sessions
- end to end, an "anonymous" web chat role is refused an owner-only tool
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents.base_agent import BaseAgent
from agents.llm_driven_orchestrator import LLMDrivenOrchestrator
from agents.turn_context import (
    TurnContext,
    bind_turn,
    bind_turn_if_unbound,
    current_turn,
    turn_for,
)
from core.config import WitsV3Config
from core.guest_access import GUEST_ALLOWED_TOOLS, chat_storage_session_id, tools_for_role
from core.pii_redaction import (
    is_pii_redaction_enabled,
    maybe_redact_for_guest,
    maybe_redact_for_storage,
)
from core.tool_context import tool_read_role, tool_user_role
from tests.agents.test_native_tool_context import _make, _result
from tests.agents.test_turn_context import DummyLLM, RecordingMemory
from tests.web.test_web_server import FakeSystem, _parse_sse
from web.server import create_app

ROOT = Path(__file__).resolve().parents[2]
WHO = {"name": "whoami", "arguments": {}}


class SubAgent(BaseAgent):
    """Decorated like every agent ``run``; reports the role it ran as."""

    @bind_turn_if_unbound
    async def run(self, user_input, conversation_history=None, session_id=None, **kwargs):
        await asyncio.sleep(0)
        await self.store_memory(content=user_input)
        yield self.stream_result(turn_for(self).user_role)


async def _roles(agent, *args, **kwargs) -> list[str]:
    return [e.content async for e in agent.run(*args, **kwargs)]


# ------------------------------------------------------------ finding 1: orchestrators


async def test_nested_owner_turn_explicit_guest_stays_guest():
    orch, _, registry = _make({"who": WHO})
    with bind_turn(TurnContext.for_request(user_role="owner")):
        out = await _result(orch, "who", session_id="n1", user_role="guest")
    assert "not available for guest" in out
    assert registry.executed == []


async def test_nested_guest_turn_explicit_owner_stays_guest():
    orch, _, registry = _make({"who": WHO})
    with bind_turn(TurnContext.for_request(user_role="guest")):
        out = await _result(orch, "who", session_id="n2", user_role="owner")
    assert "not available for guest" in out
    assert registry.executed == []


async def test_nested_owner_turn_without_argument_inherits_owner():
    orch, _, registry = _make({"who": WHO})
    with bind_turn(TurnContext.for_request(user_role="owner")):
        out = await _result(orch, "who", session_id="n3")
    assert out.endswith("role=owner") and registry.executed == ["whoami"]


async def test_unbound_orchestrator_run_without_role_is_guest():
    orch, _, _ = _make({"who": WHO})
    assert current_turn() is None
    assert "not available for guest" in await _result(orch, "who", session_id="n4")


async def test_abandoned_owner_generator_then_guest_turn_is_guest():
    orch, _, _ = _make({"who": WHO})
    owner_gen = orch.run("who a", session_id="a", user_role="owner")
    await owner_gen.__anext__()  # left suspended mid-turn, never closed
    assert current_turn() is None
    out = await _result(orch, "who b", session_id="b", user_role="guest")
    assert "not available for guest" in out
    await owner_gen.aclose()


async def test_concurrent_owner_and_guest_turns_do_not_cross():
    orch, _, _ = _make({"who": WHO})
    owner, guest = await asyncio.gather(
        _result(orch, "who a", session_id="a", user_role="owner"),
        _result(orch, "who b", session_id="b", user_role="guest"),
    )
    assert owner.endswith("session=a role=owner")
    assert "not available for guest" in guest


async def test_sub_agent_nested_guest_in_owner_turn_writes_no_memory():
    memory = RecordingMemory()
    sub = SubAgent("Sub", WitsV3Config(), DummyLLM(), memory_manager=memory)
    with bind_turn(TurnContext.for_request(user_role="owner")):
        assert await _roles(sub, "guest secret", user_role="guest") == ["guest"]
        assert current_turn().user_role == "owner"  # the outer turn is untouched
    assert memory.added == []


async def test_sub_agent_concurrent_turns_do_not_cross():
    memory = RecordingMemory()
    sub = SubAgent("Sub", WitsV3Config(), DummyLLM(), memory_manager=memory)
    results = await asyncio.gather(
        *[_roles(sub, f"m{i}", user_role=("owner" if i % 2 else "guest")) for i in range(6)]
    )
    assert results == [["guest"], ["owner"]] * 3
    assert sorted(memory.added) == ["m1", "m3", "m5"]


def test_web_only_starts_turns_through_the_control_center():
    """Proves no request path reaches turn_for's unbound owner default: web code only
    starts agent work via system.control_center.run(), which always binds a turn."""
    calls = []
    for path in sorted((ROOT / "web").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "run" and isinstance(node.func.value, ast.Attribute):
                    calls.append((path.name, ast.unparse(node.func)))
    agent_calls = [c for c in calls if "session_manager" not in c[1]]
    assert agent_calls == [("server.py", "system.control_center.run")]


async def test_background_self_repair_runs_as_owner(monkeypatch):
    from agents import self_repair_agent
    from agents.background_agent import BackgroundAgent

    seen: list[str] = []

    @bind_turn_if_unbound
    async def fake_run(self, user_input, conversation_history=None, session_id=None, **kw):
        seen.append(turn_for(self).user_role)
        yield self.stream_result("ok")

    monkeypatch.setattr(self_repair_agent.SelfRepairAgent, "run", fake_run)
    config = WitsV3Config()
    config.self_repair.enabled = True
    agent = BackgroundAgent("bg", config, DummyLLM(), None, None)
    await agent._run_self_repair({})
    assert seen == ["owner"]


# ------------------------------------------------------------ finding 2: PII


def _pii_config() -> WitsV3Config:
    config = WitsV3Config()
    config.security.pii_redaction.enabled = True
    config.security.pii_redaction.redact_before_store = True
    return config


def test_omitted_role_redacts():
    config = _pii_config()
    assert "bob@test.org" not in maybe_redact_for_storage("bob@test.org", config)
    assert is_pii_redaction_enabled(config) is True


@pytest.mark.parametrize("role", [None, "", "Owner", "anonymous", "family_adult"], ids=repr)
def test_non_owner_role_redacts(role):
    config = _pii_config()
    assert "bob@test.org" not in maybe_redact_for_storage("bob@test.org", config, role=role)
    assert "bob@test.org" not in maybe_redact_for_guest("bob@test.org", config, role=role)


# ------------------------------------------------------------ tools_for_role


def test_tools_for_role_absent_or_star_role_is_never_all_tools(monkeypatch):
    import core.guest_policy_loader as loader

    monkeypatch.setattr(loader, "policy_roles", lambda: {"helper": {"allowed_tools": ["*"]}})
    assert tools_for_role("helper") == GUEST_ALLOWED_TOOLS
    assert tools_for_role("not-in-policy") == GUEST_ALLOWED_TOOLS
    assert tools_for_role("owner") is None


# ------------------------------------------------------------ legacy tool path


class _ProbeRegistry:
    def __init__(self):
        self.seen: list[tuple] = []

    def get_tool(self, name):
        return object()

    async def execute_tool(self, name, **kwargs):
        self.seen.append((tool_user_role(kwargs.get("user_role", "owner")), tool_read_role()))
        return "ok"


@pytest.mark.parametrize(
    ("role", "profile", "expected"),
    [
        ("owner", {}, ("owner", "owner")),
        ("guest", {}, ("guest", "guest")),
        ("guest", {"rbac_role": "family_adult"}, ("guest", "family_adult")),
        ("guest", {"rbac_role": "owner"}, ("guest", "guest")),
        (None, {}, ("guest", "guest")),
    ],
)
async def test_legacy_tool_call_sees_turn_role_not_tool_default(role, profile, expected):
    registry = _ProbeRegistry()
    orch = LLMDrivenOrchestrator("T", WitsV3Config(), DummyLLM(), None, registry)
    state = {"user_role": role, "guest_profile": profile, "goal": "", "observations": []}
    # Reason: the model's own user_role="owner" must never reach the tool.
    await orch._call_tool("list_directory", {"directory_path": ".", "user_role": "owner"}, state)
    assert registry.seen == [expected]


def test_family_guest_keeps_policy_read_tools_and_is_not_owner():
    orch = LLMDrivenOrchestrator("T", WitsV3Config(), DummyLLM())
    state = {"user_role": "guest", "guest_profile": {"rbac_role": "family_adult"}, "goal": "x"}
    allowed = orch._preflight_tool_call("read_file", {"file_path": "a.txt"}, state)
    assert not allowed or "not available for guest" not in allowed
    blocked = orch._preflight_tool_call("write_file", {"file_path": "a.txt"}, state)
    assert blocked and "not available for guest" in blocked


# ------------------------------------------------------------ finding 5 + 7: web chat


def test_storage_key_for_roleless_guest_is_isolated():
    a = chat_storage_session_id("s1", None, "anonymous")
    b = chat_storage_session_id("s1", None, "guest")
    assert a.startswith("guest:unregistered-") and a != b
    assert chat_storage_session_id("s1", None, "owner") == "s1"
    assert chat_storage_session_id("s1", {"guest_id": "g"}, "guest") == "guest:g:s1"


def _seam_client(tmp_path, monkeypatch, role, control=None):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "sekrit")
    monkeypatch.chdir(tmp_path)
    system = FakeSystem(tmp_path)
    if control is not None:
        system.control_center = control
    import web.server as server

    monkeypatch.setattr(
        server,
        "resolve_auth",
        lambda request, config: {"role": role, "guest": None, "token": "t", "allow": True},
    )
    return TestClient(create_app(system)), system


def test_roleless_guest_chat_does_not_touch_owner_session(tmp_path, monkeypatch):
    client, system = _seam_client(tmp_path, monkeypatch, "guest")
    from core.schemas import ConversationHistory

    owner_conv = ConversationHistory(session_id="owner-s")
    owner_conv.add_message("user", "owner secret")
    system.session_histories["owner-s"] = owner_conv
    res = client.post("/api/chat", json={"message": "hi", "session_id": "owner-s"})
    assert res.status_code == 200
    assert [m.content for m in system.session_histories["owner-s"].messages] == ["owner secret"]
    assert system.control_center.calls[-1]["session_id"].startswith("guest:unregistered-")


@pytest.mark.parametrize("role", ["anonymous", "", "Owner"], ids=repr)
def test_web_chat_non_owner_role_is_refused_owner_only_tool(tmp_path, monkeypatch, role):
    orch, _, registry = _make({"hi": WHO})
    client, _ = _seam_client(tmp_path, monkeypatch, role, control=orch)
    res = client.post("/api/chat", json={"message": "hi"})
    results = [str(d.get("content") or d.get("final") or "") for _, d in _parse_sse(res.text)]
    assert registry.executed == []
    assert any("not available for guest" in r for r in results)


def test_web_chat_owner_role_runs_owner_only_tool(tmp_path, monkeypatch):
    orch, _, registry = _make({"hi": WHO})
    client, _ = _seam_client(tmp_path, monkeypatch, "owner", control=orch)
    client.post("/api/chat", json={"message": "hi"})
    assert registry.executed == ["whoami"]


# ------------------------------------------------------------ playbooks (no preflight)


class _RecordingRegistry:
    def __init__(self):
        self.executed: list[str] = []

    async def execute_tool(self, name, **kwargs):
        self.executed.append(name)
        return "ok"


def _playbook():
    from agents.playbook_executor import PlaybookExecutor

    registry = _RecordingRegistry()
    executor = PlaybookExecutor("pb", WitsV3Config(), DummyLLM(), None, registry)
    return executor, registry


@pytest.mark.parametrize("role", ["guest", None, "anonymous"], ids=repr)
async def test_guest_save_conversation_playbook_refused(role):
    executor, registry = _playbook()
    kwargs = {"playbook_id": "save_conversation"}
    if role is not None:
        kwargs["user_role"] = role
    events = [e async for e in executor.run("save our conversation as x.txt", **kwargs)]
    assert registry.executed == []
    assert "available for guest" in events[-1].content


async def test_family_adult_doc_qa_playbook_keeps_policy_rights():
    executor, registry = _playbook()
    profile = {"rbac_role": "family_adult"}
    gen = executor.run(
        "what do my docs say", playbook_id="doc_qa", user_role="guest", guest_profile=profile
    )
    _ = [e async for e in gen]
    assert registry.executed == ["document_search"]


async def test_owner_save_conversation_playbook_runs():
    executor, registry = _playbook()
    gen = executor.run("save our conversation as x.txt", playbook_id="save_conversation")
    with bind_turn(TurnContext.for_request(user_role="owner")):
        _ = [e async for e in gen]
    assert "read_conversation_history" in registry.executed
