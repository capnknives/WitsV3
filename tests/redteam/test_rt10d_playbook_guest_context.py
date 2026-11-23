"""RT-10 second review rework: guests keep their own policy on every tool path.

- playbook tool calls run inside the turn's guest ToolContext (policy read roots)
- the guest ToolContext is decided from the bound turn, not from ReAct state being present
- the guest profile comes from the bound turn; a nested call cannot widen it
- the native loop's ToolContext carries the same read_role as the other paths
- the owner path is unchanged (no redaction, owner read roots)
- a one-off unregistered-guest chat key never grows session_histories
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agents.playbook_executor as playbook_module
from agents.llm_driven_orchestrator import LLMDrivenOrchestrator
from agents.playbook_executor import PlaybookExecutor
from agents.turn_context import TurnContext, bind_turn, call_turn, guest_tool_context
from core.config import WitsV3Config
from core.tool_context import current_tool_context, tool_read_role, use_tool_context
from tests.agents.test_native_tool_context import _make, _result
from tests.agents.test_turn_context import DummyLLM, RecordingMemory
from tests.redteam.test_rt10c_guest_wins import SubAgent, _seam_client
from tools.file_tools import FileReadTool, ListDirectoryTool

KID = {"age_band": "child"}
ADULT = {"rbac_role": "family_adult"}


class _FileRegistry:
    """Runs the real file tools and records what each call returned."""

    def __init__(self):
        self.tools = {"read_file": FileReadTool(), "list_directory": ListDirectoryTool()}
        self.results: list[str] = []

    def get_tool(self, name):
        return self.tools.get(name)

    async def execute_tool(self, name, **kwargs):
        result = await self.tools[name].execute(**kwargs)
        self.results.append(str(result))
        return result


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Owner-only documents folder, an extra root (owner + family_adult) and a project file."""
    import core.config

    config = WitsV3Config()
    config.folders.base = str(tmp_path / "base")
    docs = tmp_path / "base" / "documents"
    extra = tmp_path / "extra"
    for folder in (docs, extra):
        folder.mkdir(parents=True)
        (folder / "secret.txt").write_text("SECRET-" + folder.name, encoding="utf-8")
    monkeypatch.setattr(core.config, "load_config", lambda *a, **k: config)
    monkeypatch.setenv("WITSV3_READ_ROOTS", str(extra))
    return {
        "docs": docs,
        "extra": extra,
        "project": (Path(__file__).resolve().parents[2] / "README.md"),
    }


async def _run_playbook(monkeypatch, target, file_path, **kwargs) -> list[str]:
    spec = {
        "id": "probe",
        "steps": [
            {"tool": "list_directory", "args": {"directory_path": str(target)}},
            {"tool": "read_file", "args": {"file_path": str(file_path)}},
        ],
        "synthesis": "done",
    }
    monkeypatch.setattr(playbook_module, "get_playbook", lambda _id: spec)
    registry = _FileRegistry()
    executor = PlaybookExecutor("pb", WitsV3Config(), DummyLLM(), None, registry)
    _ = [e async for e in executor.run("x", playbook_id="probe", **kwargs)]
    return registry.results


def _refused(results):
    return len(results) == 2 and all(r.startswith("Error") for r in results)


def _read_ok(results, marker):
    return len(results) == 2 and marker in results[1] and not results[0].startswith("Error")


# ------------------------------------------------------------ finding 1: playbooks


async def test_family_adult_playbook_reads_policy_root_not_owner_root(roots, monkeypatch):
    kw = {"user_role": "guest", "guest_profile": ADULT}
    extra, docs = roots["extra"], roots["docs"]
    assert _read_ok(await _run_playbook(monkeypatch, extra, extra / "secret.txt", **kw), "SECRET")
    assert _refused(await _run_playbook(monkeypatch, docs, docs / "secret.txt", **kw))


async def test_family_kid_playbook_reads_project_not_owner_root(roots, monkeypatch):
    kw = {"user_role": "guest", "guest_profile": KID}
    project = roots["project"]
    assert _read_ok(await _run_playbook(monkeypatch, project.parent, project, **kw), "WitsV3")
    docs = roots["docs"]
    assert _refused(await _run_playbook(monkeypatch, docs, docs / "secret.txt", **kw))


async def test_plain_guest_playbook_reads_guest_roots_only(roots, monkeypatch):
    import core.guest_access as guest_access

    # Reason: a plain guest has no file tools by policy; widen only the allow-list so the
    # read roots are what is tested.
    monkeypatch.setattr(guest_access, "tools_for_role", lambda r: {"read_file", "list_directory"})
    project, docs = roots["project"], roots["docs"]
    kw = {"user_role": "guest"}
    assert _read_ok(await _run_playbook(monkeypatch, project.parent, project, **kw), "WitsV3")
    assert _refused(await _run_playbook(monkeypatch, docs, docs / "secret.txt", **kw))


async def test_owner_playbook_unchanged(roots, monkeypatch):
    docs = roots["docs"]
    with bind_turn(TurnContext.for_request(user_role="owner")):
        results = await _run_playbook(monkeypatch, docs, docs / "secret.txt")
    assert _read_ok(results, "SECRET-documents")


# ------------------------------------------------------------ finding 2: no state


class _ProbeRegistry:
    def __init__(self):
        self.seen: list[tuple] = []

    def get_tool(self, name):
        return object()

    async def execute_tool(self, name, **kwargs):
        ctx = current_tool_context()
        self.seen.append((ctx.user_role if ctx else None, tool_read_role()))
        return "ok"


async def test_stateless_tool_call_in_guest_turn_is_guest():
    registry = _ProbeRegistry()
    orch = LLMDrivenOrchestrator("T", WitsV3Config(), DummyLLM(), None, registry)
    with bind_turn(TurnContext.for_request(user_role="guest", guest_profile=KID)):
        assert orch._legacy_guest_tool_context(None).read_role == "family_kid"
        await orch._call_tool("list_directory", {"directory_path": "."}, None)
    with bind_turn(TurnContext.for_request(user_role="owner")):
        assert orch._legacy_guest_tool_context(None) is None
        await orch._call_tool("list_directory", {"directory_path": "."}, None)
    assert registry.seen == [("guest", "family_kid"), (None, "owner")]


async def test_existing_guest_context_is_kept_but_owner_context_is_replaced():
    registry = _ProbeRegistry()
    orch = LLMDrivenOrchestrator("T", WitsV3Config(), DummyLLM(), None, registry)
    turn = TurnContext.for_request(user_role="guest", guest_profile=ADULT)
    owner_ctx = guest_tool_context(TurnContext())
    assert owner_ctx is None  # an owner turn gets no guest context
    native_ctx = guest_tool_context(turn, session_id="s")
    with bind_turn(turn), use_tool_context(native_ctx):
        await orch._call_tool("x", {}, None)
        assert current_tool_context() is native_ctx
    assert registry.seen == [("guest", "family_adult")]


# ------------------------------------------------------------ finding 3: profile from turn


async def test_nested_playbook_without_profile_keeps_caller_policy():
    registry = _ProbeRegistry()
    executor = PlaybookExecutor("pb", WitsV3Config(), DummyLLM(), None, registry)
    with bind_turn(TurnContext.for_request(user_role="guest", guest_profile=KID)):
        _ = [e async for e in executor.run("x", playbook_id="codebase_tour", user_role="guest")]
        # Reason: a nested call naming a wider profile must not get it either.
        wider = {"user_role": "guest", "guest_profile": ADULT}
        assert call_turn(wider).guest_profile == KID
    assert registry.seen and set(registry.seen) == {("guest", "family_kid")}


async def test_owner_turn_explicit_guest_takes_the_named_profile():
    with bind_turn(TurnContext.for_request(user_role="owner")):
        assert call_turn({"user_role": "guest", "guest_profile": KID}).guest_profile == KID


async def test_legacy_state_profile_comes_from_turn_not_kwarg():
    orch = LLMDrivenOrchestrator("T", WitsV3Config(), DummyLLM(), None, _ProbeRegistry())
    seen = {}

    async def fake_loop(state, session_id):
        seen.update(state)
        yield orch.stream_result("done")

    orch._execute_react_loop = fake_loop
    with bind_turn(TurnContext.for_request(user_role="guest", guest_profile=KID)):
        _ = [e async for e in orch.run("hi", user_role="guest", guest_profile=ADULT)]
    assert seen["guest_profile"] == KID


class _DatetimeProbe:
    name = "datetime"

    def get_schema(self):
        return {"type": "object", "properties": {}}

    def get_llm_description(self):
        return {"description": "datetime"}

    async def execute(self, **kwargs):
        ctx = current_tool_context()
        return f"role={ctx.user_role} read_role={ctx.read_role}"


@pytest.mark.parametrize(
    ("role", "profile", "expected"),
    [
        ("guest", ADULT, "role=guest read_role=family_adult"),
        ("guest", KID, "role=guest read_role=family_kid"),
        ("owner", None, "role=owner read_role=None"),
    ],
)
async def test_native_tool_context_has_read_role(role, profile, expected):
    orch, _, registry = _make({"time": {"name": "datetime", "arguments": {}}})
    registry.tools["datetime"] = _DatetimeProbe()
    with bind_turn(TurnContext.for_request(user_role=role, guest_profile=profile)):
        assert (await _result(orch, "time please", session_id="n")).endswith(expected)


# ------------------------------------------------------------ PII: owner path


async def test_owner_memory_is_not_redacted():
    config = WitsV3Config()
    config.security.pii_redaction.enabled = True
    config.security.pii_redaction.redact_before_store = True
    memory = RecordingMemory()
    agent = SubAgent("A", config, DummyLLM(), memory_manager=memory)
    with bind_turn(TurnContext.for_request(user_role="owner")):
        await agent.store_memory(content="mail bob@test.org")
    assert memory.added == ["mail bob@test.org"]


# ------------------------------------------------------------ finding 5: one-off key


def test_one_off_guest_key_does_not_grow_session_histories(tmp_path, monkeypatch):
    client, system = _seam_client(tmp_path, monkeypatch, "guest")
    before = set(system.session_histories)
    for _ in range(3):
        assert client.post("/api/chat", json={"message": "hi", "session_id": "s"}).status_code
    assert set(system.session_histories) == before
    assert system.control_center.calls[-1]["session_id"].startswith("guest:unregistered-")
