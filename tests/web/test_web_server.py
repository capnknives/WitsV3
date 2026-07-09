"""Tests for the WitsV3 web UI server."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.config import WitsV3Config
from core.schemas import StreamData
from web.server import create_app

# ------------------------------------------------------------------ fakes


class FakeControlCenter:
    def __init__(self):
        self.calls = []

    async def run(self, user_input, conversation_history=None, session_id=None, **kwargs):
        self.calls.append(
            {
                "user_input": user_input,
                "session_id": session_id,
                "user_role": kwargs.get("user_role", "owner"),
            }
        )
        yield StreamData(type="thinking", content="pondering", source="wcca")
        yield StreamData(type="tool_call", content="calculator(2+2)", source="orchestrator")
        yield StreamData(type="result", content="The answer is 4.", source="orchestrator")


class FakeTool(SimpleNamespace):
    pass


class FakeToolRegistry:
    def __init__(self, tmp_docs):
        async def ingest_execute(**kwargs):
            return {"success": True, "files_scanned": 1, "files_ingested": 1, "chunks_added": 2}

        self.tools = {
            "calculator": FakeTool(name="calculator", description="Does math"),
            "ingest_documents": FakeTool(
                name="ingest_documents", description="Ingest docs", execute=ingest_execute
            ),
        }

    def get_tool(self, name):
        return self.tools.get(name)


class FakeMemoryManager:
    async def search_memory(self, query_text, limit=5, min_relevance=0.0, filter_dict=None):
        seg = SimpleNamespace(
            type="DOCUMENT_CHUNK",
            source="notes.md",
            content=SimpleNamespace(text=f"match for {query_text}", tool_output=None),
            relevance_score=0.87,
        )
        return [seg][:limit]

    async def get_recent_memory(self, limit=10, filter_dict=None):
        # Produce stable fake segments for pagination tests.
        total = 2
        items = []
        for i in range(min(limit, total)):
            items.append(
                SimpleNamespace(
                    id=f"seg-{i}",
                    timestamp=datetime(2026, 1, 1, 0, 0, i, tzinfo=timezone.utc),
                    type="DOCUMENT_CHUNK",
                    source="notes.md",
                    content=SimpleNamespace(
                        text=f"recent match {i}", tool_output=None, tool_name=None
                    ),
                    metadata={"file_path": "notes.md"},
                )
            )
        return items

    async def delete_segments(self, filter_dict):
        # Pretend everything matches (tests only validate the endpoint contract).
        return 2


class FakeSystem:
    def __init__(self, tmp_path):
        self.config = WitsV3Config()
        self.config.document_rag.documents_path = str(tmp_path / "documents")
        self.control_center = FakeControlCenter()
        self.tool_registry = FakeToolRegistry(tmp_path)
        self.memory_manager = FakeMemoryManager()
        self.session_histories = {}


@pytest.fixture
def client_noauth(tmp_path, monkeypatch):
    monkeypatch.delenv("WITSV3_WEB_TOKEN", raising=False)
    system = FakeSystem(tmp_path)
    return TestClient(create_app(system)), system


@pytest.fixture
def client_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "sekrit")
    system = FakeSystem(tmp_path)
    return TestClient(create_app(system)), system


def _parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        event, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                data += line[6:]
        if data:
            events.append((event, json.loads(data)))
    return events


# ------------------------------------------------------------------ chat


def test_chat_streams_events_and_records_history(client_noauth):
    client, system = client_noauth
    res = client.post("/api/chat", json={"message": "what is 2+2?"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(res.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "session"
    assert kinds[-1] == "done"

    stream_types = [d["type"] for e, d in events if e == "stream"]
    assert stream_types == ["thinking", "tool_call", "result"]

    done = events[-1][1]
    assert done["final"] == "The answer is 4."

    # History bookkeeping: one session with user + assistant messages
    assert len(system.session_histories) == 1
    conv = next(iter(system.session_histories.values()))
    roles = [m.role for m in conv.messages]
    assert roles == ["user", "assistant"]


def test_export_writes_session_transcript(client_noauth, tmp_path, monkeypatch):
    client, system = client_noauth
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var" / "exports").mkdir(parents=True)

    chat_res = client.post("/api/chat", json={"message": "hello"})
    events = _parse_sse(chat_res.text)
    session_id = events[0][1]["session_id"]

    export_res = client.post("/api/export", json={"session_id": session_id})
    assert export_res.status_code == 200
    data = export_res.json()
    assert data["success"] is True
    assert data["message_count"] == 2
    out = tmp_path / data["file_path"]
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "USER: hello" in text
    assert "ASSISTANT:" in text


def test_export_requires_session(client_noauth):
    client, _ = client_noauth
    res = client.post("/api/export", json={"session_id": "missing"})
    assert res.status_code == 400


def test_session_persisted_to_disk(client_noauth, tmp_path, monkeypatch):
    client, system = client_noauth
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var" / "exports").mkdir(parents=True)
    (tmp_path / "var" / "sessions").mkdir(parents=True)

    chat_res = client.post("/api/chat", json={"message": "persist this turn"})
    events = _parse_sse(chat_res.text)
    session_id = events[0][1]["session_id"]

    from core.session_store import load_persisted_sessions_into

    reloaded: dict = {}
    count = load_persisted_sessions_into(reloaded, "var")
    assert count >= 1
    assert session_id in reloaded
    assert any("persist this turn" in m.content for m in reloaded[session_id].messages)


def test_export_warns_on_short_session_after_title_set(client_noauth, tmp_path, monkeypatch):
    client, system = client_noauth
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var" / "exports").mkdir(parents=True)

    chat_res = client.post("/api/chat", json={"message": "long session topic alpha"})
    session_id = _parse_sse(chat_res.text)[0][1]["session_id"]
    system.session_histories[session_id].title = "long session topic alpha"
    # Only one exchange — simulate post-restart short history
    export_res = client.post("/api/export", json={"session_id": session_id})
    data = export_res.json()
    assert export_res.status_code == 200
    assert data.get("warning")


def test_chat_reuses_session(client_noauth):
    client, system = client_noauth
    first = _parse_sse(client.post("/api/chat", json={"message": "hi"}).text)
    session_id = first[0][1]["session_id"]

    client.post("/api/chat", json={"message": "again", "session_id": session_id})
    assert len(system.session_histories) == 1
    assert len(system.session_histories[session_id].messages) == 4


def test_create_session(client_noauth):
    client, system = client_noauth
    res = client.post("/api/sessions")
    assert res.status_code == 200
    body = res.json()
    assert body["title"] == "New chat"
    assert body["session_id"]
    assert len(system.session_histories) == 1


def test_list_sessions_and_auto_title(client_noauth):
    client, system = client_noauth
    assert client.get("/api/sessions").json() == {"sessions": []}

    first = _parse_sse(
        client.post("/api/chat", json={"message": "Summarize the quarterly report"}).text
    )
    session_id = first[0][1]["session_id"]

    listed = client.get("/api/sessions").json()["sessions"]
    assert len(listed) == 1
    assert listed[0]["session_id"] == session_id
    assert listed[0]["title"] == "Summarize the quarterly report"
    assert listed[0]["message_count"] == 2
    assert "quarterly" in listed[0]["preview"]


def test_get_session_messages(client_noauth):
    client, _ = client_noauth
    first = _parse_sse(client.post("/api/chat", json={"message": "hello there"}).text)
    session_id = first[0][1]["session_id"]

    body = client.get(f"/api/sessions/{session_id}/messages").json()
    assert body["session_id"] == session_id
    assert body["title"] == "hello there"
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "hello there"
    assert body["messages"][1]["role"] == "assistant"


def test_rename_session(client_noauth):
    client, system = client_noauth
    created = client.post("/api/sessions").json()
    session_id = created["session_id"]

    res = client.patch(
        f"/api/sessions/{session_id}",
        json={"title": "Planning notes"},
    )
    assert res.status_code == 200
    assert res.json()["title"] == "Planning notes"
    assert system.session_histories[session_id].title == "Planning notes"

    listed = client.get("/api/sessions").json()["sessions"]
    assert listed[0]["title"] == "Planning notes"


def test_rename_session_requires_title(client_noauth):
    client, _ = client_noauth
    created = client.post("/api/sessions").json()
    res = client.patch(f"/api/sessions/{created['session_id']}", json={"title": "   "})
    assert res.status_code == 400


def test_get_session_messages_not_found(client_noauth):
    client, _ = client_noauth
    res = client.get("/api/sessions/missing-id/messages")
    assert res.status_code == 404


class OllamaDownControlCenter:
    async def run(self, user_input, conversation_history=None, session_id=None, **kwargs):
        yield StreamData(
            type="error",
            content="An error occurred during orchestration: Failed to connect to Ollama at http://localhost:11434.",
            source="orchestrator",
            error_details="Failed to connect to Ollama at http://localhost:11434. Please ensure Ollama is running.",
        )


def test_chat_ollama_down_shows_friendly_error(client_noauth):
    client, system = client_noauth
    system.control_center = OllamaDownControlCenter()
    events = _parse_sse(client.post("/api/chat", json={"message": "hello"}).text)
    error_events = [d for e, d in events if e == "stream" and d.get("type") == "error"]
    assert error_events
    assert error_events[0]["user_error"]["code"] == "ollama_unavailable"
    assert "Can't reach Ollama" in error_events[0]["content"]
    assert "ollama serve" in error_events[0]["user_error"]["hint"]


# ------------------------------------------------------------------ info endpoints


def test_status(client_noauth):
    client, _ = client_noauth
    res = client.get("/api/status")
    assert res.status_code == 200
    body = res.json()
    assert body["project"] == "WitsV3"
    assert body["models"]["default"] == "qwen3:8b"
    assert body["tool_count"] == 2


def test_tools(client_noauth):
    client, _ = client_noauth
    body = client.get("/api/tools").json()
    names = [t["name"] for t in body["tools"]]
    assert "calculator" in names


def test_memory_search(client_noauth):
    client, _ = client_noauth
    body = client.get("/api/memory/search", params={"q": "cats"}).json()
    assert body["results"][0]["relevance"] == 0.87
    assert "cats" in body["results"][0]["text"]


def test_memory_recent_list(client_noauth):
    client, _ = client_noauth
    body = client.get("/api/memory/recent", params={"limit": 2, "offset": 0}).json()
    assert len(body["results"]) == 2
    assert body["results"][0]["type"] == "DOCUMENT_CHUNK"
    assert "timestamp" in body["results"][0]


def test_memory_prune_requires_confirm(client_noauth):
    client, _ = client_noauth
    res = client.post(
        "/api/memory/prune",
        json={"filter_dict": {"type": "DOCUMENT_CHUNK"}, "confirm": "NOPE"},
    )
    assert res.status_code == 400


def test_memory_prune_deletes(client_noauth):
    client, _ = client_noauth
    res = client.post(
        "/api/memory/prune",
        json={"filter_dict": {"type": "DOCUMENT_CHUNK"}, "confirm": "PRUNE"},
    )
    assert res.status_code == 200
    assert res.json()["removed"] == 2


# ------------------------------------------------------------------ documents


def test_documents_list_and_upload(client_noauth, tmp_path):
    client, system = client_noauth
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "notes.md").write_text("hello")

    body = client.get("/api/documents").json()
    assert body["files"][0]["name"] == "notes.md"
    assert body["files"][0]["chunks"] == 2  # from FakeMemoryManager

    res = client.post(
        "/api/documents/upload",
        files={"file": ("new.txt", b"fresh content", "text/plain")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["saved"] == "new.txt"
    assert body["ingest"]["success"] is True
    assert (docs / "new.txt").read_bytes() == b"fresh content"


def test_upload_strips_path_traversal(client_noauth, tmp_path):
    client, _ = client_noauth
    res = client.post(
        "/api/documents/upload",
        files={"file": ("..\\..\\evil.txt", b"nope", "text/plain")},
    )
    assert res.status_code == 200
    saved = res.json()["saved"]
    assert ".." not in saved and "/" not in saved and "\\" not in saved


def test_documents_list_includes_metadata(client_noauth, tmp_path):
    client, _ = client_noauth
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "notes.md").write_text("hello")

    body = client.get("/api/documents").json()
    assert body["count"] == 1
    assert body["total_chunks"] == 2
    f = body["files"][0]
    assert f["ext"] == "md"
    assert "modified" in f and f["modified"]


def test_document_delete_removes_file_and_chunks(client_noauth, tmp_path):
    client, _ = client_noauth
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "notes.md").write_text("hello")

    res = client.post("/api/documents/delete", json={"name": "notes.md"})
    assert res.status_code == 200
    body = res.json()
    assert body["deleted"] == "notes.md"
    assert body["removed_chunks"] == 2  # from FakeMemoryManager
    assert body["file_removed"] is True
    assert not (docs / "notes.md").exists()


def test_document_delete_rejects_traversal(client_noauth, tmp_path):
    client, _ = client_noauth
    res = client.post("/api/documents/delete", json={"name": "../../evil.txt"})
    assert res.status_code == 400


def test_document_reindex(client_noauth, tmp_path):
    client, _ = client_noauth
    res = client.post("/api/documents/reindex")
    assert res.status_code == 200
    assert res.json()["ingest"]["success"] is True


# ------------------------------------------------------------------ auth


def test_auth_blocks_api_without_token(client_auth):
    client, _ = client_auth
    assert client.get("/api/status").status_code == 401


def test_auth_allows_with_bearer(client_auth):
    client, _ = client_auth
    res = client.get("/api/status", headers={"Authorization": "Bearer sekrit"})
    assert res.status_code == 200


def test_auth_wrong_token_rejected(client_auth):
    client, _ = client_auth
    res = client.get("/api/status", headers={"Authorization": "Bearer wrong"})
    assert res.status_code == 401


def test_auth_does_not_block_index(client_auth):
    client, _ = client_auth
    assert client.get("/").status_code == 200


def test_index_served(client_noauth):
    client, _ = client_noauth
    res = client.get("/")
    assert res.status_code == 200
    assert "WITS" in res.text


# ------------------------------------------------------------------ personality

PROFILE_YAML = """
wits_personality:
  name: "Test Profile"
  profile_id: "test"
  identity_label: "WITS"
  default_role: "Test Assistant"
  core_directives:
    - "Be excellent."
  communication:
    tone: "neutral"
    language_level: "plain"
    verbosity: "adaptive"
    structure_preference: "prose"
    humor: "off"
  persona_layers:
    default_persona: "Engineer"
    available_roles:
      - name: "Engineer"
      - name: "Companion"
"""


@pytest.fixture
def client_personality(tmp_path, monkeypatch):
    monkeypatch.delenv("WITSV3_WEB_TOKEN", raising=False)
    system = FakeSystem(tmp_path)
    profile = tmp_path / "wits_personality.yaml"
    profile.write_text(PROFILE_YAML, encoding="utf-8")
    system.config.personality.profile_path = str(profile)
    yield TestClient(create_app(system)), system, tmp_path
    # POST/DELETE swap the global personality manager - don't leak it
    import core.personality_manager as pm_module

    pm_module._personality_manager = None


def test_personality_get(client_personality):
    client, _, _ = client_personality
    body = client.get("/api/personality").json()
    assert body["identity_label"] == "WITS"
    assert body["tone"] == "neutral"
    assert body["available_personas"] == ["Engineer", "Companion"]
    assert "You are WITS" in body["system_prompt"]


def test_personality_save_apply_and_reset(client_personality):
    client, _, tmp_path = client_personality
    res = client.post(
        "/api/personality",
        json={
            "identity_label": "JARVIS",
            "tone": "wry and unflappable",
            "humor": "dry",
            "core_directives": ["Never lose the plot.", "   ", "Serve tea."],
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["saved"] is True
    assert "You are JARVIS" in body["system_prompt"]
    assert "wry and unflappable" in body["system_prompt"]

    overrides = tmp_path / "personality_overrides.yaml"
    assert overrides.exists()

    # GET reflects merged values; untouched fields keep base values
    merged = client.get("/api/personality").json()
    assert merged["identity_label"] == "JARVIS"
    assert merged["language_level"] == "plain"
    assert merged["core_directives"] == ["Never lose the plot.", "Serve tea."]

    # Reset removes the overrides and restores the base profile
    res = client.delete("/api/personality")
    assert res.status_code == 200
    assert res.json()["reset"] is True
    assert not overrides.exists()
    assert client.get("/api/personality").json()["identity_label"] == "WITS"


def test_personality_empty_post_rejected(client_personality):
    client, _, _ = client_personality
    assert client.post("/api/personality", json={}).status_code == 400


def test_personality_page_public_but_api_protected(client_auth):
    client, _ = client_auth
    assert client.get("/personality").status_code == 200
    assert client.get("/api/personality").status_code == 401


# ------------------------------------------------------------------ settings


@pytest.fixture
def client_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("WITSV3_WEB_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)  # config.local.yaml is written to the CWD
    system = FakeSystem(tmp_path)
    return TestClient(create_app(system)), system, tmp_path


def test_settings_get(client_settings):
    client, system, _ = client_settings
    body = client.get("/api/settings").json()
    assert body["history_window"] == system.config.agents.history_window
    assert body["escalation_model"] == "claude-opus-4-8"
    assert "claude-opus-4-8" in body["escalation_models"]
    assert isinstance(body["anthropic_key_configured"], bool)
    mr = body["model_routing"]
    assert mr["enabled"] is system.config.model_routing.enabled
    assert mr["trivial_model"] == system.config.model_routing.trivial_model
    assert mr["code_model"] == system.config.model_routing.code_model


def test_settings_post_model_routing(client_settings):
    client, system, tmp_path = client_settings
    res = client.post(
        "/api/settings",
        json={
            "routing_enabled": False,
            "routing_trivial_model": "llama3.2:3b",
            "routing_code_model": "qwen2.5-coder:7b",
            "routing_complex_model": "qwen3:8b",
            "routing_trivial_max_chars": 100,
        },
    )
    assert res.status_code == 200
    assert system.config.model_routing.enabled is False
    assert system.config.model_routing.trivial_max_chars == 100
    overrides = (tmp_path / "config.local.yaml").read_text()
    assert "model_routing:" in overrides
    assert "enabled: false" in overrides


def test_settings_post_applies_live_and_persists(client_settings):
    client, system, tmp_path = client_settings
    res = client.post(
        "/api/settings",
        json={
            "history_window": 40,
            "default_temperature": 0.3,
            "escalation_max_tokens": 1024,
        },
    )
    assert res.status_code == 200
    # Applied live to the running system config
    assert system.config.agents.history_window == 40
    assert system.config.agents.default_temperature == 0.3
    assert system.config.escalation.max_tokens == 1024
    # Persisted to the overrides file, not config.yaml
    overrides = tmp_path / "config.local.yaml"
    assert overrides.exists()
    assert "history_window: 40" in overrides.read_text()


def test_settings_post_rejects_bad_values(client_settings):
    client, system, _ = client_settings
    original = system.config.agents.history_window
    res = client.post("/api/settings", json={"history_window": 9999})
    assert res.status_code == 400
    assert system.config.agents.history_window == original


def test_local_overrides_are_loaded(client_settings, monkeypatch):
    client, _, tmp_path = client_settings
    client.post("/api/settings", json={"history_window": 44})
    from core.config import load_config

    fresh = load_config()  # cwd is tmp_path; default config + local overrides
    assert fresh.agents.history_window == 44


# --------------------------------------------------------------- escalations


@pytest.fixture
def client_escalation(tmp_path, monkeypatch):
    monkeypatch.delenv("WITSV3_WEB_TOKEN", raising=False)
    import core.escalation as escalation_module

    escalation_module._manager = None  # fresh queue per test
    system = FakeSystem(tmp_path)
    yield TestClient(create_app(system)), system
    escalation_module._manager = None


def test_escalation_flow_approve(client_escalation, monkeypatch):
    client, system = client_escalation
    from core.escalation import EscalationManager, get_escalation_manager

    async def fake_call(self, request):
        return "Claude says: use a mutex.", {"input_tokens": 100, "output_tokens": 50}

    monkeypatch.setattr(EscalationManager, "_call_claude", fake_call)

    manager = get_escalation_manager()
    request = manager.create("How do I fix this race?", context="some code")
    assert request.estimate()["max_cost_usd"] > 0

    # Pending request is visible
    body = client.get("/api/escalations").json()
    assert body["requests"][0]["status"] == "pending"

    # Approve → fake Claude call runs, answer lands in the session history
    from core.schemas import ConversationHistory

    system.session_histories["sess1"] = ConversationHistory(session_id="sess1")
    res = client.post(f"/api/escalations/{request.id}/approve", json={"session_id": "sess1"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "answered"
    assert "mutex" in body["answer"]
    assert body["cost_usd"] > 0
    assert "mutex" in system.session_histories["sess1"].messages[-1].content

    # Approving twice is rejected
    assert client.post(f"/api/escalations/{request.id}/approve").status_code == 409


def test_escalation_deny_spends_nothing(client_escalation, monkeypatch):
    client, _ = client_escalation
    from core.escalation import EscalationManager, get_escalation_manager

    called = []

    async def fake_call(self, request):
        called.append(request.id)
        return "x", {}

    monkeypatch.setattr(EscalationManager, "_call_claude", fake_call)
    request = get_escalation_manager().create("q")

    res = client.post(f"/api/escalations/{request.id}/deny")
    assert res.status_code == 200
    assert called == []  # the API was never touched
    assert get_escalation_manager().get(request.id).status == "denied"
    # A denied request can no longer be approved
    assert client.post(f"/api/escalations/{request.id}/approve").status_code == 409


def test_ask_claude_tool_queues_pending(client_escalation, tmp_path, monkeypatch):
    _, _ = client_escalation
    import asyncio

    from core.escalation import get_escalation_manager
    from tools.ask_claude_tool import AskClaudeTool

    tool = AskClaudeTool()
    result = asyncio.run(tool.execute(question="What is a monad?", context="haskell"))
    assert result["success"] is True
    assert result["status"] == "pending_user_approval"
    queued = get_escalation_manager().get(result["escalation_id"])
    assert queued.status == "pending"
    assert queued.question == "What is a monad?"


# ------------------------------------------------------------------ mcp


@pytest.fixture
def client_mcp(tmp_path, monkeypatch):
    monkeypatch.delenv("WITSV3_WEB_TOKEN", raising=False)
    system = FakeSystem(tmp_path)
    mcp_config = tmp_path / "mcp_tools.json"
    mcp_config.write_text(
        json.dumps(
            {
                "auto_connect": True,
                "servers": [{"name": "demo", "command": "node server.js"}],
            }
        )
    )
    system.config.tool_system.mcp_tool_definitions_path = str(mcp_config)
    return TestClient(create_app(system)), system, mcp_config


def test_mcp_list_servers(client_mcp):
    client, _, _ = client_mcp
    body = client.get("/api/mcp/servers").json()
    assert body["servers"][0]["name"] == "demo"
    assert body["servers"][0]["connected"] is False


def test_mcp_add_and_remove_server(client_mcp):
    client, _, mcp_config = client_mcp
    res = client.post(
        "/api/mcp/servers",
        json={
            "name": "memory",
            "command": "npx -y @modelcontextprotocol/server-memory",
        },
    )
    assert res.status_code == 200
    saved = json.loads(mcp_config.read_text())
    assert any(s["name"] == "memory" for s in saved["servers"])

    # Duplicate names are rejected
    assert (
        client.post(
            "/api/mcp/servers",
            json={
                "name": "memory",
                "command": "x",
            },
        ).status_code
        == 409
    )

    # Removal updates the config file
    assert client.delete("/api/mcp/servers/memory").status_code == 200
    saved = json.loads(mcp_config.read_text())
    assert not any(s["name"] == "memory" for s in saved["servers"])
    assert client.delete("/api/mcp/servers/memory").status_code == 404


def test_mcp_tools_requires_connection(client_mcp):
    client, _, _ = client_mcp
    assert client.get("/api/mcp/servers/demo/tools").status_code == 409


def test_mcp_status_and_search_providers(client_mcp):
    client, _, _ = client_mcp
    status = client.get("/api/mcp/status").json()
    assert status["configured_servers"] == 1
    assert status["connected_servers"] == 0

    providers = client.get("/api/search/providers").json()
    assert providers["provider_mode"] == "auto"
    assert "brave_configured" in providers


def test_mcp_all_tools_empty_when_disconnected(client_mcp):
    client, _, _ = client_mcp
    assert client.get("/api/mcp/tools").json() == {"tools": []}


def test_mcp_invoke_unknown_tool(client_mcp):
    client, _, _ = client_mcp
    res = client.post("/api/mcp/tools/unknown_tool/invoke", json={"arguments": {}})
    assert res.status_code == 409
