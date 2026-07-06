"""Tests for the WitsV3 web UI server."""

import json
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

    async def run(self, user_input, conversation_history=None, session_id=None):
        self.calls.append({"user_input": user_input, "session_id": session_id})
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
            "ingest_documents": FakeTool(name="ingest_documents", description="Ingest docs", execute=ingest_execute),
        }

    def get_tool(self, name):
        return self.tools.get(name)


class FakeMemoryManager:
    async def search_memory(self, query_text, limit=5, min_relevance=0.0, filter_dict=None):
        seg = SimpleNamespace(
            type="DOCUMENT_CHUNK", source="notes.md",
            content=SimpleNamespace(text=f"match for {query_text}", tool_output=None),
            relevance_score=0.87,
        )
        return [seg][:limit]

    async def get_recent_memory(self, limit=10, filter_dict=None):
        seg = SimpleNamespace(metadata={"file_path": "notes.md"})
        return [seg, seg]


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


def test_chat_reuses_session(client_noauth):
    client, system = client_noauth
    first = _parse_sse(client.post("/api/chat", json={"message": "hi"}).text)
    session_id = first[0][1]["session_id"]

    client.post("/api/chat", json={"message": "again", "session_id": session_id})
    assert len(system.session_histories) == 1
    assert len(system.session_histories[session_id].messages) == 4


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
