"""Tests for owner-only /shutdown and /restart controls."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core.config import WitsV3Config
from core.schemas import StreamData
from web.owner_controls import parse_owner_command
from web.server import create_app


class FakeControlCenter:
    def __init__(self):
        self.calls = []

    async def run(self, user_input, conversation_history=None, session_id=None):
        self.calls.append({"user_input": user_input, "session_id": session_id})
        yield StreamData(type="result", content="agent ran", source="wcca")


class FakeToolRegistry:
    def __init__(self):
        self.tools = {}

    def get_tool(self, name):
        return None


class FakeMemoryManager:
    async def search_memory(self, *a, **k):
        return []

    async def get_recent_memory(self, limit=10, filter_dict=None):
        return []

    async def delete_segments(self, filter_dict):
        return 0


class FakeSystem:
    def __init__(self, tmp_path):
        self.config = WitsV3Config()
        self.config.document_rag.documents_path = str(tmp_path / "documents")
        self.control_center = FakeControlCenter()
        self.tool_registry = FakeToolRegistry()
        self.memory_manager = FakeMemoryManager()
        self.session_histories = {}


def _parse_sse(text):
    import json

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


@pytest.fixture
def client_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "owner-secret")
    system = FakeSystem(tmp_path)
    system.config.web_ui.require_auth = True
    return TestClient(create_app(system)), system


@pytest.mark.parametrize(
    "message,expected",
    [
        ("/shutdown", "shutdown"),
        ("/SHUTDOWN", "shutdown"),
        ("/stop", "shutdown"),
        ("/kill", "shutdown"),
        ("/quit", "shutdown"),
        ("/restart", "restart"),
        ("/restart!", "restart"),
        ("please shut down", None),
        ("shutdown the server", None),
        ("", None),
    ],
)
def test_parse_owner_command(message, expected):
    assert parse_owner_command(message) == expected


def test_chat_shutdown_requires_token_when_configured(client_owner, monkeypatch):
    client, _system = client_owner
    scheduled = []
    monkeypatch.setattr(
        "web.server.schedule_owner_action",
        lambda action, delay_seconds=1.0: scheduled.append((action, delay_seconds))
        or {"success": True, "action": action, "message": "ok"},
    )

    res = client.post("/api/chat", json={"message": "/shutdown"})
    assert res.status_code == 401
    assert scheduled == []


def test_chat_shutdown_schedules_when_authorized(client_owner, monkeypatch):
    client, system = client_owner
    scheduled = []
    monkeypatch.setattr(
        "web.server.schedule_owner_action",
        lambda action, delay_seconds=1.0: scheduled.append((action, delay_seconds))
        or {"success": True, "action": action, "message": "ok"},
    )

    res = client.post(
        "/api/chat",
        json={"message": "/shutdown"},
        headers={"Authorization": "Bearer owner-secret"},
    )
    assert res.status_code == 200
    events = _parse_sse(res.text)
    assert events[-1][0] == "done"
    assert events[-1][1]["owner_action"] == "shutdown"
    assert "Force-shutting down" in events[-1][1]["final"]
    assert scheduled == [("shutdown", 1.0)]
    assert system.control_center.calls == []


def test_chat_restart_schedules_when_authorized(client_owner, monkeypatch):
    client, system = client_owner
    scheduled = []
    monkeypatch.setattr(
        "web.server.schedule_owner_action",
        lambda action, delay_seconds=1.0: scheduled.append(action)
        or {"success": True, "action": action, "message": "ok"},
    )

    res = client.post(
        "/api/chat",
        json={"message": "/restart"},
        headers={"Authorization": "Bearer owner-secret"},
    )
    assert res.status_code == 200
    events = _parse_sse(res.text)
    assert events[-1][1]["owner_action"] == "restart"
    assert scheduled == ["restart"]
    assert system.control_center.calls == []


def test_api_shutdown_requires_confirm(client_owner, monkeypatch):
    client, _ = client_owner
    monkeypatch.setattr(
        "web.server.schedule_owner_action",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not schedule")),
    )
    res = client.post(
        "/api/owner/shutdown",
        json={"confirm": "NOPE"},
        headers={"Authorization": "Bearer owner-secret"},
    )
    assert res.status_code == 400


def test_api_shutdown_ok(client_owner, monkeypatch):
    client, _ = client_owner
    scheduled = []
    monkeypatch.setattr(
        "web.server.schedule_owner_action",
        lambda action, delay_seconds=1.0: scheduled.append(action)
        or {
            "success": True,
            "action": action,
            "delay_seconds": delay_seconds,
            "message": "bye",
        },
    )
    res = client.post(
        "/api/owner/shutdown",
        json={"confirm": "SHUTDOWN"},
        headers={"Authorization": "Bearer owner-secret"},
    )
    assert res.status_code == 200
    assert res.json()["action"] == "shutdown"
    assert scheduled == ["shutdown"]


def test_owner_controls_refuse_without_env_token(tmp_path, monkeypatch):
    monkeypatch.delenv("WITSV3_WEB_TOKEN", raising=False)
    system = FakeSystem(tmp_path)
    system.config.web_ui.require_auth = False
    client = TestClient(create_app(system))
    scheduled = []
    monkeypatch.setattr(
        "web.server.schedule_owner_action",
        lambda *a, **k: scheduled.append(True),
    )
    res = client.post("/api/chat", json={"message": "/shutdown"})
    assert res.status_code == 200
    events = _parse_sse(res.text)
    assert "WITSV3_WEB_TOKEN" in events[-1][1]["final"]
    assert scheduled == []
