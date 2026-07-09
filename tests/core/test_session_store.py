"""Tests for var/sessions persistence."""

from core.schemas import ConversationHistory
from core.session_store import load_persisted_sessions_into, persist_session


def test_persist_and_reload_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conv = ConversationHistory(session_id="sess-abc")
    conv.title = "Test chat"
    conv.add_message("user", "hello")
    conv.add_message("assistant", "hi there")

    path = persist_session(conv, "var")
    assert path is not None
    assert path.is_file()

    store: dict = {}
    count = load_persisted_sessions_into(store, "var")
    assert count == 1
    assert store["sess-abc"].messages[0].content == "hello"
