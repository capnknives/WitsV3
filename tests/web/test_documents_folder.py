"""WP F5: uploads live in <folders.base>/documents; the old documents_path is left alone."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from core.filesystem_policy import warn_old_documents_path
from tests.web.test_web_server import FakeSystem
from web.server import create_app


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("WITSV3_WEB_TOKEN", raising=False)
    monkeypatch.setenv("WITSV3_PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    old = tmp_path / "old_docs"
    old.mkdir()
    (old / "a.md").write_text("old a")
    (old / "b.md").write_text("old b")
    system = FakeSystem(tmp_path)
    system.config.document_rag.documents_path = str(old)
    return TestClient(create_app(system)), system, tmp_path


def test_upload_list_delete_use_base_documents(env):
    client, _system, tmp = env
    res = client.post("/api/documents/upload", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert res.status_code == 200
    assert (tmp / "wits" / "documents" / "notes.txt").read_bytes() == b"hi"
    assert [f["name"] for f in client.get("/api/documents").json()["files"]] == ["notes.txt"]
    assert client.post("/api/documents/delete", json={"name": "notes.txt"}).status_code == 200
    assert not (tmp / "wits" / "documents" / "notes.txt").exists()
    assert sorted(p.name for p in (tmp / "old_docs").iterdir()) == ["a.md", "b.md"]


def test_old_documents_path_logs_one_warning_with_count(env, caplog):
    _client, system, tmp = env
    caplog.set_level(logging.WARNING, logger="WitsV3.FilesystemPolicy")
    assert warn_old_documents_path(system.config) == 2
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1 and "2 file(s)" in warnings[0] and "a.md" not in warnings[0]
    assert "not re-indexed" in warnings[0] and "Documents page" in warnings[0]
    assert sorted(p.name for p in (tmp / "old_docs").iterdir()) == ["a.md", "b.md"]


def test_no_warning_when_old_path_is_empty_or_unset(env, caplog):
    _client, system, tmp = env
    caplog.set_level(logging.WARNING, logger="WitsV3.FilesystemPolicy")
    system.config.document_rag.documents_path = ""
    assert warn_old_documents_path(system.config) == 0
    (tmp / "empty").mkdir()
    system.config.document_rag.documents_path = str(tmp / "empty")
    assert warn_old_documents_path(system.config) == 0
    assert caplog.records == []


def _real_ingest(system, tmp):
    """Swap the fake memory and ingest tool for the real ones."""
    from core.memory_manager import MemoryManager
    from tests.core.test_path_access import _EmbedLLM
    from tools.document_tools import DocumentIngestTool

    config = system.config
    config.memory_manager.backend = "basic"
    config.memory_manager.memory_file_path = str(tmp / "memory.json")
    system.memory_manager = MemoryManager(config=config, llm_interface=_EmbedLLM())
    tool = DocumentIngestTool()
    tool.set_dependencies(config, _EmbedLLM(), system.memory_manager)
    system.tool_registry.tools["ingest_documents"] = tool
    return tool


async def _chunks(memory, name):
    segs = await memory.get_recent_memory(limit=1000, filter_dict={"type": "DOCUMENT_CHUNK"})
    return [s for s in segs if s.metadata.get("file_path") == name]


async def test_old_folder_chunks_kept_until_file_gone_from_both_places(env):
    import hashlib

    client, system, tmp = env
    tool = _real_ingest(system, tmp)
    memory = system.memory_manager
    for name in ("a.md", "b.md"):
        text = (tmp / "old_docs" / name).read_text()
        await memory.add_memory(
            type="DOCUMENT_CHUNK",
            source=name,
            content_text=text,
            metadata={"file_path": name, "file_hash": hashlib.sha256(text.encode()).hexdigest()},
        )
    await tool.execute()
    client.post("/api/documents/upload", files={"file": ("new.md", b"new", "text/plain")})
    client.post("/api/documents/reindex")
    assert len(await _chunks(memory, "a.md")) == 1 and len(await _chunks(memory, "b.md")) == 1
    (tmp / "old_docs" / "a.md").unlink()
    (tmp / "old_docs" / "b.md").rename(tmp / "wits" / "documents" / "b.md")
    await tool.execute()
    assert await _chunks(memory, "a.md") == []
    assert len(await _chunks(memory, "b.md")) == 1  # moved with the same content: no duplicate


async def test_indexer_reads_documents_and_inbox_not_old_path(env):
    _client, system, tmp = env
    for name in ("documents", "inbox"):
        (tmp / "wits" / name).mkdir(parents=True)
        (tmp / "wits" / name / f"{name}.md").write_text(f"text in {name}")
    summary = await _real_ingest(system, tmp).execute()
    assert set(summary["searchable_files"]) == {"documents.md", "inbox/inbox.md"}
