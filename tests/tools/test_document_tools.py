"""Tests for the document RAG tools (ingest + search)."""

import hashlib
import os
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import pytest

from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from tools.document_tools import (
    DOCUMENT_SEGMENT_TYPE,
    DocumentIngestTool,
    DocumentSearchTool,
    _chunk_text,
)


class DummyLLM(BaseLLMInterface):
    """Deterministic bag-of-words embeddings (see test_faiss_memory_backend)."""

    def __init__(self):
        pass

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return "dummy response"

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield "dummy"

    async def get_embedding(self, text, model=None):
        embedding = np.zeros(384)
        for word in text.lower().split():
            dim = int(hashlib.md5(word.encode()).hexdigest(), 16) % 384
            embedding[dim] += 1.0
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()


@pytest.fixture
def rag_env(tmp_path, monkeypatch):
    """Config + memory manager + wired tools operating in temp directories."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    config = WitsV3Config()
    config.memory_manager.backend = "basic"
    config.memory_manager.memory_file_path = str(tmp_path / "memory.json")
    config.document_rag.documents_path = str(docs_dir)
    config.document_rag.chunk_size = 200
    config.document_rag.chunk_overlap = 40

    llm = DummyLLM()
    memory = MemoryManager(config=config, llm_interface=llm)

    ingest = DocumentIngestTool()
    search = DocumentSearchTool()
    ingest.set_dependencies(config, llm, memory)
    search.set_dependencies(config, llm, memory)

    return config, memory, ingest, search, docs_dir


# ---------------------------------------------------------------- chunking

def test_chunk_text_small_passthrough():
    assert _chunk_text("short text", 200, 40) == ["short text"]


def test_chunk_text_respects_size_and_overlap():
    paras = "\n\n".join(f"Paragraph {i} " + ("word " * 20) for i in range(10))
    chunks = _chunk_text(paras, 200, 40)
    assert len(chunks) > 1
    # Chunks stay near the target size (paragraph packing allows slight spill)
    assert all(len(c) <= 260 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_chunk_text_hard_splits_giant_paragraph():
    text = "x" * 1000  # single paragraph larger than several chunks
    chunks = _chunk_text(text, 200, 40)
    assert len(chunks) >= 4
    assert "".join(c.replace(" ", "") for c in chunks).count("x") >= 1000


def test_chunk_text_empty():
    assert _chunk_text("", 200, 40) == []
    assert _chunk_text("   \n\n  ", 200, 40) == []


# ---------------------------------------------------------------- ingest

@pytest.mark.asyncio
async def test_ingest_new_file(rag_env):
    config, memory, ingest, search, docs_dir = rag_env
    (docs_dir / "notes.md").write_text("Cats are wonderful pets.\n\nDogs are loyal companions.")

    result = await ingest.execute()

    assert result["success"] is True
    assert result["files_scanned"] == 1
    assert result["files_ingested"] == 1
    assert result["chunks_added"] >= 1

    segments = await memory.get_recent_memory(limit=100, filter_dict={"type": DOCUMENT_SEGMENT_TYPE})
    assert len(segments) == result["chunks_added"]
    seg = segments[0]
    assert seg.metadata["file_path"] == "notes.md"
    assert seg.metadata["file_hash"]
    assert seg.metadata["total_chunks"] == result["chunks_added"]
    assert seg.embedding is not None


@pytest.mark.asyncio
async def test_ingest_unchanged_file_skipped(rag_env):
    config, memory, ingest, search, docs_dir = rag_env
    (docs_dir / "notes.md").write_text("Some stable content here.")

    await ingest.execute()
    result2 = await ingest.execute()

    assert result2["files_unchanged"] == 1
    assert result2["files_ingested"] == 0
    assert result2["chunks_added"] == 0


@pytest.mark.asyncio
async def test_ingest_modified_file_replaces_chunks(rag_env):
    config, memory, ingest, search, docs_dir = rag_env
    f = docs_dir / "notes.md"
    f.write_text("Original content about topic alpha.")
    await ingest.execute()

    f.write_text("Rewritten content about topic beta.")
    result = await ingest.execute()

    assert result["files_ingested"] == 1
    segments = await memory.get_recent_memory(limit=100, filter_dict={"type": DOCUMENT_SEGMENT_TYPE})
    texts = [s.content.text for s in segments]
    assert all("beta" in t for t in texts)
    assert not any("alpha" in t for t in texts)


@pytest.mark.asyncio
async def test_ingest_deleted_file_removes_chunks(rag_env):
    config, memory, ingest, search, docs_dir = rag_env
    f = docs_dir / "temp.txt"
    f.write_text("Ephemeral content.")
    await ingest.execute()

    f.unlink()
    result = await ingest.execute()

    assert result["files_removed"] == 1
    segments = await memory.get_recent_memory(limit=100, filter_dict={"type": DOCUMENT_SEGMENT_TYPE})
    assert segments == []


@pytest.mark.asyncio
async def test_ingest_ignores_unsupported_extensions(rag_env):
    config, memory, ingest, search, docs_dir = rag_env
    (docs_dir / "binary.exe").write_bytes(b"\x00\x01")
    result = await ingest.execute()
    assert result["files_scanned"] == 0


# ---------------------------------------------------------------- search

@pytest.mark.asyncio
async def test_search_finds_relevant_document(rag_env):
    config, memory, ingest, search, docs_dir = rag_env
    (docs_dir / "pets.md").write_text("Cats are wonderful pets and cats purr loudly.")
    (docs_dir / "space.md").write_text("Rockets travel to orbit using powerful engines.")
    await ingest.execute()

    result = await search.execute(query="cats purr pets", max_results=2, min_relevance=0.0)

    assert result["success"] is True
    assert result["result_count"] >= 1
    assert result["results"][0]["file"] == "pets.md"
    assert result["results"][0]["relevance"] > 0


@pytest.mark.asyncio
async def test_search_min_relevance_filters(rag_env):
    config, memory, ingest, search, docs_dir = rag_env
    (docs_dir / "space.md").write_text("Rockets travel to orbit using powerful engines.")
    await ingest.execute()

    result = await search.execute(query="completely unrelated gibberish zzz", min_relevance=0.99)
    assert result["success"] is True
    assert result["result_count"] == 0


@pytest.mark.asyncio
async def test_search_empty_query_rejected(rag_env):
    config, memory, ingest, search, docs_dir = rag_env
    result = await search.execute(query="   ")
    assert result["success"] is False


# ---------------------------------------------------------------- wiring

@pytest.mark.asyncio
async def test_tools_report_not_ready_without_dependencies():
    ingest = DocumentIngestTool()
    search = DocumentSearchTool()
    r1 = await ingest.execute()
    r2 = await search.execute(query="anything")
    assert r1["success"] is False and "not initialized" in r1["error"]
    assert r2["success"] is False and "not initialized" in r2["error"]


def test_tool_schemas():
    ingest = DocumentIngestTool()
    search = DocumentSearchTool()
    assert ingest.get_schema()["name"] == "ingest_documents"
    schema = search.get_schema()
    assert schema["name"] == "document_search"
    assert "query" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["query"]


# ---------------------------------------------------------------- delete API

@pytest.mark.asyncio
async def test_memory_delete_segments(rag_env):
    config, memory, ingest, search, docs_dir = rag_env
    (docs_dir / "a.md").write_text("Content A.")
    (docs_dir / "b.md").write_text("Content B.")
    await ingest.execute()

    removed = await memory.delete_segments({"type": DOCUMENT_SEGMENT_TYPE, "file_path": "a.md"})
    assert removed >= 1

    remaining = await memory.get_recent_memory(limit=100, filter_dict={"type": DOCUMENT_SEGMENT_TYPE})
    assert all(s.metadata["file_path"] == "b.md" for s in remaining)

    # Empty filter must refuse to delete everything
    assert await memory.delete_segments({}) == 0
