"""Tests for ToolRegistry LLM kwarg normalization."""

import pytest

from core.tool_registry import ToolRegistry


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg._register_builtin_tools()
    return reg


def test_normalize_read_file_path_alias(registry):
    args = registry._normalize_tool_kwargs(
        "read_file", {"path": "notes.md", "tool_name": "read_file"}
    )
    assert args == {"file_path": "notes.md"}


def test_normalize_list_directory_alias(registry):
    args = registry._normalize_tool_kwargs("list_directory", {"directory": "./documents"})
    assert args == {"directory_path": "./documents"}


def test_normalize_document_search_top_k(registry):
    args = registry._normalize_tool_kwargs("document_search", {"query": "cats", "top_k": 3})
    assert args == {"query": "cats", "max_results": 3}


def test_normalize_ingest_documents_strips_hallucinated_args(registry):
    args = registry._normalize_tool_kwargs(
        "ingest_documents", {"arg1": "value", "tool_name": "ingest_documents"}
    )
    assert args == {}


def test_validate_ingest_documents_ignores_arg1(registry):
    validation = registry.validate_tool_call("ingest_documents", arg1="bogus")
    assert validation["valid"] is True


def test_validate_read_file_accepts_path_alias(registry):
    validation = registry.validate_tool_call("read_file", path="foo.txt")
    assert validation["valid"] is True


def test_validate_read_file_after_normalize(registry):
    normalized = registry._normalize_tool_kwargs("read_file", {"path": "foo.txt"})
    validation = registry.validate_tool_call("read_file", **normalized)
    assert validation["valid"] is True


def test_normalize_write_file_aliases(registry):
    args = registry._normalize_tool_kwargs(
        "write_file",
        {"path": "out.txt", "text": "hello", "tool_name": "write_file"},
    )
    assert args == {"file_path": "out.txt", "content": "hello"}


def test_validate_write_file_accepts_path_and_text_aliases(registry):
    validation = registry.validate_tool_call("write_file", path="notes.txt", body="line one")
    assert validation["valid"] is True
