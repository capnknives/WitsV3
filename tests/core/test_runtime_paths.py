"""Tests for var/ runtime path layout and legacy migration."""

from pathlib import Path

import pytest

from core.runtime_paths import (
    PROJECT_ROOT,
    ensure_runtime_layout,
    migrate_legacy_runtime_dirs,
    upgrade_runtime_path,
)


def test_upgrade_runtime_path_maps_legacy_roots():
    assert upgrade_runtime_path("data/wits_memory.json") == "var/data/wits_memory.json"
    assert upgrade_runtime_path("documents") == "var/documents"
    assert upgrade_runtime_path("logs/witsv3.log") == "var/logs/witsv3.log"
    assert upgrade_runtime_path("var/data/x.json") == "var/data/x.json"


def test_migrate_legacy_data_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "data"
    legacy.mkdir()
    (legacy / "wits_memory.json").write_text("{}", encoding="utf-8")

    moved = migrate_legacy_runtime_dirs("var")

    assert moved == ["data/ -> var/data/"]
    assert not legacy.exists()
    assert (tmp_path / "var" / "data" / "wits_memory.json").is_file()


def test_ensure_runtime_layout_creates_subdirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_runtime_layout("var")
    for sub in ("data", "documents", "exports", "logs", "workspace", "cache"):
        assert (tmp_path / "var" / sub).is_dir()


def test_load_config_upgrades_legacy_yaml_paths(tmp_path):
    from core.config import load_config

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
memory_manager:
  memory_file_path: data/wits_memory.json
document_rag:
  documents_path: documents
tool_system:
  mcp_tool_definitions_path: data/mcp_tools.json
""",
        encoding="utf-8",
    )
    config = load_config(str(cfg_file))
    assert config.memory_manager.memory_file_path == "var/data/wits_memory.json"
    assert config.document_rag.documents_path == "var/documents"
    assert config.tool_system.mcp_tool_definitions_path == "var/data/mcp_tools.json"
