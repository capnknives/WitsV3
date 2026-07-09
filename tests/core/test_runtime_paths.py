"""Tests for var/ runtime path layout and legacy migration."""

from pathlib import Path

import pytest

from core.runtime_paths import (
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
    for sub in ("data", "documents", "exports", "logs", "workspace", "cache", "sessions"):
        assert (tmp_path / "var" / sub).is_dir()


def test_merge_legacy_data_when_var_data_already_exists(tmp_path, monkeypatch):
    """Git checkout creates var/data/ templates; legacy data/ still has live files."""
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "data"
    target = tmp_path / "var" / "data"
    legacy.mkdir(parents=True)
    target.mkdir(parents=True)
    (legacy / "wits_memory.json").write_text("live-memory" * 1000, encoding="utf-8")
    (legacy / "guest_profiles.json").write_text('{"guests": {}}', encoding="utf-8")
    (legacy / "guest_audit").mkdir()
    (legacy / "guest_audit" / "g1").mkdir()
    (legacy / "guest_audit" / "g1" / "2026-07-08.jsonl").write_text("{}\n", encoding="utf-8")
    (target / "README.md").write_text("# templates", encoding="utf-8")
    (target / "mcp_tools.json").write_text("{}", encoding="utf-8")
    (target / "wits_memory.json").write_text("{}", encoding="utf-8")

    moved = migrate_legacy_runtime_dirs("var")

    assert not legacy.exists()
    assert (target / "wits_memory.json").read_text(encoding="utf-8").startswith("live-memory")
    assert (target / "guest_profiles.json").is_file()
    assert (target / "guest_audit" / "g1" / "2026-07-08.jsonl").is_file()
    assert (target / "README.md").read_text(encoding="utf-8") == "# templates"
    assert (target / "mcp_tools.json").is_file()
    assert any("merged" in m or "removed empty" in m for m in moved)


def test_merge_keeps_tracked_templates_over_legacy_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "data"
    target = tmp_path / "var" / "data"
    legacy.mkdir(parents=True)
    target.mkdir(parents=True)
    (legacy / "mcp_tools.json").write_text('{"from": "legacy"}', encoding="utf-8")
    (target / "mcp_tools.json").write_text('{"from": "repo"}', encoding="utf-8")

    migrate_legacy_runtime_dirs("var")

    assert (target / "mcp_tools.json").read_text(encoding="utf-8") == '{"from": "repo"}'
    assert not legacy.exists()


def test_merge_removes_empty_legacy_dir_when_only_stale_dupes(tmp_path, monkeypatch):
    """Legacy files smaller than var/ copies are dropped; legacy dir is removed."""
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "data"
    target = tmp_path / "var" / "data"
    legacy.mkdir(parents=True)
    target.mkdir(parents=True)
    (legacy / "knowledge_log.json").write_text("x", encoding="utf-8")
    (target / "knowledge_log.json").write_text("live" * 500, encoding="utf-8")

    migrate_legacy_runtime_dirs("var")

    assert not legacy.exists()
    assert (target / "knowledge_log.json").read_text(encoding="utf-8").startswith("live")


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
