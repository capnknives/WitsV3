"""Safe defaults: no roots outside the repo, base folders, local list replaces base."""

from pathlib import Path

import yaml

from core.config import WitsV3Config
from core.filesystem_policy import configured_read_roots, document_ingest_roots
from core.runtime_paths import ensure_base_folders, ensure_runtime_layout


def _isolate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Reason: a relative folders.base anchors here, never at the cwd (WP F2b).
    monkeypatch.setenv("WITSV3_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("WITSV3_READ_ROOTS", raising=False)


def test_default_config_has_no_roots_outside_the_repo(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    cfg = WitsV3Config()
    assert cfg.security.filesystem_read_roots == []
    assert cfg.security.document_ingest_roots == []
    assert cfg.document_rag.auto_ingest_on_startup is False
    base = (tmp_path / "var/wits_files").resolve()
    repo = Path(__file__).resolve().parents[2]
    for root in configured_read_roots(cfg) + document_ingest_roots(cfg):
        assert root == repo or root.is_relative_to(base), root


def test_repo_config_yaml_has_empty_roots():
    data = yaml.safe_load((Path(__file__).resolve().parents[2] / "config.yaml").read_text("utf-8"))
    assert data["security"]["filesystem_read_roots"] == []
    assert data["security"]["document_ingest_roots"] == []
    assert data["document_rag"]["auto_ingest_on_startup"] is False


def test_base_folders_created_and_in_effective_lists(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    cfg = WitsV3Config()
    cfg.folders.base = str(tmp_path / "WITS")
    assert not (tmp_path / "WITS").exists()  # nothing at config time
    ensure_runtime_layout("var", cfg.folders.base)
    base = (tmp_path / "WITS").resolve()
    folders = ("inbox", "documents", "outbox", "projects")
    assert sorted(p.name for p in base.iterdir()) == sorted(folders)
    reads = configured_read_roots(cfg)
    assert {base / name for name in folders} <= set(reads)
    assert document_ingest_roots(cfg) == [base / "inbox", base / "documents"]
    assert base / "outbox" not in document_ingest_roots(cfg)


def test_relative_base_resolves_under_the_anchor_not_cwd(tmp_path, monkeypatch):
    (tmp_path / "cwd").mkdir()
    _isolate(monkeypatch, tmp_path / "cwd")
    monkeypatch.setenv("WITSV3_PROJECT_ROOT", str(tmp_path / "anchor"))
    ensure_base_folders("var/wits_files")
    assert (tmp_path / "anchor/var/wits_files/inbox").is_dir()
    assert not (tmp_path / "cwd/var").exists()


def test_local_root_list_replaces_base_list_and_inbox_stays(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    extra = tmp_path / "extra"
    cfg = WitsV3Config(security={"document_ingest_roots": [str(extra)]})
    fixed = [(tmp_path / "var/wits_files" / n).resolve() for n in ("inbox", "documents")]
    assert document_ingest_roots(cfg) == [*fixed, extra.resolve()]
    cfg.security.document_ingest_roots = []  # explicit empty = no extras, both still there
    assert document_ingest_roots(cfg) == fixed
