"""Tests for the golden runner sandbox (no Ollama)."""

import asyncio
import importlib.util
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from core import runtime_paths, safe_code_editor
from core.config import WitsV3Config
from core.golden import GoldenCase
from core.schemas import StreamData

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("golden_run", ROOT / "scripts/golden_run.py")
golden_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden_run)


def owner_config() -> WitsV3Config:
    """A config shaped like the owner's: absolute roots outside any sandbox."""
    cfg = WitsV3Config()
    cfg.security.filesystem_read_roots = ["D:\\Downloads"]
    cfg.security.document_ingest_roots = ["D:\\Downloads", "/home/owner/docs"]
    cfg.memory_manager.memory_file_path = "/home/owner/mem.json"
    cfg.workspace.db_path = "D:\\wits\\workspace.db"
    return cfg


def test_sandbox_confines_every_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WITSV3_PROJECT_ROOT", raising=False)
    sandbox = tmp_path / "box"
    cfg = owner_config()
    assert golden_run.find_escapes(cfg, sandbox)  # the owner's roots really are outside
    before_root, before_cwd = safe_code_editor.PROJECT_ROOT, os.getcwd()
    with golden_run.enter_sandbox(cfg, sandbox) as project:
        assert golden_run.find_escapes(cfg, sandbox) == []
        assert cfg.security.filesystem_read_roots == [str(sandbox.resolve() / "files")]
        assert cfg.security.document_ingest_roots == cfg.security.filesystem_read_roots
        assert not cfg.document_rag.auto_ingest_on_startup
        assert not cfg.self_repair.restart_after_fix
        assert safe_code_editor.PROJECT_ROOT == project
        assert runtime_paths.data_dir().resolve().is_relative_to(sandbox.resolve())
        assert (project / "tests/test_dates.py").exists()
        assert not (project / "conftest.py").exists()
        assert (sandbox / "files/docs").is_dir()
    assert safe_code_editor.PROJECT_ROOT == before_root
    assert os.getcwd() == before_cwd
    assert "WITSV3_PROJECT_ROOT" not in os.environ


def test_load_config_inside_sandbox_finds_confined_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WITSV3_PROJECT_ROOT", raising=False)
    sandbox = tmp_path / "box"
    with golden_run.enter_sandbox(owner_config(), sandbox):
        capsys.readouterr()
        from core.config import load_config

        fresh = load_config()
        assert golden_run.find_escapes(fresh, sandbox) == []
        assert "not found" not in capsys.readouterr().out


def test_sandbox_refuses_unredirectable_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # An unredirectable source: the env root is forced outside by a fake runtime function.
    monkeypatch.setattr(golden_run, "_runtime_path_values", lambda root: [("x", "/etc/hosts")])
    with pytest.raises(golden_run.SandboxError):
        with golden_run.enter_sandbox(owner_config(), tmp_path / "box"):
            pass


def test_no_flags_exits_2_before_building():
    for extra in ([], ["--live"]):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/golden_run.py"), *extra],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert proc.returncode == 2
        assert "Traceback" not in proc.stderr


class HangingThenOk:
    async def run(self, message, conversation_history=None):
        if message == "hang":
            await asyncio.sleep(60)
        yield StreamData(type="result", content="ok", source="fake")


def test_timeout_cancels_case_and_continues(tmp_path):
    out = tmp_path / "golden.md"
    args = Namespace(runs=1, timeout_seconds=0.2, out=str(out))
    cases = [
        GoldenCase(id="hangs", message="hang", tags=["chat"]),
        GoldenCase(id="fine", message="hi", tags=["chat"]),
    ]
    code = asyncio.run(golden_run.run_cases(HangingThenOk(), cases, args))
    text = out.read_text(encoding="utf-8")
    assert "| hangs | 0/1 |" in text and "timeout" in text
    assert "| fine |" in text
    assert code in (0, 1)


def _outside_base_config(base: str) -> WitsV3Config:
    cfg = WitsV3Config()
    cfg.folders.base = base
    return cfg


def _assert_bases_inside(cfg, sandbox: Path) -> None:
    for fn in (
        runtime_paths.base_dir,
        runtime_paths.inbox_dir,
        runtime_paths.outbox_dir,
        runtime_paths.projects_dir,
    ):
        assert fn(cfg.folders.base).resolve().is_relative_to(sandbox.resolve())


@pytest.mark.parametrize("base", ["/home/owner/WITS", "D:\\WITS", "D:/WITS", "\\\\srv\\share\\w"])
def test_outside_folders_base_is_moved_into_sandbox(tmp_path, monkeypatch, base):
    from core.config import load_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WITSV3_PROJECT_ROOT", raising=False)
    sandbox = tmp_path / "box"
    cfg = _outside_base_config(base)
    assert golden_run.find_escapes(cfg, sandbox)
    with golden_run.enter_sandbox(cfg, sandbox):
        assert cfg.folders.base == str(sandbox.resolve() / "var" / "wits_files")
        _assert_bases_inside(cfg, sandbox)
        for name in ("inbox", "outbox", "projects"):
            assert (sandbox / "var" / "wits_files" / name).is_dir()
        assert not (sandbox / "config.local.yaml").exists()
        _assert_bases_inside(load_config(), sandbox)


def test_unrebased_folders_base_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(golden_run, "_rebase_base", lambda config, sandbox: None)
    with pytest.raises(golden_run.SandboxError, match=r"folders\.base"):
        with golden_run.enter_sandbox(_outside_base_config("/home/owner/WITS"), tmp_path / "box"):
            pass


def test_unknown_absolute_string_setting_is_an_escape(tmp_path):
    cfg = WitsV3Config()
    cfg.llm_interface.default_provider = "/home/owner/secret"  # any free-form string field
    bad = golden_run.find_escapes(cfg, tmp_path / "box")
    assert any("default_provider" in b for b in bad)
    cfg.llm_interface.default_provider = "E:\\elsewhere"
    assert any("default_provider" in b for b in golden_run.find_escapes(cfg, tmp_path / "box"))


def test_urls_and_default_config_are_not_flagged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WITSV3_PROJECT_ROOT", raising=False)
    assert not golden_run._is_absolute_path("http://localhost:11434")
    assert golden_run._is_absolute_path("D:\\x") and golden_run._is_absolute_path("/x")
    sandbox = tmp_path / "box"
    cfg = WitsV3Config()
    assert not any("11434" in b or "http" in b for b in golden_run.find_escapes(cfg, sandbox))
    with golden_run.enter_sandbox(cfg, sandbox):
        assert golden_run.find_escapes(cfg, sandbox) == []


def test_write_file_lands_in_sandbox_not_outside_base(tmp_path, monkeypatch):
    from tools.file_tools import FileWriteTool

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WITSV3_PROJECT_ROOT", raising=False)
    outside = tmp_path / "outside"
    outside.mkdir()
    sandbox = tmp_path / "box"
    with golden_run.enter_sandbox(_outside_base_config(str(outside)), sandbox):
        result = asyncio.run(FileWriteTool().execute("hello.txt", "hi"))
        assert "rror" not in result, result
    assert (sandbox / "var" / "wits_files" / "projects" / "hello.txt").read_text() == "hi"
    assert list(outside.iterdir()) == []


def test_sandbox_seeds_documents_where_the_document_tools_look(tmp_path, monkeypatch):
    from core.filesystem_policy import document_ingest_roots, documents_folder

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WITSV3_PROJECT_ROOT", raising=False)
    sandbox = (tmp_path / "box").resolve()
    cfg = owner_config()
    with golden_run.enter_sandbox(cfg, sandbox):
        assert golden_run.find_escapes(cfg, sandbox) == []
        docs = documents_folder(cfg).resolve()
        assert docs.is_relative_to(sandbox) and docs in document_ingest_roots(cfg)
        assert (docs / "docs" / "guide.txt").is_file()
