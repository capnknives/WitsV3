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
