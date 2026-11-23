"""Tests for golden failure files and the "why failed" column (no Ollama)."""

import asyncio
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

from core import golden_failures
from core.golden import GoldenCase
from core.schemas import StreamData

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("golden_run", ROOT / "scripts/golden_run.py")
golden_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden_run)

REPLY = "Sorry, I can't help.é | no"


class FakeControl:
    """Replies per message: 'hang' stalls after one result, anything else answers REPLY."""

    async def run(self, message, conversation_history=None):
        yield StreamData(type="result", content=REPLY, source="fake")
        if message == "hang":
            await asyncio.sleep(60)
        yield StreamData(type="error", content="boom", source="fake")


def make_args(tmp_path, **kw):
    base = dict(runs=2, timeout_seconds=0.3, out=str(tmp_path / "golden.md"))
    base.update(kw)
    return Namespace(**base)


def cases():
    return [
        GoldenCase(id="bad", message="say hi", tags=["chat"], reply_contains_any=["zzz"]),
        GoldenCase(id="good", message="hello", tags=["chat"]),
    ]


def test_failed_runs_write_files(tmp_path):
    box = tmp_path / "box"
    args = make_args(tmp_path, sandbox=str(box), live=False)
    asyncio.run(golden_run.run_cases(FakeControl(), cases(), args))
    folder = box / "failures"
    assert sorted(p.name for p in folder.iterdir()) == ["bad.run1.json", "bad.run2.json"]
    rec = json.loads((folder / "bad.run2.json").read_text(encoding="utf-8"))
    assert list(rec) == [
        "case_id",
        "run",
        "message",
        "failures",
        "reply",
        "reply_chars",
        "tools_seen",
        "first_result_ms",
        "eval_tokens",
    ]
    assert rec["case_id"] == "bad" and rec["run"] == 2 and rec["message"] == "say hi"
    assert rec["reply"] == REPLY + "\nboom"
    assert rec["reply_chars"] == len(rec["reply"])
    assert rec["failures"] and rec["tools_seen"] == []
    assert rec["first_result_ms"] is not None


def test_timeout_writes_file(tmp_path):
    box = tmp_path / "box"
    args = make_args(tmp_path, runs=1, sandbox=str(box), live=False)
    case = GoldenCase(id="slow", message="hang", tags=["chat"])
    asyncio.run(golden_run.run_cases(FakeControl(), [case], args))
    rec = json.loads((box / "failures/slow.run1.json").read_text(encoding="utf-8"))
    assert rec["failures"] == ["timeout after 0.3s"]
    assert rec["reply"] == REPLY


def test_live_mode_writes_no_failures_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(tmp_path, sandbox="", live=True)
    asyncio.run(golden_run.run_cases(FakeControl(), cases(), args))
    assert not list(tmp_path.rglob("failures"))


def test_table_has_why_failed_and_no_reply(tmp_path):
    args = make_args(tmp_path, sandbox=str(tmp_path / "box"), live=False)
    asyncio.run(golden_run.run_cases(FakeControl(), cases(), args))
    lines = (tmp_path / "golden.md").read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("| case"))
    assert header.endswith("| why failed |")
    good = next(line for line in lines if line.startswith("| good |"))
    bad = next(line for line in lines if line.startswith("| bad |"))
    assert good.endswith("| - |")
    assert len(bad.split(" | ")) == len(header.split(" | "))
    assert "reply_contains_any" in bad or "reply" in bad
    assert "Sorry" not in "\n".join(lines)


def test_stale_files_removed_and_unrelated_kept(tmp_path):
    folder = tmp_path / "box" / "failures"
    folder.mkdir(parents=True)
    (folder / "old.run1.json").write_text("{}", encoding="utf-8")
    (folder / "notes.txt").write_text("keep", encoding="utf-8")
    args = make_args(tmp_path, sandbox=str(tmp_path / "box"), live=False)
    only_good = [GoldenCase(id="good", message="hello", tags=["chat"])]
    asyncio.run(golden_run.run_cases(FakeControl(), only_good, args))
    assert not (folder / "old.run1.json").exists()
    assert (folder / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_shorten_reason():
    assert golden_failures.shorten_reason("a|b\nc") == "a/b c"
    long = golden_failures.shorten_reason("x" * 200)
    assert len(long) == 80 and long.endswith("...")


def test_replies_written_for_pass_and_fail(tmp_path):
    box = tmp_path / "box"
    args = make_args(tmp_path, runs=1, sandbox=str(box), live=False)
    asyncio.run(golden_run.run_cases(FakeControl(), cases(), args))
    folder = box / "replies"
    assert sorted(p.name for p in folder.iterdir()) == ["bad.run1.json", "good.run1.json"]
    good = json.loads((folder / "good.run1.json").read_text(encoding="utf-8"))
    bad = json.loads((folder / "bad.run1.json").read_text(encoding="utf-8"))
    assert good["passed"] is True and bad["passed"] is False
    assert good["reply"] == REPLY + "\nboom"
    fail = json.loads((box / "failures/bad.run1.json").read_text(encoding="utf-8"))
    assert "passed" not in fail
    assert {k: v for k, v in bad.items() if k != "passed"} == fail
    assert not (box / "failures/good.run1.json").exists()


def test_replies_not_written_in_live_or_without_sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for kw in (dict(sandbox="", live=True), dict(sandbox=str(tmp_path / "box"), live=True)):
        asyncio.run(golden_run.run_cases(FakeControl(), cases(), make_args(tmp_path, **kw)))
    assert not list(tmp_path.rglob("replies"))


def test_replies_cleared_at_start_and_unrelated_kept(tmp_path):
    folder = tmp_path / "box" / "replies"
    folder.mkdir(parents=True)
    (folder / "old.run1.json").write_text("{}", encoding="utf-8")
    (folder / "notes.txt").write_text("keep", encoding="utf-8")
    args = make_args(tmp_path, runs=1, sandbox=str(tmp_path / "box"), live=False)
    only_good = [GoldenCase(id="good", message="hello", tags=["chat"])]
    asyncio.run(golden_run.run_cases(FakeControl(), only_good, args))
    assert not (folder / "old.run1.json").exists()
    assert (folder / "good.run1.json").exists()
    assert (folder / "notes.txt").read_text(encoding="utf-8") == "keep"
