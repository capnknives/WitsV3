"""Tests for scripts/purge_intent_memory.py on a temporary memory file."""

import json

import pytest

from scripts import purge_intent_memory as purge


def _seg(i: int, seg_type: str) -> dict:
    return {"id": f"s{i}", "type": seg_type, "source": "t", "content": {"text": str(i)}}


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setattr(purge, "web_server_running", lambda **_: False)
    path = tmp_path / "mem.json"
    segs = [_seg(0, "USER_INPUT"), _seg(1, "INTENT_ANALYSIS"), _seg(2, "INTENT_ANALYSIS")]
    segs.append(_seg(3, "AGENT_THOUGHT"))
    path.write_text(json.dumps(segs))
    return path


def test_dry_run_changes_nothing(mem, capsys):
    before = mem.read_bytes()
    assert purge.main(["--memory-file", str(mem)]) == 0
    assert mem.read_bytes() == before
    assert list(mem.parent.glob("*.bak")) == []
    assert "2 of 4" in capsys.readouterr().out


def test_apply_removes_only_intent_rows_and_backs_up(mem):
    before = mem.read_bytes()
    assert purge.main(["--memory-file", str(mem), "--apply"]) == 0
    assert [s["id"] for s in json.loads(mem.read_text())] == ["s0", "s3"]
    backups = list(mem.parent.glob("mem.json.*.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == before


def test_wrapped_format_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(purge, "web_server_running", lambda **_: False)
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"segments": [_seg(0, "INTENT_ANALYSIS"), _seg(1, "X")]}))
    assert purge.main(["--memory-file", str(path), "--apply"]) == 0
    assert json.loads(path.read_text()) == {"segments": [_seg(1, "X")]}


def test_backup_failure_aborts_without_deleting(mem, monkeypatch):
    def boom(_path):
        raise OSError("disk full")

    monkeypatch.setattr(purge, "write_backup", boom)
    before = mem.read_bytes()
    assert purge.main(["--memory-file", str(mem), "--apply"]) == 1
    assert mem.read_bytes() == before


def test_refuses_while_server_running(mem, monkeypatch):
    monkeypatch.setattr(purge, "web_server_running", lambda **_: True)
    before = mem.read_bytes()
    assert purge.main(["--memory-file", str(mem), "--apply"]) == 3
    assert mem.read_bytes() == before
