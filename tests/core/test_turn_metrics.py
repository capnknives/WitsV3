"""Tests for core.turn_metrics and its hooks in the Ollama interface (no Ollama needed)."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from core import turn_metrics as tm
from core.config import WitsV3Config
from core.llm_interface import OllamaInterface

FIELDS = {
    "ts", "turn_id", "session_id", "entry", "path", "input_chars", "first_event_ms",
    "first_result_ms", "first_token_ms", "wall_ms", "llm_calls", "tool_calls", "outcome",
}  # fmt: skip
SECRET = "TOPSECRET-message-text"


@pytest.fixture
def log_root(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "logs_dir", lambda: tmp_path)
    monkeypatch.setattr(tm, "_cached_settings", SimpleNamespace(enabled=True, retention_days=30))
    return tmp_path / "turn_metrics"


def _lines(directory):
    return [json.loads(x) for f in directory.glob("*.jsonl") for x in f.read_text().splitlines()]


async def test_full_turn_writes_one_line(log_root):
    turn = tm.start_turn("web", "s1", len(SECRET))
    tm.note_event("thinking")
    tm.record_llm_call(
        "chat",
        "m",
        12.5,
        {
            "load_duration": 2_000_000,
            "prompt_eval_duration": 3_000_000,
            "prompt_eval_count": 7,
            "eval_duration": 4_000_000,
            "eval_count": 9,
        },
    )
    tm.record_llm_call("generate", "m", 5.0, None)
    tm.record_first_token()
    tm.record_first_token()
    tm.record_tool("file_read", True, 1.5)
    tm.note_event("result")
    await tm.finish_turn(turn, "ok")
    rows = _lines(log_root)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == FIELDS
    assert row["turn_id"].startswith("trn_") and row["entry"] == "web"
    assert row["input_chars"] == len(SECRET) and row["outcome"] == "ok"
    assert row["first_event_ms"] is not None and row["first_result_ms"] is not None
    first, second = row["llm_calls"]
    assert first["load_ms"] == 2.0 and first["prompt_ms"] == 3.0 and first["eval_ms"] == 4.0
    assert first["prompt_tokens"] == 7 and first["eval_tokens"] == 9
    assert second["load_ms"] is None and second["eval_tokens"] is None
    assert row["tool_calls"] == [{"name": "file_read", "ok": True, "ms": 1.5}]
    assert tm.current_turn() is None


async def test_no_text_is_stored(log_root):
    turn = tm.start_turn("cli", None, len(SECRET))
    tm.record_tool("file_read", False, 1.0)
    tm.record_llm_call("chat", "m", 1.0, {"response": SECRET, "message": {"content": SECRET}})
    await tm.finish_turn(turn, "error")
    assert SECRET not in "".join(f.read_text() for f in log_root.glob("*.jsonl"))


async def test_noop_outside_turn_and_write_failure(log_root, monkeypatch):
    assert tm.current_turn() is None
    tm.record_llm_call("chat", "m", 1.0, {})
    tm.record_first_token()
    tm.record_tool("x", True, 1.0)
    tm.note_event("result")
    turn = tm.start_turn("web", None, 1)
    monkeypatch.setattr(tm, "_append", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    await tm.finish_turn(turn, "ok")  # must not raise


async def test_disabled_writes_nothing(log_root, monkeypatch):
    monkeypatch.setattr(tm, "_cached_settings", SimpleNamespace(enabled=False, retention_days=30))
    turn = tm.start_turn("web", None, 1)
    assert turn.turn_id.startswith("trn_")
    await tm.finish_turn(turn, "ok")
    assert not log_root.exists()


async def test_concurrent_turns_stay_apart(log_root):
    async def run(name):
        turn = tm.start_turn("web", name, 1)
        await asyncio.sleep(0.01)
        tm.record_tool(name, True, 1.0)
        await asyncio.sleep(0.01)
        await tm.finish_turn(turn, "ok")

    await asyncio.gather(run("a"), run("b"))
    rows = {r["session_id"]: r for r in _lines(log_root)}
    assert [t["name"] for t in rows["a"]["tool_calls"]] == ["a"]
    assert [t["name"] for t in rows["b"]["tool_calls"]] == ["b"]


async def test_retention_deletes_old_file_keeps_new(log_root):
    log_root.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    old = log_root / f"{(now - timedelta(days=40)).strftime('%Y-%m-%d')}.jsonl"
    new = log_root / f"{(now - timedelta(days=5)).strftime('%Y-%m-%d')}.jsonl"
    old.write_text("{}\n")
    new.write_text("{}\n")
    await tm.finish_turn(tm.start_turn("web", None, 1), "ok")
    assert not old.exists() and new.exists()


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _Client:
    def __init__(self, data):
        self.data = data

    async def post(self, *args, **kwargs):
        return _Resp(self.data)


class _StreamResp(_Resp):
    def __init__(self, lines):
        super().__init__(None)
        self.lines = lines

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class _StreamCtx:
    def __init__(self, lines):
        self.lines = lines

    async def __aenter__(self):
        return _StreamResp(self.lines)

    async def __aexit__(self, *exc):
        return False


class _StreamClient:
    def __init__(self, lines):
        self.lines = lines

    def stream(self, *args, **kwargs):
        return _StreamCtx(self.lines)


def _iface(client):
    iface = OllamaInterface(WitsV3Config())
    iface.http_client = client
    return iface


@pytest.mark.parametrize("durations", [True, False])
async def test_ollama_methods_unchanged(log_root, durations):
    extra = {"load_duration": 1_000_000, "eval_count": 3} if durations else {}
    turn = tm.start_turn("web", None, 1)

    gen = _iface(_Client({"response": "hello", **extra}))
    assert await gen.generate_text("p") == "hello"

    chat = _iface(
        _Client(
            {
                "message": {
                    "content": "hi",
                    "tool_calls": [{"function": {"name": "t", "arguments": {"a": 1}}}],
                },
                **extra,
            }
        )
    )
    result = await chat.generate_chat([])
    assert result.content == "hi"
    assert result.tool_calls == [{"name": "t", "arguments": {"a": 1}}]

    lines = [
        json.dumps({"response": "he", "done": False}),
        json.dumps({"response": "", "done": False}),
        json.dumps({"response": "llo", "done": False}),
        json.dumps({"done": True, **extra}),
    ]
    streamer = _iface(_StreamClient(lines))
    assert [c async for c in streamer.stream_text("p")] == ["he", "", "llo"]

    assert [c["endpoint"] for c in turn.llm_calls] == ["generate", "chat", "generate_stream"]
    assert turn.first_token_ms is not None
    assert (turn.llm_calls[0]["load_ms"] == 1.0) is durations
    await tm.finish_turn(turn, "ok")
