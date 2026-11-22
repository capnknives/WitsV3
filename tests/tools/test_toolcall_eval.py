"""Tests for scripts/toolcall_eval.py using a mocked httpx transport."""

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "toolcall_eval.py"


@pytest.fixture(scope="module")
def ev():
    """Load the script by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("toolcall_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["toolcall_eval"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data(ev):
    return ev.load_cases(ev.CASES_PATH)


def _ollama(calls=(), content="", **extra):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"function": {"name": n, "arguments": a}} for n, a in calls]
    return {"message": msg, "done": True, **extra}


def _openai(calls=(), content="", finish="stop"):
    tcs = [
        {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}}
        for i, (n, a) in enumerate(calls)
    ]
    msg = {"role": "assistant", "content": content, "tool_calls": tcs or None}
    return {"choices": [{"message": msg, "finish_reason": finish}]}


def run(ev, data, backend, case_id, replies, think=None):
    """Replay one case against scripted replies; returns (row, requests sent)."""
    sent, queue = [], list(replies)

    def handler(request):
        sent.append(json.loads(request.content))
        item = queue.pop(0)
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)

    case = next(c for c in data["cases"] if c["id"] == case_id)

    async def go():
        async with httpx.AsyncClient(
            base_url="http://t", transport=httpx.MockTransport(handler)
        ) as client:
            return await ev.run_case(client, backend, "m", think, case, data, 5)

    return asyncio.run(go()), sent


S01 = {"title": "Dentist", "start": "2026-12-03T14:00", "duration_minutes": 45}


def test_cases_load_and_validate(ev, data):
    assert len(data["tools"]) == 8
    counts = {c: sum(1 for x in data["cases"] if x["category"] == c) for c in ev.CATEGORIES}
    assert counts == {"single": 12, "hard": 6, "none": 6, "parallel": 4, "chain": 4, "recovery": 4}
    props = {p for t in data["tools"] for p in t["parameters"]["properties"]}
    assert {"type", "description"} <= props
    bad = {
        "version": 2,
        "tools": data["tools"],
        "cases": [dict(data["cases"][0], expect=[{"tool": "read_file", "args": {"path": 3}}])],
    }
    assert ev.validate_cases(bad)
    assert data["version"] == 2 and ev.validate_cases(data) == []
    send = next(t for t in data["tools"] if t["name"] == "send_message")
    assert send["parameters"]["properties"]["recipient"]["description"].startswith("Name, email")


def test_validate_rejects_stray_schema_keys_and_incomplete_tools(ev, data):
    def check(tools=None, version=2):
        return ev.validate_cases({"version": version, "tools": tools or data["tools"], "cases": []})

    # The YAML slip that motivated this check: an unquoted comma in a flow mapping.
    stray = {"type": "object", "properties": {"r": {"type": "string", "email address": None}}}
    assert any("email address" in e for e in check([dict(data["tools"][0], parameters=stray)]))
    nested = {
        "type": "object",
        "properties": {"a": {"type": "array", "items": {"type": "x", "k": 1}}},
    }
    assert any(
        "a[]" in e and "'k'" in e for e in check([dict(data["tools"][0], parameters=nested)])
    )
    assert any("missing description" in e for e in check([{"name": "t", "parameters": stray}]))
    assert any("missing parameters" in e for e in check([{"name": "t", "description": "d"}]))
    assert any("missing name" in e for e in check([{"description": "d", "parameters": {}}]))
    for version in (None, "2", True, 2.0):
        assert any("version" in e for e in check(version=version))


@pytest.mark.parametrize(
    "backend,make",
    [("ollama", _ollama), ("openai", lambda c: _openai([(n, json.dumps(a)) for n, a in c]))],
)
def test_correct_call_passes(ev, data, backend, make):
    row, sent = run(ev, data, backend, "s01", [make([("calendar_create_event", S01)])])
    assert row["passed"] and not row["error"]
    assert sent[0]["tools"][0]["type"] == "function"
    assert sent[0]["messages"][-1]["content"].startswith("Create a calendar event")


def test_equal_numbers_and_extra_arguments(ev, data):
    ok = dict(S01, duration_minutes=45.0)
    assert run(ev, data, "ollama", "s01", [_ollama([("calendar_create_event", ok)])])[0]["passed"]
    # A schema-valid optional extra: strict failure, lenient pass.
    row, _ = run(
        ev,
        data,
        "ollama",
        "s01",
        [_ollama([("calendar_create_event", dict(S01, description="x"))])],
    )
    assert row["reason"] == "extra optional argument" and not row["passed"] and row["lenient"]
    # Not in the schema, or failing its schema: fails both ways as "unknown argument".
    for extra in ({"location": "x"}, {"description": 5}):
        row, _ = run(
            ev, data, "ollama", "s01", [_ollama([("calendar_create_event", dict(S01, **extra))])]
        )
        assert row["reason"] == "unknown argument" and not row["lenient"]


@pytest.mark.parametrize(
    "case_id,reply,reason",
    [
        ("s01", _ollama([("set_reminder", S01)]), "wrong tool"),
        (
            "s01",
            _ollama([("calendar_create_event", dict(S01, duration_minutes=40))]),
            "wrong arguments",
        ),
        (
            "s01",
            _ollama(content='{"name": "calendar_create_event", "arguments": {}}'),
            "tool call in text",
        ),
        ("s01", _ollama(content="Sure, done."), "no tool call"),
        ("n02", _ollama([("web_search", {"query": "capital of France"})]), "unexpected tool call"),
        ("s01", _ollama([("calendar_create_event", {})]), "empty arguments"),
        ("n02", _ollama(content="<think>hm</think>Paris"), "thinking leaked"),
        ("s01", _ollama(done_reason="length"), "truncated"),
    ],
)
def test_failure_reasons_ollama(ev, data, case_id, reply, reason):
    row, _ = run(ev, data, "ollama", case_id, [reply])
    assert not row["passed"] and not row["error"] and row["reason"] == reason


def test_openai_invalid_json_and_no_call_pass(ev, data):
    row, _ = run(ev, data, "openai", "s01", [_openai([("calendar_create_event", "{oops")])])
    assert row["reason"] == "invalid arguments JSON" and not row["error"]
    assert run(ev, data, "openai", "n02", [_openai(content="Paris.")])[0]["passed"]


def test_parallel_order_does_not_matter(ev, data):
    calls = [("read_file", {"path": "b.txt"}), ("read_file", {"path": "a.txt"})]
    assert run(ev, data, "ollama", "p02", [_ollama(calls)])[0]["passed"]
    row, _ = run(ev, data, "ollama", "p02", [_ollama(calls[:1])])
    assert row["reason"] == "wrong number of calls"


def test_chain_feeds_canned_result_back(ev, data):
    first = _ollama([("search_notes", {"query": "router password"})])
    send = {"recipient": "me@example.com", "type": "email", "body": "in the blue folder"}
    row, sent = run(ev, data, "ollama", "c01", [first, _ollama([("send_message", send)])])
    assert row["passed"] and len(sent) == 2
    assert sent[1]["messages"][-2]["role"] == "assistant"
    tool_msg = sent[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_name"] == "search_notes"
    assert "blue folder" in tool_msg["content"]
    # OpenAI variant carries the call id instead, and a wrong second call is a step failure.
    first = _openai([("search_notes", json.dumps({"query": "router"}))])
    row, sent = run(ev, data, "openai", "c01", [first, _openai(content="Done.")])
    assert sent[1]["messages"][-1]["tool_call_id"] == "c0"
    assert row["reason"] == "step 2: no tool call"


def test_server_error_and_timeout_are_errors_not_wrong_answers(ev, data):
    row, _ = run(ev, data, "ollama", "s01", [httpx.Response(500, text="boom")])
    assert row["error"] and "500" in row["error"] and not row["passed"] and not row["reason"]

    def slow(request):
        raise httpx.ReadTimeout("slow")

    case = data["cases"][0]

    async def go():
        async with httpx.AsyncClient(
            base_url="http://t", transport=httpx.MockTransport(slow)
        ) as client:
            return await ev.run_case(client, "ollama", "m", None, case, data, 5)

    assert "timeout" in asyncio.run(go())["error"]


def test_think_field_only_sent_when_set(ev, data):
    ok = _ollama([("calendar_create_event", S01)])
    assert run(ev, data, "ollama", "s01", [ok], think=False)[1][0]["think"] is False
    assert "think" not in run(ev, data, "ollama", "s01", [ok])[1][0]
    payload = run(ev, data, "ollama", "s01", [ok], think=True)[1][0]
    assert payload["stream"] is False and payload["options"] == {"temperature": 0, "num_ctx": 8192}


def test_exact_match_ignores_edge_whitespace_only(ev, data):
    padded = _ollama([("read_file", {"path": "\n  /home/user/todo.txt \n"})])
    assert run(ev, data, "ollama", "s04", [padded])[0]["passed"]
    inner = _ollama([("read_file", {"path": "/home/user/to do.txt"})])
    assert run(ev, data, "ollama", "s04", [inner])[0]["reason"] == "wrong arguments"
    # The expected side is stripped too, and `contains` / `any` behave as before.
    assert ev.args_match({"a": "x"}, {"args": {"a": " x\n"}}, False) is None
    assert (
        ev.args_match({"a": "xyz"}, {"args": {"a": "Y"}, "match": {"a": "contains"}}, False) is None
    )
    assert ev.args_match({"a": "q"}, {"args": {"a": ""}, "match": {"a": "any"}}, False) is None
    assert ev.args_match({"a": "x"}, {"args": {"a": ""}}, False) == "wrong arguments"


def test_recovery_and_none_cases_are_fair(data):
    for case in (c for c in data["cases"] if c["category"] == "recovery"):
        (first,), (second,) = case["expect"], case["then"][0]["expect"]
        assert "args" not in first and first["tool"] == second["tool"] and second["args"]
        err = case["then"][0]["result"]["error"]
        assert err.startswith("temporary failure") and "validation" not in err
    banned = ("capital", "translate", "calculate", "what is", "note", "file", "remind", "message")
    for case in (c for c in data["cases"] if c["category"] == "none"):
        assert not any(w in case["prompt"].lower() for w in banned), case["id"]


def _row(cid, passed, lenient=False, reason=None, error=None):
    return {
        "id": cid,
        "category": "single",
        "passed": passed,
        "lenient": lenient or passed,
        "reason": reason,
        "error": error,
        "latency": 1.0,
    }


def test_report_errors_count_as_failures_and_two_overall_columns(ev, tmp_path):
    rows = [
        _row("s01", True),
        _row("s02", False, reason="wrong tool"),
        _row("s03", False, lenient=True, reason="extra optional argument"),
        _row("s04", False, error="HTTP 500"),
    ]
    groups = [
        {"model": "m", "backend": "ollama", "think": "on", "rows": rows},
        {
            "model": "gone",
            "backend": "ollama",
            "think": "on",
            "rows": [],
            "note": "FAILED: HTTP 404",
        },
    ]
    text = ev.format_report(groups, "http://t", 1)
    assert "Overall (strict) | Overall (lenient)" in text
    assert "| m | ollama | on | 25% | - | - | - | - | - | 25% | 50% | 1.00 | 1 |" in text
    assert "s02 (wrong tool) x1" in text and "FAILED: HTTP 404" in text
    assert "Errored cases" in text and "- m [ollama, think=on]: s04 (HTTP 500) x1" in text
    assert "cases v" not in text and "cases v2" in ev.format_report(groups, "http://t", 1, 2)
    header = ev.format_report(groups, "http://t", 1, 2).splitlines()[0]
    assert header.startswith("## Tool-calling eval ") and header.endswith("| cases v2")
    out = tmp_path / "r.md"
    ev.append_results(out, "a\n")
    ev.append_results(out, "b\n")
    assert out.read_text() == "a\nb\n"


def _evaluate(ev, data, handler, runs=1, log_path=None):
    async def go():
        async with httpx.AsyncClient(
            base_url="http://t", transport=httpx.MockTransport(handler)
        ) as c:
            return await ev.evaluate(c, "ollama", ["m"], [None], runs, data, 5, log_path)

    return asyncio.run(go())


def test_group_aborts_after_five_consecutive_errors(ev, data):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if len(seen) == 1:
            return httpx.Response(200, json=_ollama([("calendar_create_event", S01)]))
        return httpx.Response(500, text="boom")

    (group,) = _evaluate(ev, data, handler)
    assert group["note"].startswith("ABORTED after 6 attempts: RuntimeError: HTTP 500")
    assert len(group["rows"]) == 6
    text = ev.format_report([group], "http://t", 1)
    assert "ABORTED after 6 attempts" in text and "Errored cases" in text
    assert "s02 (RuntimeError: HTTP 500: boom) x1" in text


def test_first_request_failure_is_fast_and_success_resets_streak(ev, data):
    (group,) = _evaluate(ev, data, lambda r: httpx.Response(404, text="model not found"))
    assert group["note"].startswith("FAILED: ") and len(group["rows"]) == 1
    n = []

    def flaky(request):  # 4 errors then an answer, repeatedly: never reaches 5 in a row.
        n.append(1)
        if len(n) % 5:
            return httpx.Response(200 if len(n) == 1 else 500, json=_ollama(content="hi"))
        return httpx.Response(200, json=_ollama(content="hi"))

    (group,) = _evaluate(ev, data, flaky)
    assert "note" not in group and len(group["rows"]) == len(data["cases"])


def test_failure_log_gets_one_line_per_failed_or_errored_attempt(ev, data, tmp_path):
    two = dict(data, cases=data["cases"][:2])  # s01, s02
    replies = [
        _ollama([("set_reminder", S01)], content="x" * 600),  # wrong tool
        httpx.Response(500, text="boom"),
    ]
    log = tmp_path / "out-failures.jsonl"
    log.write_text('{"keep": 1}\n')

    def handler(request):
        item = replies.pop(0)
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)

    _evaluate(ev, two, handler, log_path=log)
    keep, bad, err = [json.loads(x) for x in log.read_text().splitlines()]
    assert keep == {"keep": 1}
    assert (bad["model"], bad["backend"], bad["think"], bad["run"]) == ("m", "ollama", "n/a", 1)
    assert bad["case"] == "s01" and bad["step"] == 1 and bad["reason"] == "wrong tool"
    assert bad["expected"][0]["tool"] == "calendar_create_event" and bad["timestamp"]
    assert bad["actual"] == [{"name": "set_reminder", "args": S01}] and len(bad["content"]) == 500
    assert err["case"] == "s02" and "HTTP 500" in err["reason"] and err["actual"] == []
    assert err["expected"][0]["tool"] == "set_reminder" and err["content"] == ""


def test_failure_log_keeps_raw_arguments_and_step(ev, data, tmp_path):
    chain = dict(data, cases=[next(c for c in data["cases"] if c["id"] == "c01")])
    first = _openai([("search_notes", json.dumps({"query": "router"}))])
    replies = [first, _openai([("send_message", "{oops")])]
    log = tmp_path / "f.jsonl"

    async def go():
        async with httpx.AsyncClient(
            base_url="http://t",
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=replies.pop(0))),
        ) as c:
            return await ev.evaluate(c, "openai", ["m"], [None], 1, chain, 5, log)

    asyncio.run(go())
    (rec,) = [json.loads(x) for x in log.read_text().splitlines()]
    assert rec["step"] == 2 and rec["reason"] == "step 2: invalid arguments JSON"
    assert rec["actual"] == [{"name": "send_message", "args": "{oops"}]


def test_default_failure_log_path_and_runs_default(ev, tmp_path, monkeypatch):
    args = ev.build_parser().parse_args(["--models", "m"])
    assert args.runs == 1 and args.log_failures is None
    assert "temperature 0" in ev.build_parser().format_help()
    seen = {}

    async def fake(client, backend, models, modes, runs, data, timeout, log_path=None):
        seen["log"] = log_path
        return []

    monkeypatch.setattr(ev, "evaluate", fake)
    out = tmp_path / "res.md"
    args = ev.build_parser().parse_args(["--models", "m", "--out", str(out)])
    asyncio.run(ev.main_async(args))
    assert seen["log"] == tmp_path / "res-failures.jsonl"
    assert "cases v2" in out.read_text()
    args = ev.build_parser().parse_args(["--models", "m", "--out", str(out), "--log-failures", "x"])
    asyncio.run(ev.main_async(args))
    assert seen["log"] == Path("x")


def test_help_exits_zero_without_server():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, timeout=30)
    assert proc.returncode == 0 and b"--backend" in proc.stdout
