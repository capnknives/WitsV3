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
        "tools": data["tools"],
        "cases": [dict(data["cases"][0], expect=[{"tool": "read_file", "args": {"path": 3}}])],
    }
    assert ev.validate_cases(bad)


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
    out = tmp_path / "r.md"
    ev.append_results(out, "a\n")
    ev.append_results(out, "b\n")
    assert out.read_text() == "a\nb\n"


def _evaluate(ev, data, handler, runs=1):
    async def go():
        async with httpx.AsyncClient(
            base_url="http://t", transport=httpx.MockTransport(handler)
        ) as c:
            return await ev.evaluate(c, "ollama", ["m"], [None], runs, data, 5)

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
    assert "ABORTED after 6 attempts" in ev.format_report([group], "http://t", 1)


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


def test_help_exits_zero_without_server():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, timeout=30)
    assert proc.returncode == 0 and b"--backend" in proc.stdout
