"""Tests for core.golden and scripts/golden_run.py (no Ollama)."""

import subprocess
import sys
from pathlib import Path

import pytest

from core.golden import GoldenCase, evaluate, load_cases
from core.schemas import StreamData

ROOT = Path(__file__).resolve().parents[2]
TURNS = ROOT / "tests/golden/turns.yaml"


def ev(type_: str, source: str = "WitsControlCenter", content: str = "x", **meta):
    return StreamData(type=type_, content=content, source=source, metadata=meta or None)


def case(**kw) -> GoldenCase:
    base = {"id": "c", "message": "m", "tags": ["chat"]}
    base.update(kw)
    return GoldenCase(**base)


def test_turns_yaml_loads():
    cases = load_cases(TURNS)
    assert len(cases) == 30
    assert len({c.id for c in cases}) == 30
    assert sum("follow-up" in c.tags for c in cases) == 3
    assert all(c.history for c in cases if "follow-up" in c.tags)


def test_tools_expectations():
    assert evaluate(case(tools="none"), [ev("result")]).passed
    assert not evaluate(
        case(tools="none"), [ev("tool_call", tool_name="read_file"), ev("result")]
    ).passed
    assert evaluate(case(tools="any"), [ev("action", tool_name="datetime"), ev("result")]).passed
    res = evaluate(case(tools="any"), [ev("result")])
    assert not res.passed and "not observable" in res.failures[0]
    want = case(tools=["read_file", "datetime"])
    assert not evaluate(want, [ev("tool_call", tool_name="read_file"), ev("result")]).passed
    both = [
        ev("tool_call", tool_name="read_file"),
        ev("action", tool_name="datetime"),
        ev("result"),
    ]
    assert evaluate(want, both).passed


def test_forbid_tools_and_routes():
    forbid = case(forbid_tools=["write_file"])
    assert not evaluate(forbid, [ev("tool_call", tool_name="write_file"), ev("result")]).passed
    assert evaluate(forbid, [ev("result")]).passed
    routes = case(forbid_routes=["book", "orchestrator", "self_repair"])
    assert not evaluate(routes, [ev("thinking", source="BookWriter"), ev("result")]).passed
    assert not evaluate(
        routes, [ev("thinking", source="LLMDrivenOrchestrator"), ev("result")]
    ).passed
    assert not evaluate(routes, [ev("thinking", source="SystemDoctor"), ev("result")]).passed
    assert evaluate(routes, [ev("result")]).passed


def test_reply_expectation():
    assert not evaluate(case(reply="required"), [ev("thinking")]).passed
    assert not evaluate(case(reply="required"), [ev("result", content="  ")]).passed
    assert evaluate(case(reply="optional"), [ev("thinking")]).passed


def test_owner_sentence_fails_on_coding_route():
    owner = next(c for c in load_cases(TURNS) if c.id == "owner-software-statement")
    assert evaluate(owner, [ev("result")]).passed
    bad = [ev("thinking", source="AdvancedCoder"), ev("result")]
    res = evaluate(owner, bad)
    assert not res.passed and "coding" in res.failures[0]


def test_validation_errors(tmp_path):
    def load(text):
        f = tmp_path / "t.yaml"
        f.write_text(text, encoding="utf-8")
        return load_cases(f)

    ok = "- id: a\n  message: hi\n  tags: [chat]\n  expect: {tools: none}\n"
    assert len(load(ok)) == 1
    with pytest.raises(ValueError, match="dup"):
        load(ok + ok.replace("hi", "yo"))
    with pytest.raises(ValueError, match="a: unknown key"):
        load(ok + "  bogus: 1\n")
    with pytest.raises(ValueError, match="a: bad reply"):
        load(ok.replace("tools: none", "reply: maybe"))
    with pytest.raises(ValueError, match="a: bad forbid_routes"):
        load(ok.replace("tools: none", "forbid_routes: [moon]"))
    with pytest.raises(ValueError, match="a: bad tags"):
        load(ok.replace("[chat]", "[nope]"))


def test_golden_run_help():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/golden_run.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
    )
    assert proc.returncode == 0
    assert "--runs" in proc.stdout


def test_tool_names_authoritative_and_unobserved():
    none_case = case(tools="none", forbid_tools=["write_file"])
    res = evaluate(none_case, [ev("result")])
    assert res.passed and res.unobserved == ["tools", "forbid_tools"]
    res = evaluate(none_case, [ev("result")], tool_names=[])
    assert res.passed and res.unobserved == []
    res = evaluate(none_case, [ev("result")], tool_names=["write_file"])
    assert not res.passed
    named = case(tools=["datetime"])
    assert evaluate(named, [ev("result")], tool_names=["datetime"]).passed
    over = [ev("tool_call", tool_name="datetime"), ev("result")]
    assert not evaluate(named, over, tool_names=["read_file"]).passed
