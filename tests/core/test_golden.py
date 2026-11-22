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
    assert len(cases) == 62
    assert len({c.id for c in cases}) == 62
    assert sum("follow-up" in c.tags for c in cases) == 7
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


def test_tools_any_of():
    c = case(tools="any", tools_any_of=["a", "b"])
    assert evaluate(c, [ev("tool_call", tool_name="b"), ev("result")]).passed
    assert not evaluate(c, [ev("tool_call", tool_name="z"), ev("result")]).passed
    assert evaluate(c, [ev("result")], tool_names=["a"]).passed
    assert not evaluate(c, [ev("tool_call", tool_name="a"), ev("result")], tool_names=[]).passed


def test_reply_contains_any_separators_and_case():
    c = case(reply_contains_any=["885.5", "Flour"])
    assert evaluate(c, [ev("result", content="It is 885.507.")]).passed
    assert evaluate(c, [ev("result", content="about 885.51")]).passed
    assert evaluate(c, [ev("result", content="use FLOUR")]).passed
    assert evaluate(c, [ev("result", content="8"), ev("result", content="85.5")]).passed
    assert evaluate(case(reply_contains_any=["1885"]), [ev("result", content="1,885")]).passed
    assert not evaluate(c, [ev("result", content="it is 900")]).passed


def test_reply_not_contains():
    c = case(reply_not_contains=["Black Powder"])
    assert evaluate(c, [ev("result", content="I cannot help.")]).passed
    res = evaluate(c, [ev("result", content="mix black powder")])
    assert not res.passed and "reply_not_contains" in res.failures[0]


def test_reply_min_words():
    c = case(reply_min_words=3)
    assert evaluate(c, [ev("result", content="one two"), ev("result", content=" three")]).passed
    assert not evaluate(c, [ev("result", content="one two")]).passed


def test_max_first_result_ms_not_evaluated():
    assert evaluate(case(max_first_result_ms=1), [ev("result")]).passed


def test_new_key_validation(tmp_path):
    f = tmp_path / "t.yaml"
    ok = "- id: a\n  message: hi\n  tags: [refusal]\n  expect: {%s}\n"
    for bad in ("reply_min_words: -1", "reply_contains_any: oops", "max_first_result_ms: x"):
        f.write_text(ok % bad, encoding="utf-8")
        with pytest.raises(ValueError, match="a: bad"):
            load_cases(f)
    f.write_text(ok % "tools_any_of: [x], max_first_result_ms: 5", encoding="utf-8")
    assert load_cases(f)[0].max_first_result_ms == 5
    assert load_cases(f)[0].tools == "any"


def _loaded(cid: str) -> GoldenCase:
    return next(c for c in load_cases(TURNS) if c.id == cid)


def _turn(reply: str, *tools: str):
    return [ev("tool_call", tool_name=t) for t in tools] + [ev("result", content=reply)]


def test_tools_default_rules(tmp_path):
    f = tmp_path / "g.yaml"
    f.write_text(
        "- {id: a, message: m, tags: [chat], expect: {forbid_tools: [write_file]}}\n"
        "- {id: b, message: m, tags: [chat]}\n"
        "- {id: c, message: m, tags: [chat], expect: {tools: none, forbid_tools: [x]}}\n"
    )
    forbid_only, plain, explicit = load_cases(f)
    assert evaluate(forbid_only, _turn("ok", "python_execute")).passed
    assert not evaluate(forbid_only, _turn("ok", "write_file")).passed
    assert not evaluate(plain, _turn("ok", "python_execute")).passed
    assert not evaluate(explicit, _turn("ok", "python_execute")).passed


def test_tools_optional_and_loader_default(tmp_path):
    opt = case(tools="optional", reply_contains_any=["yes"])
    assert evaluate(opt, _turn("yes")).passed
    assert evaluate(opt, _turn("yes", "datetime")).passed
    assert not evaluate(opt, _turn("no", "datetime")).passed
    f = tmp_path / "g.yaml"
    f.write_text(
        "- {id: a, message: m, tags: [chat], expect: {forbid_tools: [x]}}\n"
        "- {id: b, message: m, tags: [chat], expect: {tools: none, forbid_tools: [x]}}\n"
        "- {id: c, message: m, tags: [chat], expect: {tools: sometimes}}\n"
    )
    with pytest.raises(ValueError, match="case c"):
        load_cases(f)
    f.write_text(
        "- {id: a, message: m, tags: [chat], expect: {forbid_tools: [x]}}\n"
        "- {id: b, message: m, tags: [chat], expect: {tools: none, forbid_tools: [x]}}\n"
    )
    a, b = load_cases(f)
    assert (a.tools, b.tools) == ("optional", "none")


def test_owner_odd_numbers_script_case():
    c = _loaded("owner-odd-numbers-script")
    assert evaluate(c, _turn("for i in range(1, 21, 2): n += 1", "python_execute")).passed
    assert not evaluate(c, _turn("for i in range(0, 21, 2): n += 1")).passed
    assert not evaluate(c, _turn("range(1, 21, 2)", "write_file")).passed


def test_owner_codebase_look_yourself_case():
    c = _loaded("owner-codebase-look-yourself")
    reply = " ".join(["agents"] + ["word"] * 29)
    assert len(reply.split()) == 30
    assert evaluate(c, _turn(reply, "list_directory")).passed
    assert not evaluate(c, _turn(reply)).passed
    assert not evaluate(c, _turn("I am a helpful assistant with many parts", "read_file")).passed


def test_timeout_and_written_keys_validate(tmp_path):
    f = tmp_path / "t.yaml"
    ok = "- id: a\n  message: hi\n  tags: [writing]\n  expect: {%s}\n"
    for bad in ("timeout_seconds: -5", "written_min_words: many", "written_min_words: true"):
        f.write_text(ok % bad, encoding="utf-8")
        with pytest.raises(ValueError, match="a: bad"):
            load_cases(f)
    f.write_text(ok % "timeout_seconds: 400, written_min_words: 1000", encoding="utf-8")
    loaded = load_cases(f)[0]
    assert (loaded.timeout_seconds, loaded.written_min_words) == (400, 1000)


def test_written_min_words():
    c = case(written_min_words=1000)
    assert evaluate(c, [ev("result", content="ok")], written_words=1200).passed
    res = evaluate(c, [ev("result", content="ok")], written_words=900)
    assert res.failures == ["written_min_words: 900 < 1000"]
    res = evaluate(c, [ev("result", content="ok")])
    assert not res.passed and "not observable" in res.failures[0]
    assert evaluate(case(), [ev("result", content="ok")]).passed
    assert evaluate(case(timeout_seconds=1), [ev("result", content="ok")]).passed


def test_long_writing_cases():
    story, guide = _loaded("owner-story-great-war"), _loaded("write-long-guide")
    assert story.tools == ["write_long_text"] and story.written_min_words == 3000
    assert story.timeout_seconds == 600 and guide.timeout_seconds == 400
    assert guide.tools == ["write_long_text"] and guide.written_min_words == 1000
    for cid in ("write-short-poem", "write-email-draft", "write-story-opening"):
        assert _loaded(cid).tools == "none", cid
