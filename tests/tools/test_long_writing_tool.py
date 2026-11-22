"""write_long_text: outline, one fresh call per section, saved as it goes (WP C5a). No Ollama."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

import core.filesystem_policy as fp
from core import runtime_paths
from core.config import WitsV3Config
from core.llm_interface import LLMResponse
from core.tool_call_guard import normalise_schema, validate
from core.tool_context import ToolContext, use_tool_context
from tools import long_writing_tool
from tools.long_writing_tool import LongWritingTool, parse_outline, section_count, slugify

BRIEF = "A dark story about an AI turning people against each other; humanity loses."


class ScriptedLLM:
    """Answers the outline call with ``outline`` and section calls from ``sections``."""

    def __init__(self, outline: str, sections=None, words: int = 450):
        self.outline, self.sections, self.words = outline, list(sections or []), words
        self.calls: list[dict] = []

    async def generate_chat(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), **kwargs})
        if "Plan a piece" in messages[-1].content:
            return LLMResponse(content=self.outline)
        if self.sections:
            return LLMResponse(content=self.sections.pop(0))
        n = len(self.calls) - 1
        return LLMResponse(content=" ".join(f"s{n}w{i}" for i in range(self.words)))


def numbered(n: int) -> str:
    return "Here is the outline:\n" + "\n".join(f"{i}. Part {i}: things happen" for i in range(n))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(fp, "PROJECT_ROOT", repo)
    monkeypatch.setattr(runtime_paths, "PROJECT_ROOT", repo)
    config = WitsV3Config()
    config.folders.base = str(tmp_path / "base")
    config.conversation.model = "resident:8b"

    def make(llm):
        tool = LongWritingTool()
        tool.set_dependencies(config, llm)
        return tool

    return make, tmp_path / "base" / "projects" / "writing", config


async def test_4500_words_ten_sections_saved_in_order(setup):
    make, folder, config = setup
    llm = ScriptedLLM(numbered(10))
    result = await make(llm).execute(title="The Great War", brief=BRIEF, target_words=4500)
    assert len(llm.calls) == 11
    assert result["success"] and result["sections"] == 10 and result["complete"]
    assert result["file"] == "writing/the-great-war.md"
    text = (folder / "the-great-war.md").read_text(encoding="utf-8")
    assert text.startswith("# The Great War\n\n")
    positions = [text.index(f"s{n}w0 ") for n in range(1, 11)]
    assert positions == sorted(positions)
    assert result["words"] == 4500 == len(text.split()) - 4
    assert len(json.dumps(result)) < 900 and "s5w10" not in json.dumps(result)
    assert result["opening"].split() == [f"s1w{i}" for i in range(60)]
    for call in llm.calls:
        assert call["model"] == "resident:8b" and call["think"] is False
        assert call["num_ctx"] == config.ollama_settings.num_ctx
    assert llm.calls[1]["num_predict"] == 990
    assert "bring the piece to its end" in llm.calls[-1]["messages"][-1].content
    assert "bring the piece to its end" not in llm.calls[5]["messages"][-1].content


async def test_section_prompts_stay_small_and_fresh(setup):
    make, _, _ = setup
    llm = ScriptedLLM(numbered(10))
    await make(llm).execute(title="The Great War", brief=BRIEF, target_words=4500)
    for n, call in enumerate(llm.calls[1:], start=1):
        prompt = "".join(m.content for m in call["messages"])
        assert len(call["messages"]) == 2 and len(prompt) < 6000
        assert BRIEF in prompt and "Part 9: things happen" in prompt
        assert f"s{n - 2}w0 " not in prompt  # earlier sections' text never comes back
        if n > 1:
            assert f"s{n - 1}w449" in prompt and f"s{n - 1}w300" not in prompt


@pytest.mark.parametrize(
    "outline",
    [
        "1) Start: a\n2) Middle: b\n3) End: c",
        "- Start: a\n* Middle: b\n\u2022 End: c",
        "Start: a\n\nMiddle: b\n\nEnd: c\n\nExtra: d",
        "**Start**: a\n### Middle\n<think>x: y</think>End: c",
    ],
)
def test_outline_parsing(outline):
    lines = parse_outline(outline, 3, "T")
    assert len(lines) == 3 and lines[0].startswith("Start") and lines[2].startswith("End")


def test_outline_fallback_and_counts():
    assert parse_outline("nothing useful here", 4, "War") == [
        f"Part {i} of War: continue the piece" for i in range(1, 5)
    ]
    assert parse_outline("1. Only: one", 5, "T") == ["Only: one"]
    assert [section_count(w) for w in (300, 450, 451, 4500, 6000)] == [1, 1, 2, 10, 14]
    assert section_count(6000, 5) == 5


async def test_short_section_retried_once_then_skipped(setup):
    make, folder, _ = setup
    long = " ".join(["word"] * 200)
    llm = ScriptedLLM(numbered(3), sections=[long, "too short", "still short", long])
    result = await make(llm).execute(title="Three Parts", brief=BRIEF, target_words=1200)
    assert len(llm.calls) == 5
    assert result["sections"] == 2 and result["words"] == 400 and not result["complete"]
    assert "short" not in (folder / "three-parts.md").read_text(encoding="utf-8")


async def test_time_limit_stops_after_current_section(setup):
    make, _, config = setup
    config.conversation.long_writing.max_seconds = 10
    llm = ScriptedLLM(numbered(10))
    tool = make(llm)
    tool.clock = lambda: len(llm.calls) * 4  # each model call "takes" four seconds
    result = await tool.execute(title="Cut Short", brief=BRIEF, target_words=4500)
    assert result["success"] and result["sections"] == 2 and result["words"] == 900
    assert result["complete"] is False and len(llm.calls) == 3


class SleepyLLM(ScriptedLLM):
    """Answers the outline and first section at once, then hangs on the next call."""

    async def generate_chat(self, messages, **kwargs):
        if len(self.calls) >= 2:
            self.calls.append({"messages": list(messages), **kwargs})
            await asyncio.sleep(30)
        return await super().generate_chat(messages, **kwargs)


async def test_time_limit_cuts_a_slow_section_without_retry(setup, monkeypatch):
    make, folder, config = setup
    monkeypatch.setattr(long_writing_tool, "MIN_CALL_SECONDS", 0.05)
    config.conversation.long_writing.max_seconds = 1
    llm = SleepyLLM(numbered(3))
    started = time.monotonic()
    result = await make(llm).execute(title="Slow One", brief=BRIEF, target_words=1200)
    assert time.monotonic() - started < 5
    assert len(llm.calls) == 3  # outline, section 1, section 2 cut off and not retried
    assert result["success"] and result["sections"] == 1 and result["complete"] is False
    assert len((folder / "slow-one.md").read_text(encoding="utf-8").split()) == 453


async def test_nothing_written_leaves_no_file(setup):
    make, folder, _ = setup
    llm = ScriptedLLM(numbered(2), sections=["", "", "no", "no"])
    result = await make(llm).execute(title="Empty", brief=BRIEF, target_words=900)
    assert result["success"] is False and "Nothing was written" in result["error"]
    assert not (folder / "empty.md").exists()


async def test_existing_file_is_never_overwritten(setup):
    make, folder, _ = setup
    folder.mkdir(parents=True)
    (folder / "the-war.md").write_text("mine", encoding="utf-8")
    result = await make(ScriptedLLM(numbered(1))).execute(title="The War", brief=BRIEF)
    assert result["file"] == "writing/the-war-2.md"
    assert (folder / "the-war.md").read_text(encoding="utf-8") == "mine"


async def test_path_characters_cannot_escape(setup):
    make, folder, _ = setup
    assert slugify("../../etc/passwd") == "etc-passwd"
    assert slugify("..\\..\\C:\\Windows") == "c-windows"
    assert slugify("???") == "untitled" and len(slugify("a" * 200)) == 60
    result = await make(ScriptedLLM(numbered(1))).execute(title="../../x/..", brief=BRIEF)
    assert result["file"] == "writing/x.md" and (folder / "x.md").exists()


async def test_write_outside_allowed_folders_is_refused(setup):
    make, folder, _ = setup
    folder.parent.mkdir(parents=True)
    folder.symlink_to(fp.PROJECT_ROOT, target_is_directory=True)  # writing/ -> the repo
    llm = ScriptedLLM(numbered(1))
    result = await make(llm).execute(title="Escape", brief=BRIEF)
    assert result["success"] is False and "Refusing" in result["error"]
    assert not llm.calls and not list(fp.PROJECT_ROOT.rglob("*.md"))


async def test_guest_is_refused(setup):
    make, folder, _ = setup
    llm = ScriptedLLM(numbered(1))
    with use_tool_context(ToolContext("s", "guest", None, None)):
        result = await make(llm).execute(title="Guest", brief=BRIEF)
    assert result["success"] is False and "owner" in result["error"] and not llm.calls


def test_guard_schema_rejects_bad_arguments():
    schema = normalise_schema(LongWritingTool().get_schema())
    assert validate({"title": "War", "brief": BRIEF, "target_words": 4500}, schema) == []
    for bad in (
        {"title": "War", "brief": BRIEF, "target_words": 50},
        {"title": "War", "brief": ""},
        {"title": "W", "brief": BRIEF},
        {"brief": BRIEF},
        {"title": "War", "brief": BRIEF, "path": "../x"},
    ):
        assert validate(bad, schema), bad
