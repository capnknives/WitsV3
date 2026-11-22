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


class OwnerTool(LongWritingTool):
    """The tool as the legacy path calls it for the owner: the role is injected."""

    async def execute(self, **kwargs):
        kwargs.setdefault("user_role", "owner")
        return await super().execute(**kwargs)


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
        tool = OwnerTool()
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


class FailingLLM(ScriptedLLM):
    """Raises (or hangs, when ``hang``) on call number ``at`` (0 is the outline)."""

    def __init__(self, outline, at, hang=False):
        super().__init__(outline)
        self.at, self.hang, self.reached = at, hang, asyncio.Event()

    async def generate_chat(self, messages, **kwargs):
        if len(self.calls) == self.at:
            self.calls.append({"messages": list(messages), **kwargs})
            if self.hang:
                self.reached.set()
                await asyncio.sleep(60)
            raise ConnectionError("secret host 10.0.0.5 refused")
        return await super().generate_chat(messages, **kwargs)


@pytest.mark.parametrize("at", [0, 1])
async def test_model_error_before_first_section_leaves_no_file(setup, at):
    make, folder, _ = setup
    result = await make(FailingLLM(numbered(3), at)).execute(title="Broken", brief=BRIEF)
    assert result["success"] is False and "ConnectionError" in result["error"]
    assert "10.0.0.5" not in result["error"] and not list(folder.glob("*"))


async def test_model_error_after_first_section_keeps_partial(setup):
    make, folder, _ = setup
    result = await make(FailingLLM(numbered(3), 2)).execute(title="Half", brief=BRIEF)
    assert result["success"] and result["complete"] is False and result["sections"] == 1
    assert result["file"] == "writing/half.md" and "ConnectionError" in result["error"]
    assert (folder / "half.md").stat().st_size > 0


@pytest.mark.parametrize("at, kept", [(1, False), (2, True)])
async def test_cancel_removes_only_an_empty_file(setup, at, kept):
    make, folder, _ = setup
    llm = FailingLLM(numbered(3), at, hang=True)
    task = asyncio.ensure_future(make(llm).execute(title="Stopped", brief=BRIEF))
    await asyncio.wait_for(llm.reached.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (folder / "stopped.md").exists() is kept
    if kept:
        assert (folder / "stopped.md").stat().st_size > 0


async def test_role_must_be_known_owner(setup):
    make, folder, config = setup
    llm = ScriptedLLM(numbered(1))
    plain = LongWritingTool()
    plain.set_dependencies(config, llm)
    result = await plain.execute(title="No Role", brief=BRIEF)  # no context, nothing injected
    assert result["success"] is False and not llm.calls and not folder.exists()
    with use_tool_context(ToolContext("s", "owner", None, None)):
        result = await plain.execute(title="Owner Role", brief=BRIEF)
    assert result["success"] and result["file"] == "writing/owner-role.md"


class GatedLLM(ScriptedLLM):
    """Holds the first section of 'First' until ``gate`` opens."""

    def __init__(self, outline):
        super().__init__(outline)
        self.gate, self.held = asyncio.Event(), False

    async def generate_chat(self, messages, **kwargs):
        prompt = messages[-1].content
        if "'First'" in prompt and "Plan a piece" not in prompt and not self.held:
            self.held = True
            await self.gate.wait()
        return await super().generate_chat(messages, **kwargs)


async def test_concurrent_calls_keep_their_own_deadline(setup):
    make, _, config = setup
    llm = GatedLLM(numbered(5))
    tool = make(llm)
    now = {"t": 0.0}
    tool.clock = lambda: now["t"]

    async def first():
        config.conversation.long_writing.max_seconds = 10
        return await tool.execute(title="First", brief=BRIEF, target_words=2250)

    async def second():
        while not llm.held:
            await asyncio.sleep(0)
        now["t"] = 100.0  # starts later, with a longer limit, on the same instance
        config.conversation.long_writing.max_seconds = 1000
        result = await tool.execute(title="Second", brief=BRIEF, target_words=2250)
        llm.gate.set()
        return result

    one, two = await asyncio.gather(first(), second())
    assert two["sections"] == 5 and two["complete"]
    # Reason: a shared deadline would now read 1100 and let 'First' finish all five.
    assert one["sections"] == 1 and one["complete"] is False


def test_windows_reserved_names_get_a_prefix():
    for name in ("CON", "prn", "Aux", "nul", "COM1", "lpt9"):
        assert slugify(name) == f"text-{name.lower()}"
    assert slugify("console") == "console" and slugify("com10") == "com10"


async def test_worst_case_prompts_stay_bounded(setup):
    make, _, config = setup
    config.conversation.long_writing.max_sections = 14
    outline = "\n".join(f"{i}. Heading {i}: " + "x" * 600 for i in range(1, 15))
    llm = ScriptedLLM(outline)
    brief = "b" * 2000
    result = await make(llm).execute(title="T" * 120, brief=brief, target_words=6000)
    assert result["sections"] == 14 and len(llm.calls) == 15
    for call in llm.calls[1:]:
        prompt = "".join(m.content for m in call["messages"])
        assert len(prompt) < 7000 and prompt.count("Heading 3:") == 1
    schema = normalise_schema(LongWritingTool().get_schema())
    assert validate({"title": "War", "brief": "b" * 2001}, schema)
    assert validate({"title": "W" * 121, "brief": BRIEF}, schema)
