"""Tests for BookWritingAgent's real file-writing and continuation pipeline.

2026-07-08 finding: a live chat asked WITS to write a "100 page story" and
save it to disk. It never touched disk — every handler only called
store_memory() and streamed prose back into chat — and each follow-up
message ("okay, make it" / "write the whole story and save it to disk")
was re-analyzed as an unrelated fresh request instead of continuing the
book already outlined, so the story never got past ~1,500 words of
disconnected content. These tests cover the real capabilities added:
chapter-by-chapter generation that actually writes to workspace/, resumable
saves, and same-session continuation.
"""

import shutil
from collections.abc import AsyncGenerator

import pytest

from agents.book_writing_agent import BookWritingAgent
from agents.book_writing_helpers import (
    extract_requested_filename,
    is_continuation_phrase,
    wants_full_book_now,
)
from agents.book_writing_models import BookStructure
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.safe_code_editor import PROJECT_ROOT, resolve_within_project


class PromptRoutedLLM(BaseLLMInterface):
    """Returns different canned responses depending on what the prompt is
    asking for, so the full run() pipeline (classify -> structure -> outline
    -> per-chapter prose) can be exercised without a live model."""

    def __init__(self, config: WitsV3Config | None = None):
        super().__init__(config or WitsV3Config())
        self.calls: list[str] = []
        self.chapter_calls = 0

    async def generate_text(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        if "Analyze this book writing request" in prompt:
            return (
                '{"task_type": "create_book", "genre": "fantasy", "style": "narrative", '
                '"topic": "a knight", "length": 900, "filename": null, "parameters": {}}'
            )
        if "Create a detailed structure" in prompt:
            return (
                "Chapter 1: The Beginning\nA knight starts his journey.\n"
                "Chapter 2: The Middle\nThe knight faces trials.\n"
                "Chapter 3: The End\nThe knight triumphs.\n"
            )
        if "Create a detailed outline" in prompt:
            return "1. Opening hook\n2. Rising action\n3. Closing summary"
        if "Continue writing the" in prompt:
            self.chapter_calls += 1
            return f"Chapter prose number {self.chapter_calls}. " * 20
        return "generic response"

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield await self.generate_text(prompt, **kwargs)

    async def get_embedding(self, text: str, model: str | None = None) -> list[float]:
        return [0.0] * 8


def _cleanup(slug: str):
    shutil.rmtree(PROJECT_ROOT / "workspace" / slug, ignore_errors=True)


@pytest.fixture
def agent():
    return BookWritingAgent(
        agent_name="TestBookWriter",
        config=WitsV3Config(),
        llm_interface=PromptRoutedLLM(),
    )


# --------------------------------------------------------- helper regexes


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Save the story as TheBigStory01", "TheBigStory01"),
        ("please call it MyNovel and post it", "MyNovel"),
        ("write a story about a dragon", None),
    ],
)
def test_extract_requested_filename(text, expected):
    assert extract_requested_filename(text) == expected


def test_wants_full_book_now_matches_live_transcript_phrasing():
    assert wants_full_book_now("Write the whole story, and save it to disk.") is True
    assert wants_full_book_now("Save the story as TheBigStory01") is True
    assert wants_full_book_now("What genres do you support?") is False


@pytest.mark.parametrize(
    "text",
    ["Okay, so make it.", "write it all", "continue", "Finish the story.", "go ahead"],
)
def test_is_continuation_phrase_matches_short_followups(text):
    assert is_continuation_phrase(text) is True


def test_is_continuation_phrase_rejects_real_content():
    assert is_continuation_phrase("write a story about a knight who saves a village") is False


# ------------------------------------------------- _handle_write_full_book


@pytest.mark.asyncio
async def test_write_full_book_saves_all_chapters_to_disk(agent):
    slug = "scratch_full_book_test"
    try:
        book_id = "book-1"
        agent.current_books[book_id] = BookStructure(
            title="Scratch Test Book",
            genre="fantasy",
            target_length=900,
            chapters=[
                {"id": "c1", "title": "The Beginning", "outline": "Start.", "status": "planned"},
                {"id": "c2", "title": "The Middle", "outline": "Trials.", "status": "planned"},
                {"id": "c3", "title": "The End", "outline": "Triumph.", "status": "planned"},
            ],
        )

        streams = [s async for s in agent._handle_write_full_book(book_id, "sess-1", filename=slug)]

        output_path = resolve_within_project(f"workspace/{slug}/{slug}.md")
        assert output_path.exists()
        text = output_path.read_text(encoding="utf-8")
        assert "Chapter 1: The Beginning" in text
        assert "Chapter 2: The Middle" in text
        assert "Chapter 3: The End" in text
        assert "Chapter prose number 1" in text
        assert "Chapter prose number 3" in text

        assert all(c["status"] == "written" for c in agent.current_books[book_id].chapters)
        assert any("complete" in s.content for s in streams if s.type == "result")
        assert agent.writing_sessions["sess-1"]["active_book_id"] == book_id
    finally:
        _cleanup(slug)


@pytest.mark.asyncio
async def test_write_full_book_resumes_without_rewriting_finished_chapters(agent):
    slug = "scratch_resume_test"
    try:
        book_id = "book-2"
        agent.current_books[book_id] = BookStructure(
            title="Resume Test Book",
            genre="fantasy",
            target_length=600,
            chapters=[
                {"id": "c1", "title": "Chapter One", "outline": "x", "status": "written"},
                {"id": "c2", "title": "Chapter Two", "outline": "y", "status": "planned"},
            ],
        )
        output_path = resolve_within_project(f"workspace/{slug}/{slug}.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "# Resume Test Book\n\n## Chapter 1: Chapter One\n\nAlready written.\n",
            encoding="utf-8",
        )

        async for _ in agent._handle_write_full_book(book_id, "sess-2", filename=slug):
            pass

        text = output_path.read_text(encoding="utf-8")
        # Original chapter untouched, not duplicated
        assert text.count("Chapter One") == 1
        assert "Already written." in text
        # New chapter appended
        assert "Chapter 2: Chapter Two" in text
    finally:
        _cleanup(slug)


@pytest.mark.asyncio
async def test_write_full_book_no_book_found_yields_error(agent):
    streams = [s async for s in agent._handle_write_full_book("missing-book", "sess-3")]
    assert any(s.type == "error" for s in streams)


# ------------------------------------------------------------- run() flow


@pytest.mark.asyncio
async def test_run_writes_and_saves_story_in_one_turn_when_save_requested(agent):
    """Regression for the exact live-chat failure: a single request with
    topic + length + an explicit save name should produce a real saved file
    with actual chapter prose, not just chat text describing a plan."""
    slug = "scratch_one_turn_story"
    try:
        request = (
            "Please write the equivalent to a 100 page story, about a knight in a medieval "
            f"town who becomes powerful. Save the story as {slug}"
        )
        streams = [s async for s in agent.run(request, session_id="sess-live")]

        output_path = resolve_within_project(f"workspace/{slug}/{slug}.md")
        assert output_path.exists(), "story was never written to disk"
        text = output_path.read_text(encoding="utf-8")
        assert "Chapter prose number" in text
        assert any("saved to" in s.content for s in streams if s.type == "result")
    finally:
        _cleanup(slug)


@pytest.mark.asyncio
async def test_run_continuation_followup_resumes_same_book(agent):
    """Regression for "Okay, so make it." after an initial outline-only
    request (no filename/full-book keywords, so the first turn only builds
    structure): the follow-up must continue the same book instead of being
    judged as unrelated casual chat or starting a fresh, different one —
    the exact shape of the live-chat failure this pipeline was built for."""
    async for _ in agent.run("Write a story about a knight.", session_id="sess-cont"):
        pass
    book_id = agent.writing_sessions["sess-cont"]["active_book_id"]
    assert all(c["status"] != "written" for c in agent.current_books[book_id].chapters)

    slug = agent._slugify(agent.current_books[book_id].title)
    try:
        streams = [s async for s in agent.run("Okay, so make it.", session_id="sess-cont")]

        assert agent.writing_sessions["sess-cont"]["active_book_id"] == book_id
        assert any(c["status"] == "written" for c in agent.current_books[book_id].chapters)
        assert any("saved to" in s.content for s in streams if s.type == "result")
        assert resolve_within_project(f"workspace/{slug}/{slug}.md").exists()
    finally:
        _cleanup(slug)
