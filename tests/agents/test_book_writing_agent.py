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
from core.runtime_paths import workspace_dir, workspace_subpath
from core.safe_code_editor import resolve_within_project


class PromptRoutedLLM(BaseLLMInterface):
    """Returns different canned responses depending on what the prompt is
    asking for, so the full run() pipeline (classify -> structure -> outline
    -> per-chapter prose) can be exercised without a live model."""

    def __init__(self, config: WitsV3Config | None = None):
        super().__init__(config or WitsV3Config())
        self.calls: list[str] = []
        self.call_kwargs: list[dict] = []
        self.chapter_calls = 0

    async def generate_text(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        self.call_kwargs.append(kwargs)
        if "Analyze this book writing request" in prompt:
            return (
                '{"task_type": "create_book", "genre": "fantasy", "style": "narrative", '
                '"topic": "a knight", "length": 900, "filename": null, "parameters": {}}'
            )
        if "Plan the chapter-by-chapter structure" in prompt:
            return (
                "Chapter 1: The Beginning\nA knight starts his journey.\n"
                "Chapter 2: The Middle\nThe knight faces trials.\n"
                "Chapter 3: The End\nThe knight triumphs.\n"
            )
        if "Create a detailed outline" in prompt:
            return "1. Opening hook\n2. Rising action\n3. Closing summary"
        if "Write a detailed scene-by-scene outline" in prompt:
            return "- Scene 1: setup\n- Scene 2: conflict\n- Scene 3: resolution"
        if "THIS PART must cover ONLY this beat" in prompt:
            self.chapter_calls += 1
            return f"Chapter prose number {self.chapter_calls}. " * 20
        return "generic response"

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield await self.generate_text(prompt, **kwargs)

    async def get_embedding(self, text: str, model: str | None = None) -> list[float]:
        return [0.0] * 8


def _cleanup(slug: str):
    shutil.rmtree(workspace_dir() / slug, ignore_errors=True)


@pytest.fixture
def agent():
    return BookWritingAgent(
        agent_name="TestBookWriter",
        config=WitsV3Config(),
        llm_interface=PromptRoutedLLM(),
    )


# ---------------------------------------------- chapter structure parsing


@pytest.mark.asyncio
async def test_extract_chapters_ignores_non_chapter_section_headers(agent):
    """2026-07-08 live-model finding: a structure response that also
    described title/audience/themes/research/style produced markdown
    section headers ("### Target Audience") that the old bare
    `line.startswith("#")` check misparsed as chapters, yielding chapters
    with no real outline that generated near-empty prose."""
    response = (
        "### Title and Subtitle\n"
        "The Knight's Tale\n\n"
        "### Chapter Breakdown\n"
        "Chapter 1: The Beginning\n"
        "A knight starts his journey.\n\n"
        "Chapter 2: The Middle\n"
        "The knight faces trials.\n\n"
        "### Target Audience\n"
        "Young adults.\n\n"
        "### Research Requirements\n"
        "Medieval history.\n"
    )
    chapters = await agent._extract_chapters_from_response(response, "book-x")
    titles = [c["title"] for c in chapters]
    assert titles == ["The Beginning", "The Middle"]
    assert "starts his journey" in chapters[0]["outline"]


@pytest.mark.asyncio
async def test_extract_chapters_strips_duplicated_chapter_prefix(agent):
    """The extracted title must not still carry "Chapter N:" — the writer
    pipeline re-adds its own "Chapter N:" prefix when saving, so keeping it
    here produced doubled headers like "## Chapter 1: Chapter 1: Title"."""
    response = "Chapter 3: The Reckoning\nEverything comes to a head.\n"
    chapters = await agent._extract_chapters_from_response(response, "book-y")
    assert chapters[0]["title"] == "The Reckoning"


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


# --------------------------------------------------------- length estimation


@pytest.mark.parametrize(
    "text,default,expected",
    [
        ("write a 100 page story", 1000, 25000),
        ("write a 5000 word story", 1000, 5000),
        ("write me a novella about a knight", 1000, 25000),
        ("write me a novel about a knight", 1000, 80000),
        ("write me a short story about a knight", 1000, 4000),
        ("write me a story about a knight", 1000, 1000),
    ],
)
def test_estimate_requested_length_prefers_explicit_over_named_form(agent, text, default, expected):
    """Explicit page/word counts win; a named form ("novella") sets a
    realistic default when nothing explicit is given — 2026-07-08 finding:
    chapters/books were consistently coming out far short of what the named
    form actually implies (a "novella" is 17.5k-40k words, not ~1-2k)."""
    assert agent._estimate_requested_length(text, default) == expected


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

        output_path = resolve_within_project(f"{workspace_subpath(slug)}/{slug}.md")
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
        output_path = resolve_within_project(f"{workspace_subpath(slug)}/{slug}.md")
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

        output_path = resolve_within_project(f"{workspace_subpath(slug)}/{slug}.md")
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
        assert resolve_within_project(f"{workspace_subpath(slug)}/{slug}.md").exists()
    finally:
        _cleanup(slug)


# --------------------------------------------------- context window / length


@pytest.mark.asyncio
async def test_write_full_book_passes_num_ctx_to_llm_calls(agent):
    """2026-07-08 finding: Ollama's own runtime context window default is
    much smaller than most models' trained context length regardless of
    max_tokens, and silently truncates a large prompt+completion for
    novella-length chapters unless num_ctx is set explicitly on the call."""
    slug = "scratch_num_ctx_test"
    try:
        book_id = "book-ctx"
        agent.current_books[book_id] = BookStructure(
            title="Ctx Test",
            genre="fantasy",
            target_length=2200,
            chapters=[{"id": "c1", "title": "One", "outline": "x", "status": "planned"}],
        )
        async for _ in agent._handle_write_full_book(book_id, "sess-ctx", filename=slug):
            pass

        expected = agent.config.agents.book_writing_agent.num_ctx
        assert expected > 0
        assert any(kw.get("num_ctx") == expected for kw in agent.llm_interface.call_kwargs)
    finally:
        _cleanup(slug)


@pytest.mark.asyncio
async def test_write_full_book_targets_novella_density_per_chapter(agent):
    """The per-chapter floor must reflect real novella density (2026-07-08
    finding: chapters were consistently coming out short), not the old
    900-word floor. Chapters are written beat-by-beat (see
    test_write_full_book_generates_one_call_per_outline_beat), so the
    per-beat word target should be the chapter target divided across the
    mocked 3-beat outline: 2200 // 3 = 733."""
    slug = "scratch_density_test"
    try:
        book_id = "book-density"
        agent.current_books[book_id] = BookStructure(
            title="Density Test",
            genre="fantasy",
            target_length=2200,  # low total, so the floor (not the average) governs
            chapters=[{"id": "c1", "title": "One", "outline": "x", "status": "planned"}],
        )
        async for _ in agent._handle_write_full_book(book_id, "sess-density", filename=slug):
            pass

        writing_calls = [
            c for c in agent.llm_interface.calls if "THIS PART must cover ONLY this beat" in c
        ]
        assert any("Write about 733 words" in c for c in writing_calls)
    finally:
        _cleanup(slug)


@pytest.mark.asyncio
async def test_write_full_book_scales_max_tokens_to_beat_target(agent):
    """2026-07-08 finding: every beat call was passed the full chapter
    max_tokens (e.g. 4500), so nothing capped how far a single beat could
    overrun its own word target -- observed live as a 2200-word-target
    chapter landing at 4124 words. Each beat call's max_tokens must scale
    down with that beat's own word target instead of the whole chapter's
    ceiling."""
    slug = "scratch_beat_tokens_test"
    try:
        # A generous chapter-level ceiling, matching config.yaml's real
        # value -- with the old bug this alone was passed to every beat
        # call regardless of that beat's own target.
        agent.config.agents.book_writing_agent.max_tokens = 4500
        book_id = "book-beat-tokens"
        agent.current_books[book_id] = BookStructure(
            title="Beat Tokens Test",
            genre="fantasy",
            target_length=2200,  # mocked 3-beat outline -> 733 words/beat
            chapters=[{"id": "c1", "title": "One", "outline": "x", "status": "planned"}],
        )
        async for _ in agent._handle_write_full_book(book_id, "sess-beat-tokens", filename=slug):
            pass

        writing_kwargs = [
            kw
            for prompt, kw in zip(
                agent.llm_interface.calls, agent.llm_interface.call_kwargs, strict=True
            )
            if "THIS PART must cover ONLY this beat" in prompt
        ]
        assert writing_kwargs
        # words_per_beat = 733 -> min(4500, max(400, int(733 * 2.3))) = 1685
        for kw in writing_kwargs:
            assert kw.get("max_tokens") == 1685
    finally:
        _cleanup(slug)


def test_prepare_payload_includes_num_ctx():
    from core.llm_interface import OllamaInterface

    async def _run():
        interface = OllamaInterface(config=WitsV3Config())
        payload = await interface._prepare_payload("hi", num_ctx=8192)
        assert payload["options"]["num_ctx"] == 8192

    import asyncio

    asyncio.run(_run())


# ------------------------------------------------- point-of-view consistency


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I walked to the door. My hands were shaking. I opened it.", "first"),
        ("He walked to the door. His hands were shaking. She watched him.", "third"),
        ("", None),
    ],
)
def test_detect_pov_from_continuity_text(agent, text, expected):
    pov = agent._detect_pov(text, "")
    if expected is None:
        assert pov is None
    else:
        assert (expected == "first") == ("first person" in (pov or ""))


def test_detect_pov_falls_back_to_world_bible_when_no_prior_text(agent):
    assert "first person" in agent._detect_pov("", "Primary perspective is first person.")
    assert agent._detect_pov("", "Told in third person throughout.") == "third person"
    assert agent._detect_pov("", "No POV notes here.") is None


@pytest.mark.asyncio
async def test_write_full_book_enforces_pov_across_chapters(agent):
    """2026-07-08 live-model finding: a first-person prologue/Chapter 1 was
    followed by Chapter 2 silently drifting into third person — "matching
    the voice already established" alone wasn't a strong enough
    instruction. Once there's first-person continuity text on disk, later
    planning/writing prompts must carry an explicit POV directive."""
    slug = "scratch_pov_test"
    try:
        book_id = "book-pov"
        agent.current_books[book_id] = BookStructure(
            title="POV Test",
            genre="drama",
            target_length=900,
            chapters=[
                {"id": "c1", "title": "One", "outline": "x", "status": "planned"},
                {"id": "c2", "title": "Two", "outline": "y", "status": "planned"},
            ],
        )
        await agent.seed_existing_chapter(
            book_id, "sess-pov", "Prologue", "I stood alone. My hands shook.", filename=slug
        )
        async for _ in agent._handle_write_full_book(book_id, "sess-pov", filename=slug):
            pass

        planning_calls = [
            c for c in agent.llm_interface.calls if "Write a detailed scene-by-scene outline" in c
        ]
        writing_calls = [
            c for c in agent.llm_interface.calls if "THIS PART must cover ONLY this beat" in c
        ]
        assert any("Narrate in first person" in c for c in writing_calls)
        assert any("point of view" in c.lower() for c in planning_calls)
    finally:
        _cleanup(slug)


def test_strip_leading_title_echo_removes_repeated_title(agent):
    content = "Rick Discovers His Powers\n\nThe sky was still dark when..."
    result = agent._strip_leading_title_echo(content, "Rick Discovers His Powers")
    assert result.strip().startswith("The sky was still dark")


def test_strip_leading_title_echo_leaves_real_prose_alone(agent):
    content = "The sky was still dark when Rick woke up."
    result = agent._strip_leading_title_echo(content, "Rick Discovers His Powers")
    assert result == content


# ------------------------------------------------------ outline beat parsing


def test_parse_outline_beats_folds_wrapped_lines_into_one_beat(agent):
    """2026-07-08 finding: an outline whose bullets wrapped/elaborated
    across multiple lines got parsed as one beat per *line*, producing 30+
    "beats" for a single chapter and an 8,000+ word chapter against a
    2,200-word target. Only an actual bullet/numbered marker should start a
    new beat; anything else is a continuation of the previous one."""
    outline = (
        "- Scene 1: Rick wakes up\n"
        "  He reflects on last night's failed jump.\n"
        "  His muscles ache.\n"
        "- Scene 2: A stranger arrives\n"
        "  She refuses to explain herself at first.\n"
    )
    beats = agent._parse_outline_beats(outline)
    assert beats == [
        "Scene 1: Rick wakes up He reflects on last night's failed jump. His muscles ache.",
        "Scene 2: A stranger arrives She refuses to explain herself at first.",
    ]


def test_parse_outline_beats_caps_beat_count(agent):
    outline = "\n".join(f"- Beat {i}" for i in range(1, 21))
    beats = agent._parse_outline_beats(outline)
    assert len(beats) == agent._MAX_BEATS_PER_CHAPTER


def test_parse_outline_beats_falls_back_to_whole_text_with_no_markers(agent):
    outline = "Just some unstructured prose with no bullets at all."
    assert agent._parse_outline_beats(outline) == [outline]


# ------------------------------------------------- repeated-paragraph removal


def test_dedupe_repeated_paragraphs_drops_near_duplicate(agent):
    """2026-07-08 live-model finding: beat-by-beat generation produced a
    chapter where a six-paragraph negotiation-and-agreement scene appeared
    twice, near-verbatim, back to back -- each beat is a separate LLM call,
    so nothing forced the model to notice it was repeating itself."""
    first = "Sir Galahad considered Alaric's words carefully before nodding."
    second = "The squire nodded back, eager to begin his training at last."
    content = f"{first}\n\n{second}\n\n{first}"
    result = agent._dedupe_repeated_paragraphs(content)
    assert result == f"{first}\n\n{second}"


def test_dedupe_repeated_paragraphs_keeps_distinct_content(agent):
    content = "The knight drew his sword.\n\nThe squire watched in awe."
    assert agent._dedupe_repeated_paragraphs(content) == content


def test_dedupe_repeated_paragraphs_drops_matches_against_reference(agent):
    """2026-07-08, second finding: the model's "bring the chapter to a
    close" instruction produces a formulaic inspirational-monologue
    ending, and since the next chapter's prompt includes the previous
    chapter's tail as "story so far" context, it sometimes reproduced that
    same closing block almost word-for-word one chapter later -- a repeat
    *across* chapters, which per-chapter-only dedup couldn't catch."""
    ending = "And so, with renewed determination, he stepped into the unknown once more."
    reference = f"Some earlier chapter content.\n\n{ending}"
    new_chapter = f"Fresh new content for this chapter.\n\n{ending}"
    result = agent._dedupe_repeated_paragraphs(new_chapter, reference=reference)
    assert result == "Fresh new content for this chapter."


# --------------------------------------------- incomplete-sentence trimming


def test_trim_incomplete_sentence_drops_trailing_fragment(agent):
    """2026-07-08 live-model finding: a token-capped beat call sometimes
    stopped mid-sentence ("...he forced himself to"), and a chapter once
    ended on the bare word "The" -- both must be trimmed back to the last
    complete sentence rather than left dangling."""
    text = "He drew his sword and turned to face the door. Robin followed suit, unleashing"
    assert agent._trim_incomplete_sentence(text) == (
        "He drew his sword and turned to face the door."
    )


def test_trim_incomplete_sentence_leaves_complete_text_alone(agent):
    text = "He drew his sword. Robin followed close behind."
    assert agent._trim_incomplete_sentence(text) == text


def test_trim_incomplete_sentence_falls_back_when_no_complete_sentence_exists(agent):
    text = "no punctuation at all here"
    assert agent._trim_incomplete_sentence(text) == text


# -------------------------------------------------- meta-commentary removal


def test_strip_meta_commentary_drops_self_referential_paragraph(agent):
    """2026-07-08 live-model finding: despite the prompt's explicit "no
    meta-commentary" instruction, one beat's output opened with "(The
    critical narrative prose section comes to an end here -- the chapter
    outline contains 4 beats after this one that must be covered in
    subsequent outputs." -- breaking character instead of writing story."""
    real_prose = "Alaric drew his sword and faced the bandits."
    meta = (
        "(The critical narrative prose section comes to an end here — the "
        "chapter outline contains 4 beats after this one that must be "
        "covered in subsequent outputs."
    )
    content = f"{real_prose}\n\n{meta}\n\n{real_prose}"
    result = agent._strip_meta_commentary(content)
    assert meta not in result
    assert result == f"{real_prose}\n\n{real_prose}"


def test_strip_meta_commentary_keeps_real_parenthetical_prose(agent):
    content = "He drew his sword. (He had trained for this moment for years.)"
    assert agent._strip_meta_commentary(content) == content


# --------------------------------------------------- outline expansion & continuity


@pytest.mark.asyncio
async def test_write_full_book_expands_outline_before_writing_prose(agent):
    """Each chapter should get a dedicated "plan it" call before the "write
    it" call — a one-liner alone produced thin, disconnected scenes."""
    slug = "scratch_outline_expansion"
    try:
        book_id = "book-outline"
        agent.current_books[book_id] = BookStructure(
            title="Outline Expansion Test",
            genre="fantasy",
            target_length=900,
            chapters=[
                {"id": "c1", "title": "Arrival", "outline": "x", "status": "planned"},
            ],
        )
        async for _ in agent._handle_write_full_book(book_id, "sess-outline", filename=slug):
            pass

        planning_calls = [
            c for c in agent.llm_interface.calls if "Write a detailed scene-by-scene outline" in c
        ]
        writing_calls = [
            c for c in agent.llm_interface.calls if "THIS PART must cover ONLY this beat" in c
        ]
        assert len(planning_calls) == 1
        # One prose call per outline beat (the mocked outline has 3), not
        # one call for the whole chapter — see _write_chapter_prose.
        assert len(writing_calls) == 3
        # The prose calls must actually be grounded in the expanded outline,
        # not just the original one-line summary.
        assert "Scene 1: setup" in writing_calls[0]
    finally:
        _cleanup(slug)


@pytest.mark.asyncio
async def test_write_full_book_passes_continuity_tail_between_chapters(agent):
    """The second chapter's prompts must include text from the first
    chapter's saved output, so chapter 2 continues chapter 1 instead of
    starting a disconnected scene."""
    slug = "scratch_continuity_test"
    try:
        book_id = "book-continuity"
        agent.current_books[book_id] = BookStructure(
            title="Continuity Test",
            genre="fantasy",
            target_length=900,
            chapters=[
                {"id": "c1", "title": "Chapter One", "outline": "x", "status": "planned"},
                {"id": "c2", "title": "Chapter Two", "outline": "y", "status": "planned"},
            ],
        )
        async for _ in agent._handle_write_full_book(book_id, "sess-continuity", filename=slug):
            pass

        planning_calls = [
            c for c in agent.llm_interface.calls if "Write a detailed scene-by-scene outline" in c
        ]
        assert len(planning_calls) == 2
        # Chapter 1 has nothing before it; chapter 2's planning call must see
        # chapter 1's actual generated text as "STORY SO FAR" context.
        assert "STORY SO FAR" not in planning_calls[0]
        assert "STORY SO FAR" in planning_calls[1]
        assert "Chapter prose number 1" in planning_calls[1]
    finally:
        _cleanup(slug)


@pytest.mark.asyncio
async def test_write_full_book_injects_world_bible_into_prompts(agent):
    slug = "scratch_world_bible_injection"
    try:
        book_id = "book-bible"
        agent.current_books[book_id] = BookStructure(
            title="Bible Injection Test",
            genre="fantasy",
            target_length=900,
            chapters=[{"id": "c1", "title": "Arrival", "outline": "x", "status": "planned"}],
            world_bible="Protagonist: Rick. Powers are tied to belief and pressure.",
        )
        async for _ in agent._handle_write_full_book(book_id, "sess-bible", filename=slug):
            pass

        planning_calls = [
            c for c in agent.llm_interface.calls if "Write a detailed scene-by-scene outline" in c
        ]
        writing_calls = [
            c for c in agent.llm_interface.calls if "THIS PART must cover ONLY this beat" in c
        ]
        assert "Powers are tied to belief and pressure" in planning_calls[0]
        assert "Powers are tied to belief and pressure" in writing_calls[0]
    finally:
        _cleanup(slug)


# ------------------------------------------------------------- seeding existing content


@pytest.mark.asyncio
async def test_seed_existing_chapter_writes_immediately_and_is_skipped_later(agent):
    """An already-written prologue the user supplies must land on disk right
    away and never be regenerated by _handle_write_full_book."""
    slug = "scratch_seed_test"
    try:
        book_id = "book-seed"
        agent.current_books[book_id] = BookStructure(
            title="Seed Test", genre="fantasy", target_length=900, chapters=[]
        )
        await agent.seed_existing_chapter(
            book_id, "sess-seed", "Prologue", "The night was still.", filename=slug
        )
        agent.current_books[book_id].chapters.append(
            {"id": "c2", "title": "Chapter One", "outline": "x", "status": "planned"}
        )

        output_path = resolve_within_project(f"{workspace_subpath(slug)}/{slug}.md")
        assert "The night was still." in output_path.read_text(encoding="utf-8")
        assert agent.current_books[book_id].chapters[0]["status"] == "written"

        async for _ in agent._handle_write_full_book(book_id, "sess-seed", filename=slug):
            pass

        # Only the real (non-seeded) chapter should have triggered generation
        # — one prose call per outline beat (the mocked outline has 3).
        writing_calls = [
            c for c in agent.llm_interface.calls if "THIS PART must cover ONLY this beat" in c
        ]
        assert len(writing_calls) == 3
        # And the seeded prologue text must show up as continuity context.
        planning_calls = [
            c for c in agent.llm_interface.calls if "Write a detailed scene-by-scene outline" in c
        ]
        assert "The night was still." in planning_calls[0]
    finally:
        _cleanup(slug)


# ------------------------------------------------------------- world bible creation


@pytest.mark.asyncio
async def test_handle_create_world_bible_saves_file_and_attaches_to_session(agent):
    slug = "scratch_bible_gen"
    try:
        streams = [
            s
            async for s in agent._handle_create_world_bible(
                {"filename": slug, "topic": "rick notes"},
                "Rick is a Texas man who develops powers tied to belief and pressure.",
                "sess-bible-gen",
            )
        ]
        bible_path = resolve_within_project(f"{workspace_subpath(slug)}/world_bible.md")
        assert bible_path.exists()
        assert agent.writing_sessions["sess-bible-gen"]["world_bible"]
        assert any(s.type == "result" and "saved to" in s.content for s in streams)
    finally:
        shutil.rmtree(workspace_dir() / slug, ignore_errors=True)


@pytest.mark.asyncio
async def test_world_bible_request_routes_before_normal_classification(agent):
    """A "turn my notes into a world bible" message must not be classified
    as a normal writing task — it's a standalone reference document. With
    no active book and no explicit filename, it saves under the plain
    "world_bible" default rather than a slug derived from the raw notes."""
    try:
        async for _ in agent.run(
            "Please organize these notes into a story bible: Rick has powers tied to belief.",
            session_id="sess-bible-route",
        ):
            pass
        assert "world_bible" in agent.writing_sessions["sess-bible-route"]
        assert not any("Analyze this book writing request" in c for c in agent.llm_interface.calls)
    finally:
        _cleanup("world_bible")


def test_wants_world_bible_creation_matches_expected_phrasing():
    from agents.book_writing_helpers import wants_world_bible_creation

    assert wants_world_bible_creation("turn these notes into a world bible") is True
    assert wants_world_bible_creation("organize these ramblings into a story bible") is True
    assert wants_world_bible_creation("write a story about a dragon") is False
