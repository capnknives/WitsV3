# agents/book_writing_writer.py
"""Full-book, chapter-by-chapter writing pipeline with real disk persistence.

2026-07-08 finding: the book writing agent never wrote anything to disk —
every handler only called store_memory() and streamed prose back into chat.
A user who asked for a "100 page story ... save it as X" got a chat reply
that *claimed* the file existed and nothing on disk. On top of that, a
single one-shot LLM call is bounded by the model's context window (roughly
1,000-1,500 words in practice), nowhere near "100 pages" no matter how the
prompt asked for length.

This mixin writes one chapter per LLM call (small enough to actually
complete) and appends each finished chapter to a real file on disk
immediately, so a story that takes many calls to finish survives a crash,
an interrupted session, or a slow model instead of vanishing as unsent
chat text.

2026-07-08, second pass: added a two-stage per-chapter pipeline (expand the
book-level one-liner into a detailed scene outline, *then* write prose from
that outline) plus cross-chapter continuity (the tail of whatever is already
on disk is fed into both the outline and the prose prompt) and a way to seed
a book with content the user already wrote (e.g. an existing prologue) so
new chapters continue it instead of ignoring it. This is what makes chapters
read as installments of one novella instead of independent one-shot scenes.
"""

import difflib
import re
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from core.safe_code_editor import PROJECT_ROOT, resolve_within_project
from core.schemas import StreamData

# How much of the story-so-far (tail of the file already on disk) gets fed
# into outline expansion and prose generation as continuity context. Roughly
# 400-500 words — enough to anchor voice/plot state without eating most of
# the prompt budget alongside the world bible and chapter outline.
_CONTINUITY_CHARS = 3000


class BookWritingWriterMixin:
    """Chapter-by-chapter book writing with incremental, resumable saves."""

    @staticmethod
    def _slugify(title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        return slug or "untitled_story"

    def _book_output_path(self, slug: str) -> Path:
        from core.runtime_paths import workspace_subpath

        resolved = resolve_within_project(f"{workspace_subpath(slug)}/{slug}.md")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    @staticmethod
    def _strip_leading_title_echo(content: str, title: str) -> str:
        """Drop a first line that just repeats the chapter title.
        2026-07-08 live-model finding: despite "no chapter headers" in the
        prompt, a chapter's prose sometimes opened by restating its own
        title as a bare first line before the real prose began."""
        lines = content.split("\n", 1)
        if not lines:
            return content
        first_line = lines[0].strip().strip("#").strip()
        normalize = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())  # noqa: E731
        if first_line and normalize(first_line) == normalize(title):
            return lines[1].lstrip("\n") if len(lines) > 1 else ""
        return content

    @staticmethod
    def _read_continuity_tail(output_path: Path) -> str:
        """The end of whatever's already on disk, used as "story so far"
        context. Reading from disk (rather than tracking generated content
        in memory) means continuity works the same way whether the previous
        chapter was just generated, seeded from user-supplied text, or
        written in an earlier, now-restarted process."""
        if not output_path.exists():
            return ""
        text = output_path.read_text(encoding="utf-8")
        if "##" not in text:
            # Only the "# Title" header has been written so far — no
            # chapter content exists yet, so there's nothing to continue.
            return ""
        return text[-_CONTINUITY_CHARS:].strip()

    async def seed_existing_chapter(
        self,
        book_id: str,
        session_id: str,
        title: str,
        content: str,
        filename: str | None = None,
        position: int = 0,
    ) -> Path:
        """Insert content the user already wrote (e.g. an existing prologue)
        as an already-complete chapter, and persist it to disk immediately
        so later generated chapters see it as continuity context instead of
        ignoring it or having it regenerated from scratch."""
        book = self.current_books[book_id]
        chapter = {
            "id": str(uuid.uuid4()),
            "title": title,
            "outline": "",
            "detailed_outline": "",
            "word_target": len(content.split()),
            "status": "written",
            "seeded": True,
        }
        book.chapters.insert(position, chapter)

        slug = self._slugify(filename or book.title)
        output_path = self._book_output_path(slug)
        if not output_path.exists():
            output_path.write_text(f"# {book.title}\n\n", encoding="utf-8")
        with output_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {title}\n\n{content.strip()}\n")

        self.writing_sessions.setdefault(session_id, {})["active_book_id"] = book_id
        return output_path

    @staticmethod
    def _detect_pov(continuity_tail: str, world_bible: str) -> str | None:
        """Best-effort detection of first- vs third-person narration, so new
        chapters can be told to match it explicitly. 2026-07-08 live-model
        finding: "matching the voice already established" alone wasn't a
        strong enough instruction — a first-person prologue and Chapter 1
        were followed by Chapter 2 quietly drifting into third person.
        Prefers evidence from actual prior prose (pronoun counting) over the
        world bible's stated intent, since that's what the model is really
        continuing from; falls back to the bible only when nothing's been
        written yet."""
        text = continuity_tail
        if not text:
            lowered_bible = (world_bible or "").lower()
            if "first person" in lowered_bible:
                return 'first person ("I"/"me"/"my")'
            if "third person" in lowered_bible:
                return "third person"
            return None

        lowered = f" {text.lower()} "
        first_person = sum(
            lowered.count(f" {w} ") for w in ("i", "i'm", "i've", "i'd", "i'll", "my", "me")
        )
        third_person = sum(
            lowered.count(f" {w} ") for w in ("he", "she", "him", "her", "his", "hers")
        )
        if first_person == 0 and third_person == 0:
            return None
        return 'first person ("I"/"me"/"my")' if first_person >= third_person else "third person"

    async def _expand_chapter_outline(
        self,
        book,
        chapter: dict,
        idx: int,
        world_bible: str,
        continuity_tail: str,
        book_settings,
    ) -> str:
        """Turn a chapter's one-line summary into a detailed scene-by-scene
        outline, grounded in the world bible and what's already been
        written, before any prose is generated for it."""
        context_blocks = []
        if world_bible:
            context_blocks.append(
                f"STORY BIBLE (characters, tone, established rules):\n{world_bible}"
            )
        if continuity_tail:
            context_blocks.append(
                f"STORY SO FAR (end of the most recent chapter):\n{continuity_tail}"
            )
        context = "\n\n".join(context_blocks)
        pov = self._detect_pov(continuity_tail, world_bible)
        pov_line = (
            f"Narrative point of view: {pov}. Keep this outline's beats consistent with it.\n"
            if pov
            else ""
        )

        prompt = (f"{context}\n\n" if context else "") + (
            f'Plan Chapter {idx} of the {book.genre} book "{book.title}".\n\n'
            f"Chapter title: {chapter['title']}\n"
            f"Brief premise for this chapter: {(chapter.get('outline') or '').strip() or 'Continue the story naturally.'}\n"
            f"{pov_line}\n"
            "Write a detailed scene-by-scene outline for THIS chapter only: what happens, "
            "in what order, key beats, and how it connects to what came before. EXACTLY "
            "4-6 beats, each ONE bullet on ONE line — no sub-bullets, no multi-line "
            "elaboration per beat. Do not write prose — outline only."
        )
        return (
            await self.generate_response(
                prompt,
                model_name=book_settings.model,
                temperature=max(0.3, book_settings.temperature - 0.2),
                max_tokens=min(800, book_settings.max_tokens),
                num_ctx=book_settings.num_ctx,
            )
        ).strip()

    # Hard cap on beats per chapter regardless of what the outline call
    # returns — 2026-07-08 finding: an outline whose bullets wrapped or
    # elaborated across multiple lines got parsed as one beat per *line*,
    # producing 30+ "beats" for a single chapter and an 8,000+ word chapter
    # against a 2,200-word target.
    _MAX_BEATS_PER_CHAPTER = 8
    _BEAT_MARKER_RE = re.compile(r"^([-*•]|\d+[.)])\s+")

    @classmethod
    def _parse_outline_beats(cls, outline: str) -> list[str]:
        """Split a detailed outline into its individual bullet-point beats.

        Only a line starting with an actual bullet/numbered marker begins a
        new beat; any other non-empty line is folded into the current beat
        as continuation text, so a bullet that wraps or gets elaborated
        across multiple lines doesn't get miscounted as multiple beats.
        """
        beats: list[str] = []
        for raw_line in outline.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            match = cls._BEAT_MARKER_RE.match(line)
            if match:
                beats.append(line[match.end() :].strip())
            elif beats:
                beats[-1] = f"{beats[-1]} {line}".strip()
            # else: prose before the first bullet marker — ignore
        beats = beats or [outline.strip()]
        return beats[: cls._MAX_BEATS_PER_CHAPTER]

    async def _write_chapter_prose(
        self,
        book,
        chapter: dict,
        idx: int,
        total_chapters: int,
        world_bible: str,
        continuity_tail: str,
        detailed_outline: str,
        target_words: int,
        book_settings,
        prior_book_text: str = "",
    ) -> str:
        """Generate the actual prose for one chapter, one outline beat per
        LLM call, with the story bible and prior text as grounding context.

        2026-07-08 finding: a single "write ~2200 words" call routinely
        stopped far short (e.g. 619/1071 words). The obvious fix — chain
        "continue, write more words" calls until long enough — made things
        *worse*: with nothing new to say, the model either restated the
        entire plot arc a second time after it had already reached a
        natural conclusion, or degenerated into repetitive boilerplate
        filler. Generating one call per outline beat instead gives every
        call distinct content to produce, which is what actually produces
        a longer chapter without padding or repetition.
        """
        context_blocks = []
        if world_bible:
            context_blocks.append(
                f"STORY BIBLE (characters, tone, established rules):\n{world_bible}"
            )
        if continuity_tail:
            context_blocks.append(
                f"STORY SO FAR (end of the most recent chapter — continue directly from this, "
                f"do not repeat it):\n{continuity_tail}"
            )
        context = "\n\n".join(context_blocks)
        pov = self._detect_pov(continuity_tail, world_bible)
        pov_line = (
            f"CRITICAL: Narrate in {pov}, exactly matching the story so far. Do not switch "
            "narrative person partway through.\n"
            if pov
            else ""
        )

        beats = self._parse_outline_beats(detailed_outline)
        words_per_beat = max(250, target_words // len(beats))
        # Token budget per beat call, scaled to that beat's own word target
        # rather than the whole chapter's ceiling. 2026-07-08 live-model
        # finding: every beat call was passed the full chapter max_tokens
        # (e.g. 4500), so nothing stopped a single beat from ballooning to
        # several times its ~350-word target (observed: 600-700+ actual
        # words/beat against a 366-word target, an entire chapter landing
        # at 4124 words against a 2200-word target).
        #
        # 2026-07-08, second finding: a first attempt at 1.6 tokens/word
        # was too tight -- the model's natural per-beat length runs
        # 600-700+ words regardless of the stated target, so most beats hit
        # the cap mid-sentence ("...he forced himself to", a chapter ending
        # on the bare word "The"), and the next beat would then clumsily
        # restate the severed opening instead of continuing past it. 2.3
        # tokens/word gives enough headroom for the model's natural length
        # to land inside the cap in the common case; _trim_incomplete_sentence
        # below is the fallback for whatever still gets cut.
        beat_max_tokens = min(book_settings.max_tokens, max(400, int(words_per_beat * 2.3)))

        chapter_so_far = ""
        for beat_idx, beat in enumerate(beats, start=1):
            is_last = beat_idx == len(beats)
            recent_tail = (chapter_so_far or continuity_tail)[-2000:]
            prompt = (f"{context}\n\n" if context else "") + (
                f"Write part {beat_idx} of {len(beats)} of Chapter {idx} of {total_chapters} of "
                f'the {book.genre} book "{book.title}".\n\n'
                f"Chapter title: {chapter['title']}\n"
                f"Full chapter outline (for context, already covered beats are done — do not "
                f"repeat them):\n{detailed_outline}\n\n"
                f"THIS PART must cover ONLY this beat: {beat}\n"
                + (
                    f"\nWhat's already been written in this chapter so far (continue directly "
                    f"from this, do not repeat or restart it):\n{recent_tail}\n"
                    if recent_tail
                    else ""
                )
                + f"\n{pov_line}"
                f"Write about {words_per_beat} words of narrative prose covering only this "
                "beat, flowing naturally from what came before. "
                + (
                    "This is the final beat — bring the chapter to a close."
                    if is_last
                    else "Do NOT conclude the chapter or wrap up the story yet — more beats "
                    "follow after this one."
                )
                + " Output ONLY the prose — no chapter headers, no meta-commentary, no "
                "restatement of the outline."
            )
            segment = (
                await self.generate_response(
                    prompt,
                    model_name=book_settings.model,
                    temperature=book_settings.temperature,
                    max_tokens=beat_max_tokens,
                    num_ctx=book_settings.num_ctx,
                )
            ).strip()
            segment = self._trim_incomplete_sentence(segment)
            if segment:
                chapter_so_far = f"{chapter_so_far.rstrip()}\n\n{segment}".strip()

        return self._dedupe_repeated_paragraphs(
            self._strip_meta_commentary(chapter_so_far), reference=prior_book_text
        )

    # Matches a paragraph where the model breaks character to comment on its
    # own generation process instead of writing story prose -- 2026-07-08
    # live-model finding: one beat's output opened with "(The critical
    # narrative prose section comes to an end here -- the chapter outline
    # contains 4 beats after this one that must be covered in subsequent
    # outputs." despite the prompt's explicit "no meta-commentary"
    # instruction. Real prose doesn't open a paragraph by naming "beats",
    # "outline", or "output" inside parentheses, so this is a safe tell.
    _META_COMMENTARY_RE = re.compile(
        r"^\(.*\b(beat|outline|narrative prose section|subsequent output)s?\b", re.IGNORECASE
    )

    @classmethod
    def _strip_meta_commentary(cls, content: str) -> str:
        """Drop paragraphs where the model narrates its own generation
        process rather than the story."""
        paragraphs = [p for p in content.split("\n\n") if p.strip()]
        kept = [p for p in paragraphs if not cls._META_COMMENTARY_RE.match(p.strip())]
        return "\n\n".join(kept)

    @staticmethod
    def _trim_incomplete_sentence(text: str) -> str:
        """Cut a trailing sentence fragment left when a beat call hits its
        token cap mid-sentence, instead of leaving prose that stops
        mid-word/mid-clause (e.g. "...he forced himself to", a chapter
        ending on the bare word "The"). 2026-07-08 live-model finding: a
        dangling fragment like this also confused the *next* beat, which
        would clumsily restate the severed opening rather than continue
        past it. Falls back to the untouched text if no complete sentence
        is found at all, since a half-finished beat still beats an empty
        one."""
        stripped = text.rstrip()
        if not stripped or stripped[-1] in ".!?”’\"'":
            return stripped
        last_end = max(stripped.rfind(ch) for ch in ".!?")
        if last_end == -1:
            return stripped
        end = last_end + 1
        if end < len(stripped) and stripped[end] in "”’\"'":
            end += 1
        return stripped[:end]

    @staticmethod
    def _dedupe_repeated_paragraphs(
        content: str, similarity_threshold: float = 0.8, reference: str = ""
    ) -> str:
        """Drop paragraphs that near-duplicate an earlier paragraph, either
        within the same chapter or (via `reference`) already written
        earlier in the book. 2026-07-08 finding: beat-by-beat generation
        occasionally produced a near-verbatim repeat of an earlier beat's
        resolution within one chapter (e.g. two beats both landing on the
        same "agrees to mentor him" exchange, word-for-word across six
        consecutive paragraphs).

        2026-07-08, second finding: the same thing happened *across*
        chapters — the model's "bring the chapter to a close" instruction
        produces a formulaic inspirational-monologue ending, and because
        the next chapter's prompt includes the previous chapter's tail as
        "story so far" context, the model sometimes reproduced that same
        ten-paragraph closing block almost word-for-word one chapter
        later. Passing the book's prior text as `reference` catches that
        case too, not just repeats within the current chapter."""
        reference_normalized = [
            re.sub(r"\s+", " ", p.strip().lower()) for p in reference.split("\n\n") if p.strip()
        ]
        paragraphs = [p for p in content.split("\n\n") if p.strip()]
        kept: list[str] = []
        normalized_kept: list[str] = list(reference_normalized)
        for para in paragraphs:
            normalized = re.sub(r"\s+", " ", para.strip().lower())
            if any(
                difflib.SequenceMatcher(None, normalized, prior).ratio() >= similarity_threshold
                for prior in normalized_kept
            ):
                continue
            kept.append(para)
            normalized_kept.append(normalized)
        return "\n\n".join(kept)

    async def _handle_write_full_book(
        self,
        book_id: str,
        session_id: str,
        filename: str | None = None,
    ) -> AsyncGenerator[StreamData, None]:
        """Write every not-yet-written chapter of `book_id` to disk.

        Safe to call more than once for the same book: chapters already
        marked "written" are skipped, so re-invoking after an interruption
        (or a follow-up "keep going") resumes instead of restarting. Each
        chapter is planned (detailed outline) and then written using the
        story bible and the tail of whatever's already on disk, so chapters
        build on each other instead of reading as disconnected scenes.
        """
        book = self.current_books.get(book_id)
        if not book:
            yield self.stream_error(
                "No book project found to write — start with a new story request."
            )
            return

        chapters = book.chapters
        if not chapters:
            yield self.stream_error("This book has no chapters to write yet.")
            return

        slug = self._slugify(filename or book.title)
        output_path = self._book_output_path(slug)
        if not output_path.exists():
            output_path.write_text(f"# {book.title}\n\n", encoding="utf-8")

        book_settings = self.config.agents.book_writing_agent
        world_bible = (book.world_bible or "").strip()
        # Novella-density floor. Standard novellas run 17,500-40,000 words
        # (~70-160 pages at 250 words/page); at typical chapter counts that's
        # roughly 2,000-3,500 words per chapter, not the 900-word floor this
        # used to have (2026-07-08 finding: chapters were consistently
        # coming out too short even when target_length/chapter count implied
        # more, and max_tokens/num_ctx weren't sized to support longer output
        # anyway).
        per_chapter_target = max(2200, book.target_length // max(len(chapters), 1))

        total_words = 0
        chapters_written = 0
        for idx, chapter in enumerate(chapters, start=1):
            if chapter.get("status") == "written":
                continue

            continuity_tail = self._read_continuity_tail(output_path)
            prior_book_text = (
                output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            )

            yield self.stream_action(
                f"Planning chapter {idx}/{len(chapters)}: {chapter['title']}..."
            )
            try:
                detailed_outline = await self._expand_chapter_outline(
                    book, chapter, idx, world_bible, continuity_tail, book_settings
                )
            except Exception as e:
                yield self.stream_error(f"Failed to plan chapter {idx}: {e}")
                break
            chapter["detailed_outline"] = detailed_outline

            yield self.stream_action(
                f"Writing chapter {idx}/{len(chapters)}: {chapter['title']}..."
            )
            try:
                content = await self._write_chapter_prose(
                    book,
                    chapter,
                    idx,
                    len(chapters),
                    world_bible,
                    continuity_tail,
                    detailed_outline,
                    per_chapter_target,
                    book_settings,
                    prior_book_text=prior_book_text,
                )
            except Exception as e:
                yield self.stream_error(f"Failed to generate chapter {idx}: {e}")
                break
            content = self._strip_leading_title_echo(content, chapter["title"]).strip()

            with output_path.open("a", encoding="utf-8") as f:
                f.write(f"\n## Chapter {idx}: {chapter['title']}\n\n{content}\n")

            chapter["status"] = "written"
            chapter_words = len(content.split())
            total_words += chapter_words
            chapters_written += 1

            rel_path = output_path.relative_to(PROJECT_ROOT)
            yield self.stream_observation(
                f"Chapter {idx} saved ({chapter_words} words, running total {total_words}) "
                f"-> {rel_path}"
            )

        rel_path = output_path.relative_to(PROJECT_ROOT)

        if chapters_written == 0:
            yield self.stream_result(f"'{book.title}' was already fully written at {rel_path}.")
            return

        await self.store_memory(
            content=(
                f"Wrote full book '{book.title}' ({total_words} words, "
                f"{chapters_written} chapters) to {rel_path}"
            ),
            segment_type="BOOK_PROJECT",
            importance=0.9,
            metadata={
                "book_id": book_id,
                "session_id": session_id,
                "file_path": str(rel_path),
                "word_count": total_words,
            },
        )

        self.writing_sessions.setdefault(session_id, {})["active_book_id"] = book_id

        remaining = sum(1 for c in chapters if c.get("status") != "written")
        if remaining:
            yield self.stream_result(
                f"'{book.title}' — {chapters_written} more chapter(s) written "
                f"({total_words} words this pass), saved to {rel_path}. "
                f'{remaining} chapter(s) still to go — say "continue" to keep going.'
            )
        else:
            yield self.stream_result(
                f"'{book.title}' complete — {total_words} words across "
                f"{chapters_written} chapter(s), saved to {rel_path}"
            )
