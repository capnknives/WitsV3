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
        resolved = resolve_within_project(f"workspace/{slug}/{slug}.md")
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
            "in what order, key beats, and how it connects to what came before. 4-8 bullet "
            "points. Do not write prose — outline only."
        )
        return (
            await self.generate_response(
                prompt,
                model_name=book_settings.model,
                temperature=max(0.3, book_settings.temperature - 0.2),
                max_tokens=min(600, book_settings.max_tokens),
            )
        ).strip()

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
    ) -> str:
        """Generate the actual prose for one chapter from its detailed
        outline, with the story bible and prior text as grounding context."""
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

        prompt = (f"{context}\n\n" if context else "") + (
            f'Write Chapter {idx} of {total_chapters} of the {book.genre} book "{book.title}".\n\n'
            f"Chapter title: {chapter['title']}\n"
            f"Chapter outline (follow this):\n{detailed_outline}\n"
            f"{pov_line}\n"
            f"Write approximately {target_words} words of narrative prose for this chapter "
            "only, matching the voice and tense already established in the story so far. "
            "Output ONLY the prose — no chapter headers, no meta-commentary, no restatement "
            "of the outline."
        )
        return (
            await self.generate_response(
                prompt,
                model_name=book_settings.model,
                temperature=book_settings.temperature,
                max_tokens=book_settings.max_tokens,
            )
        ).strip()

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
        # Novella-density floor (was 400) — a "30 page" request should read
        # like chapters of a real novella, not a series of short vignettes.
        per_chapter_target = max(900, book.target_length // max(len(chapters), 1))

        total_words = 0
        chapters_written = 0
        for idx, chapter in enumerate(chapters, start=1):
            if chapter.get("status") == "written":
                continue

            continuity_tail = self._read_continuity_tail(output_path)

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
