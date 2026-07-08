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
"""

import re
from collections.abc import AsyncGenerator
from pathlib import Path

from core.safe_code_editor import PROJECT_ROOT, resolve_within_project
from core.schemas import StreamData


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

    async def _handle_write_full_book(
        self,
        book_id: str,
        session_id: str,
        filename: str | None = None,
    ) -> AsyncGenerator[StreamData, None]:
        """Write every not-yet-written chapter of `book_id` to disk.

        Safe to call more than once for the same book: chapters already
        marked "written" are skipped, so re-invoking after an interruption
        (or a follow-up "keep going") resumes instead of restarting.
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
        per_chapter_target = max(400, book.target_length // max(len(chapters), 1))

        total_words = 0
        chapters_written = 0
        for idx, chapter in enumerate(chapters, start=1):
            if chapter.get("status") == "written":
                continue

            yield self.stream_action(
                f"Writing chapter {idx}/{len(chapters)}: {chapter['title']}..."
            )

            outline = (chapter.get("outline") or "").strip() or "Continue the story naturally."
            prompt = (
                f'Continue writing the {book.genre} story "{book.title}".\n\n'
                f"Chapter {idx}: {chapter['title']}\n"
                f"Chapter outline: {outline}\n\n"
                f"Write approximately {per_chapter_target} words of narrative prose for this "
                "chapter only. Output ONLY the prose — no chapter headers, no meta-commentary, "
                "no restatement of the outline."
            )

            try:
                content = (
                    await self.generate_response(
                        prompt,
                        model_name=book_settings.model,
                        temperature=book_settings.temperature,
                        max_tokens=book_settings.max_tokens,
                    )
                ).strip()
            except Exception as e:
                yield self.stream_error(f"Failed to generate chapter {idx}: {e}")
                break

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
