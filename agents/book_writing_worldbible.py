# agents/book_writing_worldbible.py
"""Turns free-form, rambled notes into a structured story bible.

Bonus feature requested alongside the chapter-by-chapter rewrite: rather
than requiring the user to hand-author a document like
Rick_Series_Unified_Story_Bible.pdf, let them dictate loose notes and have
the agent organize them into the same kind of structured reference — then
attach it to the active book so every outline/chapter prompt stays grounded
in it instead of drifting.
"""

from collections.abc import AsyncGenerator
from typing import Any

from core.safe_code_editor import PROJECT_ROOT
from core.schemas import StreamData


class BookWritingWorldBibleMixin:
    """Ramble-to-world-bible generation, reusing the writer mixin's file
    helpers (_slugify / _book_output_path) via the shared BookWritingAgent."""

    async def _handle_create_world_bible(
        self,
        task_analysis: dict[str, Any],
        raw_notes: str,
        session_id: str,
    ) -> AsyncGenerator[StreamData, None]:
        yield self.stream_action("Organizing your notes into a story bible...")

        prompt = (
            "The user dictated freeform notes about a story/book idea. Organize them "
            "into a structured story bible using ONLY facts present in the notes — do "
            "not invent new plot points, characters, or rules that aren't implied by "
            "the notes. If a section has no relevant notes, omit that section rather "
            "than making something up.\n\n"
            "Use this structure (markdown headers), skipping any section with nothing "
            "to put in it:\n\n"
            "# <Working Title>\n\n"
            "## Core Premise\n## Protagonist\n## Supporting Characters\n## Themes\n"
            "## Rules / Power System (if applicable)\n## Narrative Structure & POV\n"
            "## Tone & Voice\n## Chapter / Book Goals\n\n"
            f"RAW NOTES:\n{raw_notes}\n"
        )
        yield self.stream_thinking("Structuring the story bible...")
        world_bible_text = (
            await self.generate_response(prompt, temperature=0.4, max_tokens=2000)
        ).strip()

        session_state = self.writing_sessions.setdefault(session_id, {})

        # Prefer saving alongside the book already in progress for this
        # session (workspace/<book>/world_bible.md next to <book>.md) over
        # an explicit filename, and only fall back to a plain default —
        # never the raw notes themselves, which produced unreadable
        # workspace/ directory names when no title was given.
        active_book_id = session_state.get("active_book_id")
        if active_book_id and active_book_id in self.current_books:
            slug_source = self.current_books[active_book_id].title
        else:
            slug_source = task_analysis.get("filename") or "world_bible"
        slug = self._slugify(slug_source)
        output_path = self._book_output_path(slug).parent / "world_bible.md"
        output_path.write_text(world_bible_text, encoding="utf-8")
        rel_path = output_path.relative_to(PROJECT_ROOT)

        session_state["world_bible"] = world_bible_text
        session_state["world_bible_path"] = str(rel_path)

        # If a book is already in progress for this session, attach the
        # bible immediately so the next chapter written uses it.
        if active_book_id and active_book_id in self.current_books:
            self.current_books[active_book_id].world_bible = world_bible_text

        await self.store_memory(
            content=f"Created story bible ({rel_path}): {world_bible_text[:500]}",
            segment_type="WORLD_BIBLE",
            importance=0.9,
            metadata={"session_id": session_id, "file_path": str(rel_path)},
        )

        yield self.stream_result(f"Story bible created and saved to {rel_path}")
        yield self.stream_result(world_bible_text)
