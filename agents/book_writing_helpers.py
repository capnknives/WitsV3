# agents/book_writing_helpers.py
"""Outline parsing and narrative analysis helpers for BookWritingAgent."""

import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from core.schemas import StreamData

_FILENAME_RE = re.compile(
    r"(?:save\s+(?:it|this|the\s+story|the\s+book)?\s*as|"
    r"call\s+it|name\s+it|named)\s+"
    r"([A-Za-z0-9_][A-Za-z0-9 _-]{1,60}?)"
    r"(?:\.(?:txt|md|docx?)\b)?"
    r"(?=[.,!?;:]|\s+(?:and|with|please)\b|$)",
    re.IGNORECASE,
)

# Phrases that mean "write the complete thing now and save it" rather than
# "just sketch an outline" — used so a single request that already contains
# enough detail (topic + length + save intent) doesn't stall after producing
# only a structure/outline the way the 2026-07-08 live-chat transcript did.
_FULL_BOOK_NOW_RE = re.compile(
    r"\b(save (it|this|the story|the book)|save (it |this )?to disk|"
    r"write (it all|the whole|the entire|the full)|write it\b|make it\b)",
    re.IGNORECASE,
)


def extract_requested_filename(text: str) -> str | None:
    """Pull a user-requested save name out of free text, e.g. "save the
    story as TheBigStory01" -> "TheBigStory01". Returns None if no explicit
    filename was requested."""
    match = _FILENAME_RE.search(text)
    if not match:
        return None
    name = match.group(1).strip()
    return name or None


def wants_full_book_now(text: str) -> bool:
    """True when the request asks to write everything and save it now,
    rather than just producing an outline/structure."""
    return bool(_FULL_BOOK_NOW_RE.search(text))


# A short "keep going" reply carries no new content of its own — treated as
# "continue writing the book already in progress for this session" rather
# than re-analyzed as a fresh, unrelated request (which is how a follow-up
# like "Okay, so make it." ended up producing a completely different,
# shorter outline in the 2026-07-08 live-chat transcript).
_CONTINUATION_RE = re.compile(
    r"^\s*(okay,?\s*|ok,?\s*|alright,?\s*|so\s+)*"
    r"(please\s+)?"
    r"(make it|do it|write it( all)?|go ahead|finish it|finish (it|the (story|book|chapter))|"
    r"continue( writing)?|keep (going|writing)|"
    r"write the (whole|entire|full) (story|book|thing)|"
    r"save (it|this)( now)?( to disk)?)\s*\.?\s*$",
    re.IGNORECASE,
)


def is_continuation_phrase(text: str) -> bool:
    """True for a short "keep going" reply with no new content of its own."""
    return bool(_CONTINUATION_RE.match(text.strip()))


class BookWritingHelpersMixin:
    """Extracted helpers to keep handler mixin under the 500-line limit."""

    async def _extract_chapters_from_response(
        self,
        response: str,
        book_id: str,
    ) -> list[dict[str, Any]]:
        """Extract chapter information from LLM response."""
        chapters = []
        lines = response.split("\n")

        current_chapter = None
        for line in lines:
            line = line.strip()
            if line.startswith("Chapter") or line.startswith("ch.") or line.startswith("#"):
                if current_chapter:
                    chapters.append(current_chapter)

                current_chapter = {
                    "id": str(uuid.uuid4()),
                    "title": line,
                    "outline": "",
                    "word_target": 2000,
                    "status": "planned",
                }
            elif current_chapter and line:
                current_chapter["outline"] += line + " "

        if current_chapter:
            chapters.append(current_chapter)

        if not chapters:
            for i in range(5):
                chapters.append(
                    {
                        "id": str(uuid.uuid4()),
                        "title": f"Chapter {i+1}",
                        "outline": f"Content for chapter {i+1}",
                        "word_target": 2000,
                        "status": "planned",
                    }
                )

        return chapters

    async def _generate_chapter_outline(
        self,
        book_id: str,
        chapter_id: str,
    ) -> AsyncGenerator[StreamData, None]:
        """Generate detailed outline for a specific chapter."""
        if book_id not in self.current_books:
            yield self.stream_error("Book not found")
            return

        book = self.current_books[book_id]
        chapter_info = None

        for chapter in book.chapters:
            if chapter["id"] == chapter_id:
                chapter_info = chapter
                break

        if not chapter_info:
            yield self.stream_error("Chapter not found")
            return

        outline_prompt = f"""
        Create a detailed outline for this chapter in a {book.genre} book:

        Book Title: {book.title}
        Chapter Title: {chapter_info['title']}
        Chapter Overview: {chapter_info['outline']}
        Target Length: {chapter_info.get('word_target', 2000)} words

        Provide:
        1. Section breakdown with subheadings
        2. Key points to cover in each section
        3. Examples, anecdotes, or details to include
        4. Transitions between sections
        5. Opening hook and closing summary
        """

        yield self.stream_thinking(f"Creating detailed outline for {chapter_info['title']}...")

        detailed_outline = await self.generate_response(outline_prompt, temperature=0.6)

        chapter_info["detailed_outline"] = detailed_outline
        chapter_info["status"] = "outlined"

        yield self.stream_result(f"Detailed outline for {chapter_info['title']}:")
        yield self.stream_result(detailed_outline)

        await self.store_memory(
            content=f"Chapter outline: {detailed_outline}",
            segment_type="CHAPTER_OUTLINE",
            importance=0.8,
            metadata={
                "book_id": book_id,
                "chapter_id": chapter_id,
                "chapter_title": chapter_info["title"],
            },
        )

    async def _extract_research_concepts(self, research_text: str) -> list[str]:
        """Extract key concepts from research text."""
        concepts = []
        concept_patterns = [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            r"\d+%|\$\d+|\d+\s+(?:million|billion|thousand)",
            r"(?:research|study|survey|report)\s+(?:shows|indicates|suggests|found)",
        ]

        for pattern in concept_patterns:
            matches = re.findall(pattern, research_text)
            concepts.extend(matches)

        unique_concepts = list(set(concepts))
        return [
            concept for concept in unique_concepts if len(concept.split()) <= 3 and len(concept) > 3
        ][:10]

    async def _parse_outline_structure(self, outline_text: str) -> dict[str, Any]:
        """Parse outline text into structured data."""
        structure: dict[str, Any] = {
            "sections": [],
            "estimated_length": 0,
            "key_topics": [],
        }

        lines = outline_text.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if (
                line.startswith(("#", "##", "1.", "2.", "I.", "A."))
                or line.isupper()
                or line.endswith(":")
            ):

                if current_section:
                    structure["sections"].append(current_section)

                current_section = {
                    "title": line.rstrip(":"),
                    "points": [],
                    "estimated_words": 500,
                }

            elif current_section and line.startswith(("-", "*", "•")):
                current_section["points"].append(line[1:].strip())

            elif current_section:
                current_section["points"].append(line)

        if current_section:
            structure["sections"].append(current_section)

        structure["estimated_length"] = len(structure["sections"]) * 500
        structure["key_topics"] = [section["title"] for section in structure["sections"]]

        return structure

    async def _analyze_hero_journey(self, content: str) -> dict[str, Any]:
        """Analyze content using hero's journey structure."""
        stages = [
            "ordinary_world",
            "call_to_adventure",
            "refusal",
            "mentor",
            "crossing_threshold",
            "tests",
            "ordeal",
            "reward",
            "return",
        ]

        analysis = {"structure": "hero_journey", "stages_present": []}

        if self.neural_web and self.enable_narrative_intelligence:
            for stage in stages:
                stage_concepts = await self.neural_web._find_relevant_concepts(f"{stage} {content}")
                if stage_concepts:
                    analysis["stages_present"].append(stage)

        return analysis

    async def _analyze_three_act_structure(self, content: str) -> dict[str, Any]:
        """Analyze content using three-act structure."""
        acts = ["setup", "confrontation", "resolution"]
        analysis = {"structure": "three_act", "acts_present": []}

        if self.neural_web and self.enable_narrative_intelligence:
            for act in acts:
                act_concepts = await self.neural_web._find_relevant_concepts(f"{act} {content}")
                if act_concepts:
                    analysis["acts_present"].append(act)

        return analysis

    async def _analyze_spiral_narrative(self, content: str) -> dict[str, Any]:
        """Analyze content using spiral narrative structure."""
        return {
            "structure": "spiral",
            "themes_present": [],
            "recurring_motifs": [],
        }
