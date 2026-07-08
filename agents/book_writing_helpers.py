# agents/book_writing_helpers.py
"""Outline parsing and narrative analysis helpers for BookWritingAgent."""

import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from core.schemas import StreamData


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
