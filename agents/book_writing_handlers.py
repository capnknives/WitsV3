# agents/book_writing_handlers.py
"""Task handlers and helpers for BookWritingAgent."""

import uuid
from typing import Any, AsyncGenerator, Dict, List

from core.schemas import StreamData

from agents.book_writing_helpers import BookWritingHelpersMixin
from agents.book_writing_models import BookStructure, Chapter


class BookWritingHandlersMixin(BookWritingHelpersMixin):
    """Book creation, chapter writing, research, and narrative analysis handlers."""

    async def _handle_book_creation(
        self,
        task_analysis: Dict[str, Any],
        session_id: str,
    ) -> AsyncGenerator[StreamData, None]:
        """Handle creation of a new book project."""

        yield self.stream_action("Creating new book project...")

        # Generate book structure
        book_id = str(uuid.uuid4())

        structure_prompt = f"""
        Create a detailed structure for a {task_analysis['genre']} book about {task_analysis['topic']}.

        Generate:
        1. A compelling title and subtitle
        2. Chapter breakdown with titles and brief descriptions
        3. Target audience
        4. Key themes and messages
        5. Research requirements
        6. Writing style guidelines

        Book should be approximately {task_analysis.get('length', 50000)} words.
        """

        yield self.stream_thinking("Generating book structure...")
        structure_response = await self.generate_response(structure_prompt, temperature=0.7)

        # Create book structure object
        book_structure = BookStructure(
            title=f"Book about {task_analysis['topic']}",  # Will be refined
            genre=task_analysis['genre'],
            target_length=task_analysis.get('length', 50000),
        )

        # Parse structure response to create chapters
        chapters = await self._extract_chapters_from_response(structure_response, book_id)
        book_structure.chapters = chapters

        self.current_books[book_id] = book_structure

        # Store in memory
        await self.store_memory(
            content=f"Created book project: {book_structure.title}",
            segment_type="BOOK_PROJECT",
            importance=0.9,
            metadata={"book_id": book_id, "session_id": session_id},
        )

        yield self.stream_result(f"Created book project '{book_structure.title}' with {len(chapters)} chapters")

        # Generate first chapter outline
        if chapters:
            yield self.stream_action("Generating outline for first chapter...")
            async for stream in self._generate_chapter_outline(book_id, chapters[0]['id']):
                yield stream

    async def _handle_chapter_writing(
        self,
        task_analysis: Dict[str, Any],
        session_id: str,
    ) -> AsyncGenerator[StreamData, None]:
        """Handle writing of a specific chapter."""

        yield self.stream_action("Writing chapter content...")

        # If no specific book context, create a standalone chapter
        chapter_id = str(uuid.uuid4())

        writing_prompt = f"""
        Write a compelling chapter about {task_analysis['topic']} in {task_analysis['style']} style.

        Requirements:
        - Genre: {task_analysis['genre']}
        - Style: {task_analysis['style']}
        - Length: approximately {task_analysis.get('length', 2000)} words
        - Maintain consistent tone and voice
        - Include vivid details and engaging narrative
        """

        yield self.stream_thinking("Crafting chapter content...")

        # Generate chapter content (non-streaming since streaming isn't supported)
        yield self.stream_thinking("Generating chapter content...")
        chapter_content = await self.generate_response(writing_prompt, temperature=0.8)

        # Simulate progress updates
        content_length = len(chapter_content)
        for i in range(0, content_length, 500):
            if i > 0:
                word_count = len(chapter_content[:i].split())
                yield self.stream_action(f"Written {word_count} words...")

        # Create chapter object
        chapter = Chapter(
            id=chapter_id,
            title=f"Chapter: {task_analysis['topic']}",
            outline="",
            content=chapter_content,
            word_count=len(chapter_content.split()),
            status="drafted",
        )

        # Store in memory
        await self.store_memory(
            content=f"Wrote chapter: {chapter.title} ({chapter.word_count} words)",
            segment_type="CHAPTER_CONTENT",
            importance=0.8,
            metadata={"chapter_id": chapter_id, "session_id": session_id},
        )

        yield self.stream_result(f"Completed chapter: {chapter.word_count} words written")

        # Offer to continue with related chapters
        yield self.stream_action("Would you like me to suggest related chapters or continue this story?")

    async def _handle_research(
        self,
        task_analysis: Dict[str, Any],
        session_id: str,
    ) -> AsyncGenerator[StreamData, None]:
        """Handle research for book topics."""

        yield self.stream_action("Conducting research...")

        topic = task_analysis['topic']

        # Use neural web to find related concepts
        if self.neural_web:
            yield self.stream_thinking("Searching knowledge network for related concepts...")

            # Add topic to neural web if not present
            await self.neural_web.add_concept(
                topic.replace(" ", "_"),
                f"Research topic: {topic}",
                "research_topic",
            )

            # Find related concepts through activation
            related_concepts = await self.neural_web.activate_concept(topic.replace(" ", "_"))

            if related_concepts:
                yield self.stream_observation(f"Found {len(related_concepts)} related concepts in knowledge network")

        # Use tools for web research if available
        research_results = []
        if self.tool_registry:
            try:
                # Try to use web search tool
                search_result = await self.tool_registry.execute_tool(
                    "web_search",
                    query=f"{topic} research facts statistics",
                )
                research_results.append(search_result)
                yield self.stream_observation("Gathered web research data")
            except Exception as e:
                self.logger.debug(f"Web search not available: {e}")

        # Generate research synthesis
        research_prompt = f"""
        Compile comprehensive research on: {topic}

        Provide:
        1. Key facts and statistics
        2. Historical context
        3. Current trends and developments
        4. Expert opinions and quotes
        5. Potential areas for deeper investigation
        6. Reliable sources to cite
        7. Controversial aspects or debates
        8. Practical applications or implications

        Format as structured research notes suitable for book writing.
        """

        yield self.stream_thinking("Synthesizing research findings...")
        research_synthesis = await self.generate_response(research_prompt, temperature=0.5)

        # Store research in memory with high importance
        await self.store_memory(
            content=f"Research on {topic}: {research_synthesis}",
            segment_type="RESEARCH_NOTES",
            importance=0.9,
            metadata={"topic": topic, "session_id": session_id},
        )

        # Add research concepts to neural web
        if self.neural_web:
            research_concepts = await self._extract_research_concepts(research_synthesis)
            for concept in research_concepts:
                await self.neural_web.add_concept(
                    concept.replace(" ", "_"),
                    f"Research finding: {concept}",
                    "research_finding",
                )
                # Connect to main topic
                await self.neural_web.connect_concepts(
                    topic.replace(" ", "_"),
                    concept.replace(" ", "_"),
                    "supports",
                    strength=0.8,
                )

        yield self.stream_result(f"Research completed on {topic}")
        yield self.stream_result(research_synthesis)

    async def _handle_revision(
        self,
        task_analysis: Dict[str, Any],
        session_id: str,
    ) -> AsyncGenerator[StreamData, None]:
        """Handle content revision and improvement."""

        yield self.stream_action("Analyzing content for revision...")

        # This would need the actual content to revise
        # For now, provide revision guidelines

        revision_prompt = f"""
        Provide detailed revision guidelines for {task_analysis['genre']} writing about {task_analysis['topic']}.

        Include:
        1. Structure and organization improvements
        2. Style and voice refinements
        3. Clarity and flow enhancements
        4. Fact-checking requirements
        5. Grammar and syntax review
        6. Audience engagement strategies
        7. Pacing and rhythm adjustments
        """

        yield self.stream_thinking("Generating revision guidelines...")
        revision_guidelines = await self.generate_response(revision_prompt, temperature=0.6)

        yield self.stream_result("Revision Guidelines:")
        yield self.stream_result(revision_guidelines)

        # Store revision guidelines
        await self.store_memory(
            content=f"Revision guidelines for {task_analysis['topic']}: {revision_guidelines}",
            segment_type="REVISION_GUIDELINES",
            importance=0.7,
            metadata={"topic": task_analysis['topic'], "session_id": session_id},
        )

    async def _handle_outline_generation(
        self,
        task_analysis: Dict[str, Any],
        session_id: str,
    ) -> AsyncGenerator[StreamData, None]:
        """Handle generation of book or chapter outlines."""

        yield self.stream_action("Generating detailed outline...")

        outline_prompt = f"""
        Create a comprehensive outline for {task_analysis['genre']} content about {task_analysis['topic']}.

        Include:
        1. Main sections/chapters with titles
        2. Key points for each section
        3. Supporting details and examples
        4. Logical flow and transitions
        5. Estimated word counts
        6. Research requirements
        7. Visual elements (if applicable)

        Target length: {task_analysis.get('length', 5000)} words
        Style: {task_analysis.get('style', 'expository')}
        """

        yield self.stream_thinking("Structuring outline...")
        outline_content = await self.generate_response(outline_prompt, temperature=0.6)

        # Parse outline to create structured data
        outline_structure = await self._parse_outline_structure(outline_content)

        # Store outline
        await self.store_memory(
            content=f"Outline for {task_analysis['topic']}: {outline_content}",
            segment_type="CONTENT_OUTLINE",
            importance=0.8,
            metadata={
                "topic": task_analysis['topic'],
                "session_id": session_id,
                "structure": outline_structure,
            },
        )

        yield self.stream_result("Detailed Outline:")
        yield self.stream_result(outline_content)

        # Offer to start writing based on outline
        yield self.stream_action("Would you like me to start writing content based on this outline?")

    async def _handle_general_writing(
        self,
        task_analysis: Dict[str, Any],
        session_id: str,
    ) -> AsyncGenerator[StreamData, None]:
        """Handle general writing requests."""

        yield self.stream_action("Writing content...")

        writing_prompt = f"""
        Write high-quality {task_analysis['genre']} content about {task_analysis['topic']}.

        Requirements:
        - Style: {task_analysis.get('style', 'expository')}
        - Length: approximately {task_analysis.get('length', 1000)} words
        - Engaging and well-structured
        - Clear and compelling narrative
        - Appropriate for the target audience
        """

        yield self.stream_thinking("Crafting content...")

        # Generate content (non-streaming since streaming isn't supported)
        yield self.stream_thinking("Generating content...")
        content = await self.generate_response(writing_prompt, temperature=0.7)

        # Simulate progress updates
        content_length = len(content)
        for i in range(0, content_length, 500):
            if i > 0:
                word_count = len(content[:i].split())
                yield self.stream_action(f"Written {word_count} words...")

        # Store content
        await self.store_memory(
            content=f"Written content about {task_analysis['topic']}: {content}",
            segment_type="WRITTEN_CONTENT",
            importance=0.7,
            metadata={
                "topic": task_analysis['topic'],
                "word_count": len(content.split()),
                "session_id": session_id,
            },
        )

        word_count = len(content.split())
        yield self.stream_result(f"Content completed: {word_count} words")
        yield self.stream_result(content)

