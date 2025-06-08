"""
Task handlers for book writing agent
"""

import re
import json
import uuid
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List

from core.schemas import StreamData
from .models import BookStructure, Chapter, SUPPORTED_GENRES, WRITING_STYLES


class BookWritingHandlers:
    """Handles different book writing tasks"""
    
    def __init__(self, agent):
        """Initialize handlers with reference to parent agent"""
        self.agent = agent
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def handle_book_creation(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle creation of a new book project"""
        
        yield self.agent.stream_action("Creating new book project...")
        
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
        
        yield self.agent.stream_thinking("Generating book structure...")
        structure_response = await self.agent.generate_response(structure_prompt, temperature=0.7)
        
        # Create book structure object
        book_structure = BookStructure(
            title=f"Book about {task_analysis['topic']}",  # Will be refined
            genre=task_analysis['genre'],
            target_length=task_analysis.get('length', 50000)
        )
        
        # Parse structure response to create chapters
        chapters = await self._extract_chapters_from_response(structure_response, book_id)
        book_structure.chapters = chapters
        
        self.agent.current_books[book_id] = book_structure
        
        # Store in memory
        await self.agent.store_memory(
            content=f"Created book project: {book_structure.title}",
            segment_type="BOOK_PROJECT",
            importance=0.9,
            metadata={"book_id": book_id, "session_id": session_id}
        )
        
        yield self.agent.stream_result(f"Created book project '{book_structure.title}' with {len(chapters)} chapters")
        
        # Generate first chapter outline
        if chapters:
            yield self.agent.stream_action("Generating outline for first chapter...")
            async for stream in self._generate_chapter_outline(book_id, chapters[0]['id']):
                yield stream
    
    async def handle_chapter_writing(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle writing of a specific chapter"""
        
        yield self.agent.stream_action("Writing chapter content...")
        
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
        
        yield self.agent.stream_thinking("Crafting chapter content...")
        
        # Generate chapter content (non-streaming since streaming isn't supported)
        yield self.agent.stream_thinking("Generating chapter content...")
        chapter_content = await self.agent.generate_response(writing_prompt, temperature=0.8)
        
        # Simulate progress updates
        content_length = len(chapter_content)
        for i in range(0, content_length, 500):
            if i > 0:
                word_count = len(chapter_content[:i].split())
                yield self.agent.stream_action(f"Written {word_count} words...")
        
        # Create chapter object
        chapter = Chapter(
            id=chapter_id,
            title=f"Chapter: {task_analysis['topic']}",
            outline="",
            content=chapter_content,
            word_count=len(chapter_content.split()),
            status="drafted"
        )
        
        # Store in memory
        await self.agent.store_memory(
            content=f"Wrote chapter: {chapter.title} ({chapter.word_count} words)",
            segment_type="CHAPTER_CONTENT",
            importance=0.8,
            metadata={"chapter_id": chapter_id, "session_id": session_id}
        )
        
        yield self.agent.stream_result(f"Completed chapter: {chapter.word_count} words written")
        
        # Offer to continue with related chapters
        yield self.agent.stream_action("Would you like me to suggest related chapters or continue this story?")
    
    async def handle_research(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle research for book topics"""
        
        yield self.agent.stream_action("Conducting research...")
        
        topic = task_analysis['topic']
        
        # Use neural web to find related concepts
        if self.agent.neural_web:
            yield self.agent.stream_thinking("Searching knowledge network for related concepts...")
            
            # Add topic to neural web if not present
            await self.agent.neural_web.add_concept(
                topic.replace(" ", "_"), 
                f"Research topic: {topic}", 
                "research_topic"
            )
            
            # Find related concepts through activation
            related_concepts = await self.agent.neural_web.activate_concept(topic.replace(" ", "_"))
            
            if related_concepts:
                yield self.agent.stream_observation(f"Found {len(related_concepts)} related concepts in knowledge network")
        
        # Use tools for web research if available
        research_results = []
        if self.agent.tool_registry:
            try:
                # Try to use web search tool
                search_result = await self.agent.tool_registry.execute_tool(
                    "web_search", 
                    query=f"{topic} research facts statistics"
                )
                research_results.append(search_result)
                yield self.agent.stream_observation("Gathered web research data")
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
        
        yield self.agent.stream_thinking("Synthesizing research findings...")
        research_synthesis = await self.agent.generate_response(research_prompt, temperature=0.5)
        
        # Store research in memory with high importance
        await self.agent.store_memory(
            content=f"Research on {topic}: {research_synthesis}",
            segment_type="RESEARCH_NOTES",
            importance=0.9,
            metadata={"topic": topic, "session_id": session_id}
        )
        
        # Add research concepts to neural web
        if self.agent.neural_web:
            research_concepts = await self._extract_research_concepts(research_synthesis)
            for concept in research_concepts:
                await self.agent.neural_web.add_concept(
                    concept.replace(" ", "_"),
                    f"Research finding: {concept}",
                    "research_finding"
                )
                # Connect to main topic
                await self.agent.neural_web.connect_concepts(
                    topic.replace(" ", "_"),
                    concept.replace(" ", "_"),
                    "supports",
                    strength=0.8
                )
        
        yield self.agent.stream_result(f"Research completed on {topic}")
        yield self.agent.stream_result(research_synthesis)
    
    async def handle_revision(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle content revision and improvement"""
        
        yield self.agent.stream_action("Analyzing content for revision...")
        
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
        
        yield self.agent.stream_thinking("Generating revision guidelines...")
        revision_guidelines = await self.agent.generate_response(revision_prompt, temperature=0.6)
        
        yield self.agent.stream_result("Revision Guidelines:")
        yield self.agent.stream_result(revision_guidelines)
        
        # Store revision guidelines
        await self.agent.store_memory(
            content=f"Revision guidelines for {task_analysis['topic']}: {revision_guidelines}",
            segment_type="REVISION_GUIDELINES",
            importance=0.7,
            metadata={"topic": task_analysis['topic'], "session_id": session_id}
        )
    
    async def handle_outline_generation(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle generation of book or chapter outlines"""
        
        yield self.agent.stream_action("Generating detailed outline...")
        
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
        
        yield self.agent.stream_thinking("Structuring outline...")
        outline_content = await self.agent.generate_response(outline_prompt, temperature=0.6)
        
        # Parse outline to create structured data
        outline_structure = await self._parse_outline_structure(outline_content)
        
        # Store outline
        await self.agent.store_memory(
            content=f"Outline for {task_analysis['topic']}: {outline_content}",
            segment_type="CONTENT_OUTLINE",
            importance=0.8,
            metadata={
                "topic": task_analysis['topic'],
                "session_id": session_id,
                "structure": outline_structure
            }
        )
        
        yield self.agent.stream_result("Detailed Outline:")
        yield self.agent.stream_result(outline_content)
        
        # Offer to begin writing
        yield self.agent.stream_action("Would you like me to begin writing based on this outline?")
    
    async def handle_general_writing(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle general writing requests"""
        
        yield self.agent.stream_action("Processing writing request...")
        
        # Determine the best approach based on the request
        topic = task_analysis['topic']
        genre = task_analysis.get('genre', 'general')
        style = task_analysis.get('style', 'narrative')
        
        # Generate appropriate content
        writing_prompt = f"""
        Create {genre} content about: {topic}
        
        Style: {style}
        Length: approximately {task_analysis.get('length', 1000)} words
        
        Ensure the content is:
        - Well-structured with clear beginning, middle, and end
        - Engaging and appropriate for the genre
        - Properly formatted with paragraphs
        - Consistent in tone and voice
        """
        
        yield self.agent.stream_thinking("Generating content...")
        content = await self.agent.generate_response(writing_prompt, temperature=0.7)
        
        # Store the generated content
        await self.agent.store_memory(
            content=f"Generated {genre} content: {content[:500]}...",
            segment_type="GENERATED_CONTENT",
            importance=0.7,
            metadata={
                "topic": topic,
                "genre": genre,
                "style": style,
                "word_count": len(content.split()),
                "session_id": session_id
            }
        )
        
        yield self.agent.stream_result(content)
        
        # Analyze narrative structure if applicable
        if genre in ["fiction", "story", "narrative"]:
            yield self.agent.stream_action("Analyzing narrative structure...")
            analysis = await self.agent.narrative_analyzer.analyze_narrative_structure(content)
            
            if analysis.get("recommendations"):
                yield self.agent.stream_observation("Structural recommendations:")
                for rec in analysis["recommendations"]:
                    yield self.agent.stream_observation(f"- {rec}")
    
    # Helper methods
    async def _extract_chapters_from_response(
        self, 
        response: str, 
        book_id: str
    ) -> List[Dict[str, Any]]:
        """Extract chapter information from LLM response"""
        chapters = []
        
        # Simple extraction - look for numbered chapters
        lines = response.split('\n')
        chapter_pattern = r'(?:Chapter\s+(\d+)|(\d+)\.|Part\s+(\d+))'
        
        for i, line in enumerate(lines):
            match = re.search(chapter_pattern, line, re.IGNORECASE)
            if match:
                chapter_num = match.group(1) or match.group(2) or match.group(3)
                
                # Extract title (rest of the line after chapter number)
                title_match = re.search(r'(?:Chapter\s+\d+\s*:?\s*|^\d+\.\s*)(.*)', line, re.IGNORECASE)
                title = title_match.group(1) if title_match else f"Chapter {chapter_num}"
                
                # Look for description in the next few lines
                description = ""
                for j in range(i+1, min(i+4, len(lines))):
                    if lines[j].strip() and not re.search(chapter_pattern, lines[j]):
                        description += lines[j].strip() + " "
                
                chapters.append({
                    'id': f"{book_id}_ch{chapter_num}",
                    'number': int(chapter_num),
                    'title': title.strip(),
                    'description': description.strip()
                })
        
        # If no chapters found, create a default structure
        if not chapters:
            for i in range(1, 11):  # Default 10 chapters
                chapters.append({
                    'id': f"{book_id}_ch{i}",
                    'number': i,
                    'title': f"Chapter {i}",
                    'description': "To be developed"
                })
        
        return chapters
    
    async def _generate_chapter_outline(
        self, 
        book_id: str, 
        chapter_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Generate outline for a specific chapter"""
        book = self.agent.current_books.get(book_id)
        if not book:
            yield self.agent.stream_error("Book not found")
            return
        
        # Find chapter info
        chapter_info = next((ch for ch in book.chapters if ch['id'] == chapter_id), None)
        if not chapter_info:
            yield self.agent.stream_error("Chapter not found")
            return
        
        outline_prompt = f"""
        Create a detailed outline for {chapter_info['title']} in a {book.genre} book.
        
        Chapter description: {chapter_info.get('description', 'N/A')}
        
        Include:
        1. Opening hook
        2. Main points/scenes (3-5)
        3. Key information to convey
        4. Character development (if applicable)
        5. Transition to next chapter
        
        Target length: {book.target_length // len(book.chapters)} words
        """
        
        outline = await self.agent.generate_response(outline_prompt, temperature=0.6)
        
        # Store outline
        await self.agent.store_memory(
            content=f"Chapter outline for {chapter_info['title']}: {outline}",
            segment_type="CHAPTER_OUTLINE",
            importance=0.8,
            metadata={"book_id": book_id, "chapter_id": chapter_id}
        )
        
        yield self.agent.stream_result(f"Outline for {chapter_info['title']}:")
        yield self.agent.stream_result(outline)
    
    async def _extract_research_concepts(self, research_text: str) -> List[str]:
        """Extract key concepts from research text"""
        # Simple extraction based on capitalized phrases and key terms
        concepts = []
        
        # Look for capitalized phrases (potential proper nouns/concepts)
        capitalized_pattern = r'[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*'
        matches = re.findall(capitalized_pattern, research_text)
        
        # Filter out common words and add unique concepts
        common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An'}
        for match in matches:
            if match not in common_words and len(match.split()) <= 3:
                concepts.append(match)
        
        # Also look for terms in quotes
        quoted_pattern = r'"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, research_text)
        concepts.extend(quoted_matches)
        
        # Deduplicate
        return list(set(concepts))[:20]  # Limit to 20 concepts
    
    async def _parse_outline_structure(self, outline_text: str) -> Dict[str, Any]:
        """Parse outline text into structured format"""
        structure = {
            "sections": [],
            "total_estimated_words": 0,
            "research_needed": []
        }
        
        # Parse sections (look for numbered or bulleted items)
        section_pattern = r'(?:^|\n)\s*(?:\d+\.|[-•*])\s*(.+)'
        matches = re.findall(section_pattern, outline_text, re.MULTILINE)
        
        for match in matches:
            section = {
                "title": match.strip(),
                "subsections": [],
                "estimated_words": 500  # Default estimate
            }
            structure["sections"].append(section)
        
        structure["total_estimated_words"] = len(structure["sections"]) * 500
        
        # Look for research requirements
        if "research" in outline_text.lower():
            research_lines = [line for line in outline_text.split('\n') if 'research' in line.lower()]
            structure["research_needed"] = research_lines[:5]
        
        return structure