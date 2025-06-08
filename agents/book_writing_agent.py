# agents/book_writing_agent.py
"""
Book Writing Agent for WitsV3
Specialized agent for creating books, novels, documentation, and long-form content
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime

from agents.base_agent import BaseAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, ConversationHistory
from core.neural_web_core import NeuralWeb


@dataclass
class BookStructure:
    """Represents the structure of a book"""
    title: str
    subtitle: Optional[str] = None
    genre: str = "non-fiction"
    target_length: int = 50000  # words
    chapters: List[Dict[str, Any]] = field(default_factory=list)
    style_guide: Dict[str, Any] = field(default_factory=dict)
    research_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.chapters is None:
            self.chapters = []
        if self.style_guide is None:
            self.style_guide = {}
        if self.research_notes is None:
            self.research_notes = []


@dataclass
class Chapter:
    """Represents a book chapter"""
    id: str
    title: str
    outline: str
    content: str = ""
    word_count: int = 0
    status: str = "planned"  # planned, outlined, drafted, revised, complete
    dependencies: List[str] = field(default_factory=list)  # Use default_factory for mutable types
    research_requirements: List[str] = field(default_factory=list)  # Use default_factory for mutable types
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.research_requirements is None:
            self.research_requirements = []


class BookWritingAgent(BaseAgent):
    """
    Specialized agent for book writing and long-form content creation
    """
    
    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: Optional[MemoryManager] = None,
        neural_web: Optional[NeuralWeb] = None,
        tool_registry: Optional[Any] = None
    ):
        super().__init__(agent_name, config, llm_interface, memory_manager)
        
        self.neural_web = neural_web
        self.tool_registry = tool_registry
        
        # Book writing specific state
        self.current_books: Dict[str, BookStructure] = {}
        self.writing_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Writing capabilities
        self.genres = [
            "fiction", "non-fiction", "technical", "academic", 
            "biography", "mystery", "sci-fi", "fantasy", "romance"
        ]
        
        self.writing_styles = [
            "narrative", "expository", "descriptive", "persuasive",
            "academic", "conversational", "formal", "creative"
        ]
        
        # Neural web integration for narrative intelligence
        self.narrative_patterns = {
            "hero_journey": self._analyze_hero_journey,
            "three_act": self._analyze_three_act_structure,
            "spiral": self._analyze_spiral_narrative
        }
        
        self.character_networks: Dict[str, Any] = {}
        self.world_building_graph: Dict[str, Any] = {}
        
        # Enhanced writing capabilities with neural web
        if self.neural_web:
            self.enable_narrative_intelligence = True
            self.logger.info("Neural web enabled for narrative intelligence")
        else:
            self.enable_narrative_intelligence = False
        
        self.logger.info("Book Writing Agent initialized")
    
    async def run(
        self,
        user_input: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Process book writing requests
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        yield self.stream_thinking("Analyzing book writing request...")
        
        # Parse the request to understand what type of book writing task this is
        task_analysis = await self._analyze_writing_task(user_input)
        
        yield self.stream_thinking(f"Identified task type: {task_analysis['task_type']}")
        
        # Route to appropriate handler
        if task_analysis['task_type'] == 'create_book':
            async for stream in self._handle_book_creation(task_analysis, session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'write_chapter':
            async for stream in self._handle_chapter_writing(task_analysis, session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'research_topic':
            async for stream in self._handle_research(task_analysis, session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'revise_content':
            async for stream in self._handle_revision(task_analysis, session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'generate_outline':
            async for stream in self._handle_outline_generation(task_analysis, session_id):
                yield stream
        
        else:
            async for stream in self._handle_general_writing(task_analysis, session_id):
                yield stream
    
    async def _analyze_writing_task(self, request: str) -> Dict[str, Any]:
        """Analyze the writing request to determine task type and parameters"""
        
        analysis_prompt = f"""
        Analyze this book writing request and determine the task type and parameters:
        
        Request: {request}
        
        Respond with JSON containing:
        {{
            "task_type": "create_book" | "write_chapter" | "research_topic" | "revise_content" | "generate_outline" | "general_writing",
            "genre": "fiction" | "non-fiction" | "technical" | "academic" | etc.,
            "style": "narrative" | "expository" | "academic" | etc.,
            "topic": "main topic or theme",
            "length": estimated length in words,
            "parameters": {{additional specific parameters}}
        }}
        """
        
        try:
            response = await self.generate_response(analysis_prompt, temperature=0.3)
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            self.logger.warning(f"Failed to parse task analysis: {e}")
        
        # Fallback analysis with better genre detection
        genre = "non-fiction"
        if any(keyword in request.lower() for keyword in ["horror", "scary", "thriller", "suspense"]):
            genre = "horror"
        elif any(keyword in request.lower() for keyword in ["fiction", "novel", "story", "fantasy", "sci-fi", "romance"]):
            genre = "fiction"
            
        # Detect if this is a book creation request
        task_type = "general_writing"
        if any(phrase in request.lower() for phrase in ["write a book", "create a book", "make a book", "write me a book"]):
            task_type = "create_book"
            
        return {
            "task_type": task_type,
            "genre": genre,
            "style": "narrative" if genre in ["horror", "fiction"] else "expository",
            "topic": request[:100],
            "length": 10000 if task_type == "create_book" else 1000,
            "parameters": {}
        }
    
    async def _handle_book_creation(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle creation of a new book project"""
        
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
            target_length=task_analysis.get('length', 50000)
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
            metadata={"book_id": book_id, "session_id": session_id}
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
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle writing of a specific chapter"""
        
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
            status="drafted"
        )
        
        # Store in memory
        await self.store_memory(
            content=f"Wrote chapter: {chapter.title} ({chapter.word_count} words)",
            segment_type="CHAPTER_CONTENT",
            importance=0.8,
            metadata={"chapter_id": chapter_id, "session_id": session_id}
        )
        
        yield self.stream_result(f"Completed chapter: {chapter.word_count} words written")
        
        # Offer to continue with related chapters
        yield self.stream_action("Would you like me to suggest related chapters or continue this story?")
    
    async def _handle_research(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle research for book topics"""
        
        yield self.stream_action("Conducting research...")
        
        topic = task_analysis['topic']
        
        # Use neural web to find related concepts
        if self.neural_web:
            yield self.stream_thinking("Searching knowledge network for related concepts...")
            
            # Add topic to neural web if not present
            await self.neural_web.add_concept(
                topic.replace(" ", "_"), 
                f"Research topic: {topic}", 
                "research_topic"
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
                    query=f"{topic} research facts statistics"
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
            metadata={"topic": topic, "session_id": session_id}
        )
        
        # Add research concepts to neural web
        if self.neural_web:
            research_concepts = await self._extract_research_concepts(research_synthesis)
            for concept in research_concepts:
                await self.neural_web.add_concept(
                    concept.replace(" ", "_"),
                    f"Research finding: {concept}",
                    "research_finding"
                )
                # Connect to main topic
                await self.neural_web.connect_concepts(
                    topic.replace(" ", "_"),
                    concept.replace(" ", "_"),
                    "supports",
                    strength=0.8
                )
        
        yield self.stream_result(f"Research completed on {topic}")
        yield self.stream_result(research_synthesis)
    
    async def _handle_revision(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle content revision and improvement"""
        
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
            metadata={"topic": task_analysis['topic'], "session_id": session_id}
        )
    
    async def _handle_outline_generation(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle generation of book or chapter outlines"""
        
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
                "structure": outline_structure
            }
        )
        
        yield self.stream_result("Detailed Outline:")
        yield self.stream_result(outline_content)
        
        # Offer to start writing based on outline
        yield self.stream_action("Would you like me to start writing content based on this outline?")
    
    async def _handle_general_writing(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle general writing requests"""
        
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
                "session_id": session_id
            }
        )
        
        word_count = len(content.split())
        yield self.stream_result(f"Content completed: {word_count} words")
        yield self.stream_result(content)
    
    async def _extract_chapters_from_response(
        self, 
        response: str, 
        book_id: str
    ) -> List[Dict[str, Any]]:
        """Extract chapter information from LLM response"""
        
        # Simple parsing - could be enhanced with more sophisticated NLP
        chapters = []
        lines = response.split('\n')
        
        current_chapter = None
        for line in lines:
            line = line.strip()
            if line.startswith('Chapter') or line.startswith('ch.') or line.startswith('#'):
                if current_chapter:
                    chapters.append(current_chapter)
                
                current_chapter = {
                    'id': str(uuid.uuid4()),
                    'title': line,
                    'outline': '',
                    'word_target': 2000,
                    'status': 'planned'
                }
            elif current_chapter and line:
                current_chapter['outline'] += line + ' '
        
        if current_chapter:
            chapters.append(current_chapter)
        
        # If no chapters found, create a basic structure
        if not chapters:
            for i in range(5):
                chapters.append({
                    'id': str(uuid.uuid4()),
                    'title': f"Chapter {i+1}",
                    'outline': f"Content for chapter {i+1}",
                    'word_target': 2000,
                    'status': 'planned'
                })
        
        return chapters
    
    async def _generate_chapter_outline(
        self, 
        book_id: str, 
        chapter_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Generate detailed outline for a specific chapter"""
        
        if book_id not in self.current_books:
            yield self.stream_error("Book not found")
            return
        
        book = self.current_books[book_id]
        chapter_info = None
        
        for chapter in book.chapters:
            if chapter['id'] == chapter_id:
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
        
        # Update chapter with detailed outline
        chapter_info['detailed_outline'] = detailed_outline
        chapter_info['status'] = 'outlined'
        
        yield self.stream_result(f"Detailed outline for {chapter_info['title']}:")
        yield self.stream_result(detailed_outline)
        
        # Store outline in memory
        await self.store_memory(
            content=f"Chapter outline: {detailed_outline}",
            segment_type="CHAPTER_OUTLINE",
            importance=0.8,
            metadata={
                "book_id": book_id,
                "chapter_id": chapter_id,
                "chapter_title": chapter_info['title']
            }
        )
    
    async def _extract_research_concepts(self, research_text: str) -> List[str]:
        """Extract key concepts from research text"""
        
        # Simple keyword extraction - could be enhanced with NLP
        concepts = []
        
        # Look for bullet points, numbered items, key phrases
        import re
        
        # Find items that look like concepts (capitalized, important terms)
        concept_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\d+%|\$\d+|\d+\s+(?:million|billion|thousand)',  # Statistics
            r'(?:research|study|survey|report)\s+(?:shows|indicates|suggests|found)',  # Research findings
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, research_text)
            concepts.extend(matches)
        
        # Remove duplicates and filter
        unique_concepts = list(set(concepts))
        
        # Filter out common words
        filtered_concepts = [
            concept for concept in unique_concepts 
            if len(concept.split()) <= 3 and len(concept) > 3
        ]
        
        return filtered_concepts[:10]  # Return top 10 concepts
    
    async def _parse_outline_structure(self, outline_text: str) -> Dict[str, Any]:
        """Parse outline text into structured data"""
        
        structure = {
            'sections': [],
            'estimated_length': 0,
            'key_topics': []
        }
        
        lines = outline_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers (various formats)
            if (line.startswith(('#', '##', '1.', '2.', 'I.', 'A.')) or 
                line.isupper() or 
                line.endswith(':')):
                
                if current_section:
                    structure['sections'].append(current_section)
                
                current_section = {
                    'title': line.rstrip(':'),
                    'points': [],
                    'estimated_words': 500
                }
            
            elif current_section and line.startswith(('-', '*', '•')):
                current_section['points'].append(line[1:].strip())
            
            elif current_section:
                current_section['points'].append(line)
        
        if current_section:
            structure['sections'].append(current_section)
        
        # Calculate estimated length
        structure['estimated_length'] = len(structure['sections']) * 500
        
        # Extract key topics
        structure['key_topics'] = [
            section['title'] for section in structure['sections']
        ]
        
        return structure
    
    async def get_writing_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get writing statistics and progress"""
        
        # Search memory for writing-related content
        writing_memories = []
        if self.memory_manager:
            for segment_type in ['BOOK_PROJECT', 'CHAPTER_CONTENT', 'WRITTEN_CONTENT']:
                memories = await self.memory_manager.search_memory(
                    segment_type, 
                    filter_dict={'session_id': session_id} if session_id else None
                )
                writing_memories.extend(memories)
        
        # Calculate statistics
        total_books = len(self.current_books)
        total_chapters = sum(len(book.chapters) for book in self.current_books.values())
        total_words = 0
        
        for memory in writing_memories:
            if hasattr(memory, 'metadata') and 'word_count' in memory.metadata:
                total_words += memory.metadata['word_count']
        
        return {
            'total_books': total_books,
            'total_chapters': total_chapters,
            'total_words_written': total_words,
            'writing_sessions': len(writing_memories),
            'current_projects': list(self.current_books.keys()),
            'average_words_per_session': total_words / max(len(writing_memories), 1)
        }
    
    async def _analyze_hero_journey(self, content: str) -> Dict[str, Any]:
        """Analyze content using hero's journey structure"""
        stages = [
            "ordinary_world", "call_to_adventure", "refusal", "mentor",
            "crossing_threshold", "tests", "ordeal", "reward", "return"
        ]
        
        analysis = {"structure": "hero_journey", "stages_present": []}
        
        # Use neural web to identify narrative patterns
        if self.neural_web and self.enable_narrative_intelligence:
            for stage in stages:
                stage_concepts = await self.neural_web._find_relevant_concepts(f"{stage} {content}")
                if stage_concepts:
                    analysis["stages_present"].append(stage)
        
        return analysis
    
    async def _analyze_three_act_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content using three-act structure"""
        acts = ["setup", "confrontation", "resolution"]
        
        analysis = {"structure": "three_act", "acts_present": []}
        
        if self.neural_web and self.enable_narrative_intelligence:
            for act in acts:
                act_concepts = await self.neural_web._find_relevant_concepts(f"{act} {content}")
                if act_concepts:
                    analysis["acts_present"].append(act)
        
        return analysis
    
    async def _analyze_spiral_narrative(self, content: str) -> Dict[str, Any]:
        """Analyze content using spiral narrative structure"""
        return {
            "structure": "spiral",
            "themes_present": [],
            "recurring_motifs": []
        }
    


# Test function
async def test_book_writing_agent():
    """Test the book writing agent functionality"""
    from core.config import load_config
    from core.llm_interface import OllamaInterface
    
    try:
        config = load_config("config.yaml")
        llm_interface = OllamaInterface(config=config)
        
        agent = BookWritingAgent(
            agent_name="BookWriter",
            config=config,
            llm_interface=llm_interface
        )
        
        print("Testing book writing agent...")
        
        # Test book creation
        async for stream_data in agent.run("Create a science fiction book about AI consciousness"):
            print(f"[{stream_data.type.upper()}] {stream_data.content}")
        
        # Get statistics
        stats = await agent.get_writing_statistics()
        print(f"Writing statistics: {stats}")
        
    except Exception as e:
        print(f"Test completed with expected errors: {e}")


if __name__ == "__main__":
    asyncio.run(test_book_writing_agent())
