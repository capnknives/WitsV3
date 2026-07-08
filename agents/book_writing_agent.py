# agents/book_writing_agent.py
"""
Book Writing Agent for WitsV3
Specialized agent for creating books, novels, documentation, and long-form content
"""

import asyncio
import json
import re
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator

from agents.base_agent import BaseAgent
from agents.book_writing_handlers import BookWritingHandlersMixin
from agents.book_writing_models import BookStructure, Chapter
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, ConversationHistory
from core.neural_web_core import NeuralWeb

# Re-export models for backward compatibility
__all__ = ["BookWritingAgent", "BookStructure", "Chapter"]


class BookWritingAgent(BookWritingHandlersMixin, BaseAgent):
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
        tool_registry: Optional[Any] = None,
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
            "biography", "mystery", "sci-fi", "fantasy", "romance",
        ]

        self.writing_styles = [
            "narrative", "expository", "descriptive", "persuasive",
            "academic", "conversational", "formal", "creative",
        ]

        # Neural web integration for narrative intelligence
        self.narrative_patterns = {
            "hero_journey": self._analyze_hero_journey,
            "three_act": self._analyze_three_act_structure,
            "spiral": self._analyze_spiral_narrative,
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
        **kwargs,
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
            "parameters": {},
        }

    async def get_writing_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get writing statistics and progress"""

        # Search memory for writing-related content
        writing_memories = []
        if self.memory_manager:
            for segment_type in ['BOOK_PROJECT', 'CHAPTER_CONTENT', 'WRITTEN_CONTENT']:
                memories = await self.memory_manager.search_memory(
                    segment_type,
                    filter_dict={'session_id': session_id} if session_id else None,
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
            'average_words_per_session': total_words / max(len(writing_memories), 1),
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
            llm_interface=llm_interface,
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
