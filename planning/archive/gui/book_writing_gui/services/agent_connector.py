import asyncio
import logging
import json
import sys
import os
from typing import Dict, List, Any, Optional
import aiohttp
import time

# Add the parent directory to the path to import WitsV3 modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    from agents.book_writing_agent import BookWritingAgent
    from core.config import WitsV3Config
    from core.llm_interface import OllamaInterface
    
    # Import the load_config function directly
    from core.config import load_config
except ImportError as e:
    logging.warning(f"Could not import WitsV3 modules: {e}. Will use API mode.")

logger = logging.getLogger(__name__)

class BookWritingAgentConnector:
    """Connector for the Book Writing Agent"""

    def __init__(self, use_api: bool = False, api_url: str = "http://localhost:8001"):
        """
        Initialize the connector.

        Args:
            use_api: Whether to use the API or direct agent instantiation
            api_url: The URL of the API if use_api is True
        """
        self.use_api = use_api
        self.api_url = api_url
        self.agent = None
        self.session = None

    async def initialize(self):
        """Initialize the connector"""
        logger.info("Initializing BookWritingAgentConnector")
        if self.use_api:
            logger.info("Using API mode for agent connector")
            self.session = aiohttp.ClientSession()
            logger.info("API session created successfully")
        else:
            try:
                logger.info("Using direct agent mode for connector")
                # Load config
                config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../config.yaml"))
                logger.info(f"Loading config from {config_path}")
                config = load_config(config_path)
                
                # Initialize LLM interface
                logger.info("Initializing LLM interface")
                llm_interface = OllamaInterface(config=config)
                
                # Initialize agent with required parameters
                logger.info("Creating BookWritingAgent instance")
                self.agent = BookWritingAgent(
                    agent_name="BookWriter",
                    config=config,
                    llm_interface=llm_interface
                )
                
                logger.info("Book writing agent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize book writing agent: {e}")
                logger.info("Falling back to API mode")
                self.use_api = True
                self.session = aiohttp.ClientSession()
                logger.info("API session created successfully (fallback)")

    async def generate_book(self, book_config: Dict[str, Any], use_enhanced: bool = False) -> str:
        """
        Generate a complete book.

        Args:
            book_config: Configuration for the book
            use_enhanced: Whether to use the enhanced agent (ignored in direct mode)

        Returns:
            The generated book content
        """
        if self.use_api or self.agent is None:
            if not self.use_api:
                logger.warning("Agent is None, falling back to API mode")
                self.use_api = True
                if not self.session:
                    self.session = aiohttp.ClientSession()
            return await self._api_generate_book(book_config, use_enhanced)

        try:
            # Prepare the input for the agent's run method
            user_input = f"Create a {book_config.get('genre', 'fiction')} book about {book_config.get('title', 'an interesting topic')}"
            
            # Collect the results from the agent's run method
            result_content = ""
            async for stream_data in self.agent.run(user_input):
                if stream_data.type == "result":
                    result_content += stream_data.content + "\n"
            
            return result_content
        except Exception as e:
            logger.error(f"Error generating book: {e}")
            # Fallback to API if direct agent fails
            self.use_api = True
            if not self.session:
                self.session = aiohttp.ClientSession()
            return await self._api_generate_book(book_config, use_enhanced)

    async def generate_chapter(self, book_id: str, chapter_config: Dict[str, Any]) -> str:
        """
        Generate content for a chapter.

        Args:
            book_id: The ID of the book
            chapter_config: Configuration for the chapter

        Returns:
            The generated chapter content
        """
        if self.use_api or self.agent is None:
            if not self.use_api:
                logger.warning("Agent is None, falling back to API mode")
                self.use_api = True
                if not self.session:
                    self.session = aiohttp.ClientSession()
            return await self._api_generate_chapter(book_id, chapter_config)

        try:
            # Prepare the input for the agent's run method
            user_input = f"Write chapter '{chapter_config.get('title', 'Untitled Chapter')}' for the book '{chapter_config.get('book_title', 'Untitled')}'. Chapter outline: {chapter_config.get('outline', '')}"
            
            # Collect the results from the agent's run method
            result_content = ""
            async for stream_data in self.agent.run(user_input):
                if stream_data.type == "result":
                    result_content += stream_data.content + "\n"
            
            return result_content
        except Exception as e:
            logger.error(f"Error generating chapter: {e}")
            # Fallback to API if direct agent fails
            self.use_api = True
            if not self.session:
                self.session = aiohttp.ClientSession()
            return await self._api_generate_chapter(book_id, chapter_config)

    async def research_topic(self, topic: str, depth: str = "basic") -> Dict[str, Any]:
        """
        Research a topic for book writing.

        Args:
            topic: The topic to research
            depth: Research depth (basic, detailed, comprehensive)

        Returns:
            Research results
        """
        if self.use_api or self.agent is None:
            if not self.use_api:
                logger.warning("Agent is None, falling back to API mode")
                self.use_api = True
                if not self.session:
                    self.session = aiohttp.ClientSession()
            return await self._api_research_topic(topic, depth)

        try:
            # Prepare the input for the agent's run method
            user_input = f"Research the topic '{topic}' with {depth} depth for book writing purposes."
            
            # Collect the results from the agent's run method
            research_results = {}
            research_content = ""
            async for stream_data in self.agent.run(user_input):
                if stream_data.type == "result":
                    research_content += stream_data.content + "\n"
            
            # Format the research results
            research_results = {
                "topic": topic,
                "depth": depth,
                "content": research_content,
                "timestamp": time.time()
            }
            
            return research_results
        except Exception as e:
            logger.error(f"Error researching topic: {e}")
            # Fallback to API if direct agent fails
            self.use_api = True
            if not self.session:
                self.session = aiohttp.ClientSession()
            return await self._api_research_topic(topic, depth)

    async def _api_generate_book(self, book_config: Dict[str, Any], use_enhanced: bool) -> str:
        """Call the API to generate a book"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.api_url}/generate/book"
        payload = {
            "config": book_config,
            "use_enhanced": use_enhanced
        }

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("content", "")
                else:
                    error = await response.text()
                    logger.error(f"API error: {error}")
                    return f"Error generating book: {error}"
        except Exception as e:
            logger.error(f"API request error: {e}")
            return f"Error connecting to API: {e}"

    async def _api_generate_chapter(self, book_id: str, chapter_config: Dict[str, Any]) -> str:
        """Call the API to generate a chapter"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.api_url}/generate/chapter"
        payload = {
            "book_id": book_id,
            "config": chapter_config
        }

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("content", "")
                else:
                    error = await response.text()
                    logger.error(f"API error: {error}")
                    return f"Error generating chapter: {error}"
        except Exception as e:
            logger.error(f"API request error: {e}")
            return f"Error connecting to API: {e}"

    async def _api_research_topic(self, topic: str, depth: str = "basic") -> Dict[str, Any]:
        """Call the API to research a topic"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.api_url}/research"
        payload = {
            "topic": topic,
            "depth": depth
        }

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error = await response.text()
                    logger.error(f"API error: {error}")
                    return {"error": f"Error researching topic: {error}"}
        except Exception as e:
            logger.error(f"API request error: {e}")
            return {"error": f"Error connecting to API: {e}"}

    async def close(self):
        """Close the connector"""
        if self.session:
            await self.session.close()

    async def __aenter__(self):
        """Async context manager enter"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
