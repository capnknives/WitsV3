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
    from agents.enhanced_book_agent_with_fallback import EnhancedBookAgentWithFallback
except ImportError:
    logging.warning("Could not import WitsV3 agents directly. Will use API mode.")

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
        self.enhanced_agent = None
        self.session = None

    async def initialize(self):
        """Initialize the connector"""
        if self.use_api:
            self.session = aiohttp.ClientSession()
        else:
            try:
                self.agent = BookWritingAgent()
                self.enhanced_agent = EnhancedBookAgentWithFallback()
                await self.agent.initialize()
                await self.enhanced_agent.initialize()
                logger.info("Book writing agents initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize book writing agents: {e}")
                self.use_api = True
                self.session = aiohttp.ClientSession()

    async def generate_book(self, book_config: Dict[str, Any], use_enhanced: bool = True) -> str:
        """
        Generate a complete book.

        Args:
            book_config: Configuration for the book
            use_enhanced: Whether to use the enhanced agent

        Returns:
            The generated book content
        """
        if self.use_api:
            return await self._api_generate_book(book_config, use_enhanced)

        agent = self.enhanced_agent if use_enhanced else self.agent

        try:
            result = await agent.generate_book(
                title=book_config.get("title", "Untitled"),
                genre=book_config.get("genre", "Fiction"),
                characters=book_config.get("characters", []),
                chapters=book_config.get("chapters", [])
            )
            return result
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
        if self.use_api:
            return await self._api_generate_chapter(book_id, chapter_config)

        try:
            result = await self.agent.generate_chapter(
                book_title=chapter_config.get("book_title", "Untitled"),
                chapter_title=chapter_config.get("title", "Untitled Chapter"),
                chapter_outline=chapter_config.get("outline", ""),
                characters=chapter_config.get("characters", [])
            )
            return result
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
        if self.use_api:
            return await self._api_research_topic(topic, depth)

        try:
            # Pass depth to the agent if the method supports it
            try:
                result = await self.agent.research_topic(topic, depth)
            except TypeError:
                # Fallback to just topic if the agent doesn't support depth
                result = await self.agent.research_topic(topic)
            return result
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
