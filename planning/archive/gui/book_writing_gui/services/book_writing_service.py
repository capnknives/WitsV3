import asyncio
import logging
from typing import Dict, List, Optional, Any
import uuid
import os
from datetime import datetime

from .agent_connector import BookWritingAgentConnector
from ..models.database import Database

logger = logging.getLogger(__name__)

class BookWritingService:
    """Service for managing book writing operations with persistent storage"""
    
    def __init__(self, data_dir: str = "data"):
        self.agent_connector = None
        self.db = Database(data_dir)
        self.generation_tasks = {}  # In-memory storage for generation tasks
    
    async def initialize(self):
        """Initialize the book writing service"""
        try:
            # Ensure data directory exists
            os.makedirs(self.db.data_dir, exist_ok=True)
            
            # Initialize agent connector
            self.agent_connector = BookWritingAgentConnector()
            await self.agent_connector.initialize()
            
            logger.info("Book writing service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize book writing service: {e}")
            raise
    
    # Book operations
    async def create_book(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new book"""
        return self.db.create_book(book_data)
    
    async def get_book(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Get a book by ID"""
        return self.db.get_book(book_id)
    
    async def get_all_books(self) -> List[Dict[str, Any]]:
        """Get all books"""
        return self.db.get_all_books()
    
    async def update_book(self, book_id: str, book_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a book"""
        return self.db.update_book(book_id, book_data)
    
    async def delete_book(self, book_id: str) -> bool:
        """Delete a book"""
        return self.db.delete_book(book_id)
    
    # Character operations
    async def add_character(self, book_id: str, character_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add a character to a book"""
        return self.db.add_character(book_id, character_data)
    
    async def get_character(self, book_id: str, character_id: str) -> Optional[Dict[str, Any]]:
        """Get a character by ID"""
        return self.db.get_character(book_id, character_id)
    
    async def get_all_characters(self, book_id: str) -> List[Dict[str, Any]]:
        """Get all characters for a book"""
        return self.db.get_all_characters(book_id)
    
    # Chapter operations
    async def add_chapter(self, book_id: str, chapter_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add a chapter to a book"""
        return self.db.add_chapter(book_id, chapter_data)
    
    async def get_chapter(self, book_id: str, chapter_id: str) -> Optional[Dict[str, Any]]:
        """Get a chapter by ID"""
        return self.db.get_chapter(book_id, chapter_id)
    
    async def get_all_chapters(self, book_id: str) -> List[Dict[str, Any]]:
        """Get all chapters for a book"""
        return self.db.get_all_chapters(book_id)
    
    # Generation operations
    async def generate_book(self, book_id: str, use_enhanced: bool = True) -> str:
        """Generate a complete book"""
        # Ensure agent connector is initialized
        if self.agent_connector is None:
            await self.initialize()
            if self.agent_connector is None:
                raise RuntimeError("Failed to initialize agent connector")
        
        book = await self.get_book(book_id)
        if not book:
            raise ValueError(f"Book with ID {book_id} not found")
        
        # In a real implementation, this would be an async task
        # For now, we'll just use a placeholder
        try:
            book_config = {
                "title": book["title"],
                "genre": book["genre"],
                "characters": await self.get_all_characters(book_id),
                "chapters": await self.get_all_chapters(book_id)
            }
            
            result = await self.agent_connector.generate_book(book_config, use_enhanced=use_enhanced)
            return result
        except Exception as e:
            logger.error(f"Error generating book: {e}")
            raise
    
    async def generate_chapter(self, book_id: str, chapter_id: str) -> str:
        """Generate content for a chapter"""
        # Ensure agent connector is initialized
        if self.agent_connector is None:
            await self.initialize()
            if self.agent_connector is None:
                raise RuntimeError("Failed to initialize agent connector")
        
        book = await self.get_book(book_id)
        if not book:
            raise ValueError(f"Book with ID {book_id} not found")
        
        chapter = await self.get_chapter(book_id, chapter_id)
        if not chapter:
            raise ValueError(f"Chapter with ID {chapter_id} not found")
        
        # In a real implementation, this would be an async task
        # For now, we'll just use a placeholder
        try:
            chapter_config = {
                "book_id": book_id,
                "chapter_id": chapter_id,
                "title": chapter.get("title"),
                "outline": chapter.get("outline"),
                "book_title": book["title"],
                "book_genre": book["genre"],
                "characters": await self.get_all_characters(book_id)
            }
            
            result = await self.agent_connector.generate_chapter(book_id, chapter_config)
            
            # Update the chapter with the generated content
            updated_chapter = {
                "content": result,
                "word_count": len(result.split()),
                "status": "completed"
            }
            
            # Use the database to update the chapter
            self.db.update_chapter(book_id, chapter_id, updated_chapter)
            
            return result
        except Exception as e:
            logger.error(f"Error generating chapter: {e}")
            raise
    
    # Research operations
    async def research_topic(self, topic: str) -> Dict[str, Any]:
        """Research a topic for book writing"""
        # Ensure agent connector is initialized
        if self.agent_connector is None:
            await self.initialize()
            if self.agent_connector is None:
                raise RuntimeError("Failed to initialize agent connector")

        try:
            result = await self.agent_connector.research_topic(topic)
            return result
        except Exception as e:
            logger.error(f"Error researching topic: {e}")
            raise
            
    async def add_research(self, book_id: Optional[str], research_data: Dict[str, Any]) -> str:
        """Add research to a book or as standalone research"""
        # Create a copy of research_data to avoid modifying the original
        research_data_copy = dict(research_data)
        
        # If book_id is provided, add it to the research data
        if book_id:
            research_data_copy["book_id"] = book_id
            
        # Add the research to the database
        return self.db.add_research(book_id, research_data_copy)
        
    async def get_research(self, research_id: str) -> Optional[Dict[str, Any]]:
        """Get research by ID"""
        return self.db.get_research(research_id)
        
    async def get_all_research(self, book_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all research for a book or all general research if book_id is None"""
        if book_id is None:
            # Get all research entries that don't have a book_id
            all_research = []
            for book_id, research_list in self.db.research.items():
                if book_id == "general":
                    all_research.extend(research_list)
            return all_research
        
        return self.db.get_all_research(book_id)
        
    async def delete_research(self, research_id: str) -> bool:
        """Delete research"""
        return self.db.delete_research(research_id)
