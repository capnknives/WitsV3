import logging
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class Database:
    """Simple file-based database for the book writing GUI"""
    
    def __init__(self, data_dir: str = "data"):
        # Use absolute path based on the application directory
        base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = os.path.join(base_dir, data_dir)
        logger.info(f"Database using data directory: {self.data_dir}")
        
        self.books_file = os.path.join(self.data_dir, "books.json")
        self.characters_file = os.path.join(self.data_dir, "characters.json")
        self.chapters_file = os.path.join(self.data_dir, "chapters.json")
        self.research_file = os.path.join(self.data_dir, "research.json")
        
        self.books = {}
        self.characters = {}
        self.chapters = {}
        self.research = {}
        
        self._ensure_data_dir()
        self._load_data()
    
    def _ensure_data_dir(self):
        """Ensure the data directory exists"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _load_data(self):
        """Load data from files"""
        try:
            if os.path.exists(self.books_file):
                with open(self.books_file, 'r') as f:
                    self.books = json.load(f)
            
            if os.path.exists(self.characters_file):
                with open(self.characters_file, 'r') as f:
                    self.characters = json.load(f)
            
            if os.path.exists(self.chapters_file):
                with open(self.chapters_file, 'r') as f:
                    self.chapters = json.load(f)
            
            if os.path.exists(self.research_file):
                with open(self.research_file, 'r') as f:
                    self.research = json.load(f)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save data to files"""
        try:
            with open(self.books_file, 'w') as f:
                json.dump(self.books, f, indent=2)
            
            with open(self.characters_file, 'w') as f:
                json.dump(self.characters, f, indent=2)
            
            with open(self.chapters_file, 'w') as f:
                json.dump(self.chapters, f, indent=2)
            
            with open(self.research_file, 'w') as f:
                json.dump(self.research, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    # Book operations
    def create_book(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new book"""
        book_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        new_book = {
            "id": book_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "characters": [],
            "chapters": [],
            **book_data
        }
        
        self.books[book_id] = new_book
        self._save_data()
        return new_book
    
    def get_book(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Get a book by ID"""
        return self.books.get(book_id)
    
    def get_all_books(self) -> List[Dict[str, Any]]:
        """Get all books"""
        return list(self.books.values())
    
    def update_book(self, book_id: str, book_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a book"""
        if book_id not in self.books:
            return None
        
        book = self.books[book_id]
        update_data = {k: v for k, v in book_data.items() if k not in ["characters", "chapters"]}
        
        book.update(update_data)
        book["updated_at"] = datetime.utcnow().isoformat()
        
        self._save_data()
        return book
    
    def delete_book(self, book_id: str) -> bool:
        """Delete a book"""
        if book_id not in self.books:
            return False
        
        del self.books[book_id]
        
        # Also delete associated characters and chapters
        if book_id in self.characters:
            del self.characters[book_id]
        
        if book_id in self.chapters:
            del self.chapters[book_id]
        
        self._save_data()
        return True
    
    # Character operations
    def add_character(self, book_id: str, character_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add a character to a book"""
        if book_id not in self.books:
            return None
        
        character_id = str(uuid.uuid4())
        
        new_character = {
            "id": character_id,
            "book_id": book_id,
            **character_data
        }
        
        if book_id not in self.characters:
            self.characters[book_id] = []
        
        self.characters[book_id].append(new_character)
        
        # Also add to the book's characters list
        character_ref = {
            "id": character_id,
            "name": character_data.get("name", "Unnamed Character")
        }
        self.books[book_id]["characters"].append(character_ref)
        self.books[book_id]["updated_at"] = datetime.utcnow().isoformat()
        
        self._save_data()
        return new_character
    
    def get_character(self, book_id: str, character_id: str) -> Optional[Dict[str, Any]]:
        """Get a character by ID"""
        if book_id not in self.characters:
            return None
        
        for character in self.characters[book_id]:
            if character["id"] == character_id:
                return character
        
        return None
    
    def get_all_characters(self, book_id: str) -> List[Dict[str, Any]]:
        """Get all characters for a book"""
        if book_id not in self.characters:
            return []
        
        return self.characters[book_id]
    
    def update_character(self, book_id: str, character_id: str, character_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a character"""
        if book_id not in self.characters:
            return None
        
        for i, character in enumerate(self.characters[book_id]):
            if character["id"] == character_id:
                self.characters[book_id][i].update(character_data)
                
                # Also update the character reference in the book
                for j, char_ref in enumerate(self.books[book_id]["characters"]):
                    if char_ref["id"] == character_id:
                        self.books[book_id]["characters"][j]["name"] = character_data.get("name", char_ref["name"])
                        break
                
                self.books[book_id]["updated_at"] = datetime.utcnow().isoformat()
                
                self._save_data()
                return self.characters[book_id][i]
        
        return None
    
    def delete_character(self, book_id: str, character_id: str) -> bool:
        """Delete a character"""
        if book_id not in self.characters:
            return False
        
        for i, character in enumerate(self.characters[book_id]):
            if character["id"] == character_id:
                del self.characters[book_id][i]
                
                # Also remove from the book's characters list
                for j, char_ref in enumerate(self.books[book_id]["characters"]):
                    if char_ref["id"] == character_id:
                        del self.books[book_id]["characters"][j]
                        break
                
                self.books[book_id]["updated_at"] = datetime.utcnow().isoformat()
                
                self._save_data()
                return True
        
        return False
    
    # Chapter operations
    def add_chapter(self, book_id: str, chapter_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add a chapter to a book"""
        if book_id not in self.books:
            return None
        
        chapter_id = str(uuid.uuid4())
        
        new_chapter = {
            "id": chapter_id,
            "book_id": book_id,
            "content": None,
            "word_count": 0,
            "status": "draft",
            **chapter_data
        }
        
        if book_id not in self.chapters:
            self.chapters[book_id] = []
        
        self.chapters[book_id].append(new_chapter)
        
        # Also add to the book's chapters list
        chapter_ref = {
            "id": chapter_id,
            "title": chapter_data.get("title", f"Chapter {len(self.books[book_id]['chapters']) + 1}"),
            "order": chapter_data.get("order", len(self.books[book_id]["chapters"]) + 1)
        }
        self.books[book_id]["chapters"].append(chapter_ref)
        self.books[book_id]["updated_at"] = datetime.utcnow().isoformat()
        
        self._save_data()
        return new_chapter
    
    def get_chapter(self, book_id: str, chapter_id: str) -> Optional[Dict[str, Any]]:
        """Get a chapter by ID"""
        if book_id not in self.chapters:
            return None
        
        for chapter in self.chapters[book_id]:
            if chapter["id"] == chapter_id:
                return chapter
        
        return None
    
    def get_all_chapters(self, book_id: str) -> List[Dict[str, Any]]:
        """Get all chapters for a book"""
        if book_id not in self.chapters:
            return []
        
        return sorted(self.chapters[book_id], key=lambda x: x.get("order", 0))
    
    def update_chapter(self, book_id: str, chapter_id: str, chapter_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a chapter"""
        if book_id not in self.chapters:
            return None
        
        for i, chapter in enumerate(self.chapters[book_id]):
            if chapter["id"] == chapter_id:
                self.chapters[book_id][i].update(chapter_data)
                
                # Also update the chapter reference in the book
                for j, chap_ref in enumerate(self.books[book_id]["chapters"]):
                    if chap_ref["id"] == chapter_id:
                        if "title" in chapter_data:
                            self.books[book_id]["chapters"][j]["title"] = chapter_data["title"]
                        if "order" in chapter_data:
                            self.books[book_id]["chapters"][j]["order"] = chapter_data["order"]
                        break
                
                self.books[book_id]["updated_at"] = datetime.utcnow().isoformat()
                
                self._save_data()
                return self.chapters[book_id][i]
        
        return None
    
    def delete_chapter(self, book_id: str, chapter_id: str) -> bool:
        """Delete a chapter"""
        if book_id not in self.chapters:
            return False
        
        for i, chapter in enumerate(self.chapters[book_id]):
            if chapter["id"] == chapter_id:
                del self.chapters[book_id][i]
                
                # Also remove from the book's chapters list
                for j, chap_ref in enumerate(self.books[book_id]["chapters"]):
                    if chap_ref["id"] == chapter_id:
                        del self.books[book_id]["chapters"][j]
                        break
                
                self.books[book_id]["updated_at"] = datetime.utcnow().isoformat()
                
                self._save_data()
                return True
        
        return False
    
    # Research operations
    def add_research(self, book_id: Optional[str], research_data: Dict[str, Any]) -> str:
        """Add a research entry, optionally associated with a book"""
        research_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        new_research = {
            "id": research_id,
            "created_at": timestamp,
            **research_data
        }
        
        # Store research by book_id for easier retrieval
        if book_id:
            if book_id not in self.research:
                self.research[book_id] = []
            self.research[book_id].append(new_research)
        else:
            # Store general research not associated with any book
            if "general" not in self.research:
                self.research["general"] = []
            self.research["general"].append(new_research)
        
        self._save_data()
        return research_id
    
    def get_research(self, research_id: str) -> Optional[Dict[str, Any]]:
        """Get a research entry by ID"""
        for book_id, research_list in self.research.items():
            for research in research_list:
                if research["id"] == research_id:
                    return research
        return None
    
    def get_all_research(self, book_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all research entries for a book or all general research"""
        if book_id is None:
            # Return general research
            return self.research.get("general", [])
        
        # Return research for a specific book
        return self.research.get(book_id, [])
    
    def delete_research(self, research_id: str) -> bool:
        """Delete a research entry"""
        for book_id, research_list in self.research.items():
            for i, research in enumerate(research_list):
                if research["id"] == research_id:
                    del self.research[book_id][i]
                    self._save_data()
                    return True
        return False
