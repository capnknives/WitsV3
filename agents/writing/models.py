"""
Data models for book writing agent
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


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


@dataclass 
class WritingSession:
    """Represents a writing session with metadata"""
    session_id: str
    book_id: Optional[str] = None
    start_time: str = ""
    end_time: Optional[str] = None
    words_written: int = 0
    chapters_worked_on: List[str] = field(default_factory=list)
    revisions_made: int = 0
    research_performed: bool = False
    
    
@dataclass
class CharacterProfile:
    """Represents a character in fiction writing"""
    name: str
    role: str  # protagonist, antagonist, supporting, etc.
    description: str
    backstory: str = ""
    motivations: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)  # character_name -> relationship_type
    arc: str = ""  # character development arc
    
    
@dataclass
class WorldElement:
    """Represents an element of world-building"""
    name: str
    category: str  # location, culture, technology, magic_system, etc.
    description: str
    rules: List[str] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)  # connected to other elements
    importance: float = 0.5  # 0-1 scale


# Constants for book writing
SUPPORTED_GENRES = [
    "fiction", "non-fiction", "technical", "academic", 
    "biography", "mystery", "sci-fi", "fantasy", "romance",
    "thriller", "horror", "historical", "self-help",
    "poetry", "children", "young-adult"
]

WRITING_STYLES = [
    "narrative", "expository", "descriptive", "persuasive",
    "academic", "conversational", "formal", "creative",
    "journalistic", "poetic", "technical", "instructional"
]

NARRATIVE_STRUCTURES = [
    "linear", "non-linear", "circular", "parallel",
    "frame_story", "epistolary", "stream_of_consciousness",
    "in_medias_res", "reverse_chronological"
]

CHAPTER_STATUS = [
    "planned",      # Initial planning stage
    "outlined",     # Detailed outline created
    "drafted",      # First draft written
    "revised",      # Undergone revision
    "edited",       # Copy edited
    "complete"      # Final version
]