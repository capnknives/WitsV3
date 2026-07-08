# agents/book_writing_models.py
"""Data models for the book writing agent."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BookStructure:
    """Represents the structure of a book."""

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
    """Represents a book chapter."""

    id: str
    title: str
    outline: str
    content: str = ""
    word_count: int = 0
    status: str = "planned"  # planned, outlined, drafted, revised, complete
    dependencies: List[str] = field(default_factory=list)
    research_requirements: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.research_requirements is None:
            self.research_requirements = []
