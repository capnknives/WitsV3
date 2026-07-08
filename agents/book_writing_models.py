# agents/book_writing_models.py
"""Data models for the book writing agent."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BookStructure:
    """Represents the structure of a book."""

    title: str
    subtitle: str | None = None
    genre: str = "non-fiction"
    target_length: int = 50000  # words
    chapters: list[dict[str, Any]] = field(default_factory=list)
    style_guide: dict[str, Any] = field(default_factory=dict)
    research_notes: list[str] = field(default_factory=list)
    # Persistent story-bible text (characters, themes, POV, established
    # rules/lore) injected into every outline/chapter generation prompt for
    # this book so long-form content stays consistent chapter to chapter.
    world_bible: str = ""

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
    dependencies: list[str] = field(default_factory=list)
    research_requirements: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.research_requirements is None:
            self.research_requirements = []
