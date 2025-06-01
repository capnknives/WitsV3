from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

# Book models
class BookBase(BaseModel):
    """Base book model with common attributes."""
    title: str
    author: str
    genre: str
    description: str

class BookCreate(BookBase):
    """Book creation model."""
    pass

class BookUpdate(BaseModel):
    """Book update model with optional fields."""
    title: Optional[str] = None
    author: Optional[str] = None
    genre: Optional[str] = None
    description: Optional[str] = None

class Book(BookBase):
    """Complete book model with all attributes."""
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Chapter models
class ChapterBase(BaseModel):
    """Base chapter model with common attributes."""
    book_id: UUID
    title: str
    order: int
    summary: str
    content: Optional[str] = None

class ChapterCreate(ChapterBase):
    """Chapter creation model."""
    pass

class ChapterUpdate(BaseModel):
    """Chapter update model with optional fields."""
    title: Optional[str] = None
    order: Optional[int] = None
    summary: Optional[str] = None
    content: Optional[str] = None

class Chapter(ChapterBase):
    """Complete chapter model with all attributes."""
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Character models
class CharacterBase(BaseModel):
    """Base character model with common attributes."""
    book_id: UUID
    name: str
    role: str
    description: str

class CharacterCreate(CharacterBase):
    """Character creation model."""
    pass

class CharacterUpdate(BaseModel):
    """Character update model with optional fields."""
    name: Optional[str] = None
    role: Optional[str] = None
    description: Optional[str] = None

class Character(CharacterBase):
    """Complete character model with all attributes."""
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Generation models
class GenerationRequest(BaseModel):
    """Generation request model."""
    book_id: UUID
    type: str = Field(..., description="Type of generation: chapter, character, plot, scene")
    prompt: str
    chapter_id: Optional[UUID] = None

class GenerationResponse(BaseModel):
    """Generation response model."""
    id: UUID
    book_id: UUID
    type: str
    prompt: str
    result: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# Research models
class ResearchRequest(BaseModel):
    """Research request model."""
    book_id: Optional[UUID] = None
    topic: str
    depth: str = Field(..., description="Research depth: basic, detailed, comprehensive")
    
    class Config:
        from_attributes = True
        
    @validator('depth')
    def validate_depth(cls, v):
        allowed_values = ['basic', 'detailed', 'comprehensive']
        if v not in allowed_values:
            raise ValueError(f"depth must be one of {allowed_values}")
        return v

class ResearchResponse(BaseModel):
    """Research response model."""
    id: UUID
    book_id: Optional[UUID] = None
    topic: str
    depth: str
    results: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True
