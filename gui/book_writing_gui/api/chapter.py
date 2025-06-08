from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Optional
from uuid import UUID

from ..schemas.models import Chapter, ChapterCreate, ChapterUpdate
from ..services.book_writing_service import BookWritingService

router = APIRouter()
book_service = BookWritingService()

@router.get("/book/{book_id}", response_model=List[Chapter])
async def get_chapters(book_id: UUID):
    """
    Get all chapters for a book.
    """
    chapters = await book_service.get_all_chapters(str(book_id))
    return chapters

@router.get("/{chapter_id}", response_model=Chapter)
async def get_chapter(chapter_id: UUID, book_id: UUID):
    """
    Get a chapter by ID.
    """
    chapter = await book_service.get_chapter(str(book_id), str(chapter_id))
    if not chapter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chapter with ID {chapter_id} not found"
        )
    return chapter

@router.post("/", response_model=Chapter, status_code=status.HTTP_201_CREATED)
async def create_chapter(chapter: ChapterCreate):
    """
    Create a new chapter.
    """
    return await book_service.add_chapter(str(chapter.book_id), chapter.dict())

@router.put("/{chapter_id}", response_model=Chapter)
async def update_chapter(chapter_id: UUID, chapter_update: ChapterUpdate, book_id: UUID):
    """
    Update a chapter.
    """
    chapter = await book_service.get_chapter(str(book_id), str(chapter_id))
    if not chapter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chapter with ID {chapter_id} not found"
        )
    
    # Update the chapter
    for key, value in chapter_update.dict(exclude_unset=True).items():
        chapter[key] = value
    
    # Save the updated chapter
    for i, ch in enumerate(book_service.chapters[str(book_id)]):
        if ch["id"] == str(chapter_id):
            book_service.chapters[str(book_id)][i] = chapter
            break
    
    return chapter

@router.delete("/{chapter_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chapter(chapter_id: UUID, book_id: UUID):
    """
    Delete a chapter.
    """
    chapter = await book_service.get_chapter(str(book_id), str(chapter_id))
    if not chapter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chapter with ID {chapter_id} not found"
        )
    
    # Remove the chapter
    book_service.chapters[str(book_id)] = [
        ch for ch in book_service.chapters[str(book_id)] 
        if ch["id"] != str(chapter_id)
    ]
    
    # Also remove from book
    book = await book_service.get_book(str(book_id))
    if book:
        book["chapters"] = [
            ch for ch in book["chapters"] 
            if ch["id"] != str(chapter_id)
        ]
    
    return None

@router.post("/{chapter_id}/generate", response_model=str)
async def generate_chapter_content(chapter_id: UUID, book_id: UUID):
    """
    Generate content for a chapter.
    """
    try:
        content = await book_service.generate_chapter(str(book_id), str(chapter_id))
        return content
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating chapter content: {str(e)}"
        )
