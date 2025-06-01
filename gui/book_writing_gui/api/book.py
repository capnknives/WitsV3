from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from uuid import UUID, uuid4
from datetime import datetime
import logging
import re

from ..services.book_writing_service import BookWritingService
from ..schemas.models import Book, BookCreate, BookUpdate

# Create router
router = APIRouter()

# Get service instance
book_service = BookWritingService()

@router.get("/", response_model=List[Book])
async def get_all_books():
    """
    Get all books.
    """
    return await book_service.get_all_books()

@router.get("/{book_id}", response_model=Book)
async def get_book(book_id: UUID):
    """
    Get a book by ID.
    """
    book = await book_service.get_book(str(book_id))
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID {book_id} not found"
        )
    return book

@router.post("/", response_model=Book, status_code=status.HTTP_201_CREATED)
async def create_book(book: BookCreate):
    """
    Create a new book.
    """
    try:
        logging.info(f"Creating book: {book}")
        created_book = await book_service.create_book(book.dict())
        logging.info(f"Book created successfully: {created_book}")
        return created_book
    except Exception as e:
        logging.error(f"Error creating book: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create book: {str(e)}"
        )

@router.put("/{book_id}", response_model=Book)
async def update_book(book_id: UUID, book_update: BookUpdate):
    """
    Update a book.
    """
    updated_book = await book_service.update_book(str(book_id), book_update.dict())
    if not updated_book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID {book_id} not found"
        )
    return updated_book

@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_book(book_id: UUID):
    """
    Delete a book.
    """
    success = await book_service.delete_book(str(book_id))
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID {book_id} not found"
        )
    return None

@router.post("/import-markdown", response_model=Book, status_code=status.HTTP_201_CREATED)
async def import_markdown(
    title: str = Form(...),
    author: str = Form(...),
    genre: str = Form(None),
    markdown_file: UploadFile = File(...)
):
    """
    Import a book from a markdown file.
    """
    try:
        # Read the markdown content
        content = await markdown_file.read()
        markdown_text = content.decode("utf-8")
        
        # Create the book
        book_data = {
            "title": title,
            "author": author,
            "genre": genre or "Fiction",
            "description": "Imported from markdown file"
        }
        
        logging.info(f"Creating book from markdown: {book_data}")
        created_book = await book_service.create_book(book_data)
        
        # Parse markdown into chapters
        # Simple parsing: Assume each heading level 1 or 2 starts a new chapter
        chapters = []
        current_chapter = {"title": "", "content": ""}
        chapter_number = 1
        
        for line in markdown_text.split('\n'):
            # Check if line is a heading
            if line.startswith('# ') or line.startswith('## '):
                # If we have content in the current chapter, save it
                if current_chapter["title"] and current_chapter["content"]:
                    chapters.append(current_chapter)
                    chapter_number += 1
                
                # Start a new chapter
                current_chapter = {
                    "title": line.replace('# ', '').replace('## ', '').strip(),
                    "content": "",
                    "order": chapter_number
                }
            else:
                # Add line to current chapter content
                if current_chapter["title"]:
                    current_chapter["content"] += line + "\n"
                else:
                    # If no chapter title yet, use a default
                    current_chapter["title"] = f"Chapter {chapter_number}"
                    current_chapter["content"] = line + "\n"
        
        # Add the last chapter if it has content
        if current_chapter["title"] and current_chapter["content"]:
            chapters.append(current_chapter)
        
        # Create chapters in the database
        for chapter in chapters:
            await book_service.add_chapter(str(created_book["id"]), chapter)
        
        logging.info(f"Successfully imported {len(chapters)} chapters from markdown")
        return created_book
    except Exception as e:
        logging.error(f"Error importing markdown: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import markdown: {str(e)}"
        )
