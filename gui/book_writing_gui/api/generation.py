from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
from uuid import UUID

from ..schemas.models import GenerationRequest, GenerationResponse
from ..services.book_writing_service import BookWritingService

router = APIRouter()
book_service = BookWritingService()

@router.post("/", response_model=GenerationResponse)
async def generate_content(request: GenerationRequest):
    """
    Generate content based on the request type.
    """
    try:
        if request.type == "chapter":
            if not request.chapter_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Chapter ID is required for chapter generation"
                )
            
            content = await book_service.generate_chapter(
                str(request.book_id), 
                str(request.chapter_id)
            )
            
            return {
                "id": UUID(),
                "book_id": request.book_id,
                "type": request.type,
                "prompt": request.prompt,
                "result": content,
                "created_at": None  # Will be set by Pydantic
            }
        
        elif request.type == "book":
            # Generate a complete book
            book = await book_service.get_book(str(request.book_id))
            if not book:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Book with ID {request.book_id} not found"
                )
            
            content = await book_service.generate_book(str(request.book_id))
            
            return {
                "id": UUID(),
                "book_id": request.book_id,
                "type": request.type,
                "prompt": request.prompt,
                "result": "Book generation started. This will take some time.",
                "created_at": None  # Will be set by Pydantic
            }
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported generation type: {request.type}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating content: {str(e)}"
        )
