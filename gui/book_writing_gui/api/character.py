from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Optional
from uuid import UUID

from ..schemas.models import Character, CharacterCreate, CharacterUpdate
from ..services.book_writing_service import BookWritingService

router = APIRouter()
book_service = BookWritingService()

@router.get("/book/{book_id}", response_model=List[Character])
async def get_characters(book_id: UUID):
    """
    Get all characters for a book.
    """
    characters = await book_service.get_all_characters(str(book_id))
    return characters

@router.get("/{character_id}", response_model=Character)
async def get_character(character_id: UUID, book_id: UUID):
    """
    Get a character by ID.
    """
    character = await book_service.get_character(str(book_id), str(character_id))
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character with ID {character_id} not found"
        )
    return character

@router.post("/", response_model=Character, status_code=status.HTTP_201_CREATED)
async def create_character(character: CharacterCreate):
    """
    Create a new character.
    """
    return await book_service.add_character(str(character.book_id), character.dict())

@router.put("/{character_id}", response_model=Character)
async def update_character(character_id: UUID, character_update: CharacterUpdate, book_id: UUID):
    """
    Update a character.
    """
    character = await book_service.get_character(str(book_id), str(character_id))
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character with ID {character_id} not found"
        )
    
    # Update the character
    for key, value in character_update.dict(exclude_unset=True).items():
        character[key] = value
    
    # Save the updated character
    for i, ch in enumerate(book_service.characters[str(book_id)]):
        if ch["id"] == str(character_id):
            book_service.characters[str(book_id)][i] = character
            break
    
    # Also update in book
    book = await book_service.get_book(str(book_id))
    if book:
        for i, ch in enumerate(book["characters"]):
            if ch["id"] == str(character_id):
                book["characters"][i] = character
                break
    
    return character

@router.delete("/{character_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_character(character_id: UUID, book_id: UUID):
    """
    Delete a character.
    """
    character = await book_service.get_character(str(book_id), str(character_id))
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character with ID {character_id} not found"
        )
    
    # Remove the character
    book_service.characters[str(book_id)] = [
        ch for ch in book_service.characters[str(book_id)] 
        if ch["id"] != str(character_id)
    ]
    
    # Also remove from book
    book = await book_service.get_book(str(book_id))
    if book:
        book["characters"] = [
            ch for ch in book["characters"] 
            if ch["id"] != str(character_id)
        ]
    
    return None

@router.post("/generate", response_model=Character)
async def generate_character(book_id: UUID, name: str, role: str):
    """
    Generate a character for a book.
    """
    try:
        # In a real implementation, this would call the agent to generate a character
        # For now, we'll just create a basic character
        character_data = {
            "name": name,
            "role": role,
            "description": f"A {role} named {name}. This character was auto-generated."
        }
        
        character = await book_service.add_character(str(book_id), character_data)
        return character
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating character: {str(e)}"
        )
