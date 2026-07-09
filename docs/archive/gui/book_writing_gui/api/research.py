from fastapi import APIRouter, HTTPException, status, Request
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime
import logging

from ..schemas.models import ResearchRequest, ResearchResponse
from ..services.book_writing_service import BookWritingService
from ..services.agent_connector import BookWritingAgentConnector

router = APIRouter()
book_service = BookWritingService()
logger = logging.getLogger(__name__)

@router.post("/", response_model=ResearchResponse)
async def research_topic(request: ResearchRequest, raw_request: Request):
    """
    Research a topic for book writing.
    """
    # Log the request payload for debugging
    body = await raw_request.json()
    logger.info(f"Research request received: {body}")
    
    try:
        # Validate book_id if provided
        if request.book_id:
            book = await book_service.get_book(str(request.book_id))
            if not book:
                logger.error(f"Book with ID {request.book_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Book with ID {request.book_id} not found"
                )
        
        # Connect to the agent
        logger.info(f"Connecting to agent for research on topic: {request.topic}")
        async with BookWritingAgentConnector() as agent:
            research_response = await agent.research_topic(request.topic, request.depth)
            logger.info(f"Research completed for topic: {request.topic}")

        # Check if there was an error
        if isinstance(research_response, dict) and "error" in research_response:
            error_message = research_response["error"]
            logger.warning(f"Research error: {error_message}")
            # Create a fallback research result with the error message
            research_results = [f"Research could not be completed: {error_message}", 
                               "Please try again later or check if the research service is running."]
        else:
            # Process successful research results
            if isinstance(research_response, dict) and "content" in research_response:
                # Extract content from the response
                content = research_response.get("content", "")
                # Split content into paragraphs for better display
                research_results = [p.strip() for p in content.split("\n\n") if p.strip()]
                if not research_results:
                    research_results = [content]  # Use the whole content if splitting didn't work
            elif isinstance(research_response, list):
                research_results = research_response
            else:
                # Convert to string and then to list if it's neither dict nor list
                research_results = [str(research_response)]

        # Store the research results
        research_data = {
            "topic": request.topic,
            "depth": request.depth,
            "results": research_results
        }
        
        # Store the research results
        book_id = str(request.book_id) if request.book_id else None
        logger.info(f"Storing research results for topic: {request.topic}")
        research_id = await book_service.add_research(
            book_id,
            research_data
        )

        response_data = {
            "id": research_id,
            "topic": request.topic,
            "depth": request.depth,
            "results": research_results,
            "created_at": datetime.now()
        }
        
        # Add book_id to response if provided
        if request.book_id:
            response_data["book_id"] = request.book_id
            
        logger.info(f"Research stored successfully with ID: {research_id}")
        return response_data
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error researching topic: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error researching topic: {str(e)}"
        )

@router.get("/general", response_model=List[ResearchResponse])
async def get_general_research():
    """
    Get all general research not associated with any book.
    """
    logger.info("Getting all general research")
    try:
        research_entries = await book_service.get_all_research(None)
        logger.info(f"Found {len(research_entries)} general research entries")
        
        # Process each research entry to ensure it has the correct format
        processed_entries = []
        for entry in research_entries:
            # Check if the entry has an error field instead of results
            if "error" in entry and "results" not in entry:
                # Convert error to a list of results
                entry["results"] = [
                    f"Research could not be completed: {entry['error']}", 
                    "Please try again later or check if the research service is running."
                ]
                # Remove the error field to avoid confusion
                entry.pop("error", None)
            
            # Ensure created_at is a valid datetime
            if "created_at" not in entry or entry["created_at"] is None:
                entry["created_at"] = datetime.now()
            
            # Ensure results is a list
            if "results" not in entry or not isinstance(entry["results"], list):
                if "results" in entry and not isinstance(entry["results"], list):
                    # Convert non-list results to a list
                    entry["results"] = [str(entry["results"])]
                else:
                    # Set default results if missing
                    entry["results"] = ["No research results available"]
            
            processed_entries.append(entry)
        
        return processed_entries
    except Exception as e:
        logger.error(f"Error getting general research: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting general research: {str(e)}"
        )

@router.get("/book/{book_id}", response_model=List[ResearchResponse])
async def get_research_for_book(book_id: UUID):
    """
    Get all research for a book.
    """
    logger.info(f"Getting all research for book: {book_id}")
    try:
        # Validate book exists
        book = await book_service.get_book(str(book_id))
        if not book:
            logger.error(f"Book with ID {book_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Book with ID {book_id} not found"
            )
            
        research_entries = await book_service.get_all_research(str(book_id))
        logger.info(f"Found {len(research_entries)} research entries for book: {book_id}")
        
        # Process each research entry to ensure it has the correct format
        processed_entries = []
        for entry in research_entries:
            # Check if the entry has an error field instead of results
            if "error" in entry and "results" not in entry:
                # Convert error to a list of results
                entry["results"] = [
                    f"Research could not be completed: {entry['error']}", 
                    "Please try again later or check if the research service is running."
                ]
                # Remove the error field to avoid confusion
                entry.pop("error", None)
            
            # Ensure created_at is a valid datetime
            if "created_at" not in entry or entry["created_at"] is None:
                entry["created_at"] = datetime.now()
            
            # Ensure results is a list
            if "results" not in entry or not isinstance(entry["results"], list):
                if "results" in entry and not isinstance(entry["results"], list):
                    # Convert non-list results to a list
                    entry["results"] = [str(entry["results"])]
                else:
                    # Set default results if missing
                    entry["results"] = ["No research results available"]
            
            processed_entries.append(entry)
        
        return processed_entries
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting research for book {book_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting research: {str(e)}"
        )

@router.get("/{research_id}", response_model=ResearchResponse)
async def get_research(research_id: UUID):
    """
    Get research by ID.
    """
    logger.info(f"Getting research with ID: {research_id}")
    try:
        entry = await book_service.get_research(str(research_id))
        if not entry:
            logger.error(f"Research with ID {research_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research with ID {research_id} not found"
            )
        
        # Process the research entry to ensure it has the correct format
        # Check if the entry has an error field instead of results
        if "error" in entry and "results" not in entry:
            # Convert error to a list of results
            entry["results"] = [
                f"Research could not be completed: {entry['error']}", 
                "Please try again later or check if the research service is running."
            ]
            # Remove the error field to avoid confusion
            entry.pop("error", None)
        
        # Ensure created_at is a valid datetime
        if "created_at" not in entry or entry["created_at"] is None:
            entry["created_at"] = datetime.now()
        
        # Ensure results is a list
        if "results" not in entry or not isinstance(entry["results"], list):
            if "results" in entry and not isinstance(entry["results"], list):
                # Convert non-list results to a list
                entry["results"] = [str(entry["results"])]
            else:
                # Set default results if missing
                entry["results"] = ["No research results available"]
        
        logger.info(f"Found research with ID: {research_id}")
        return entry
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting research {research_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting research: {str(e)}"
        )

@router.delete("/{research_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_research(research_id: UUID):
    """
    Delete research.
    """
    logger.info(f"Deleting research with ID: {research_id}")
    try:
        research = await book_service.get_research(str(research_id))
        if not research:
            logger.error(f"Research with ID {research_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research with ID {research_id} not found"
            )

        success = await book_service.delete_research(str(research_id))
        if not success:
            logger.error(f"Failed to delete research with ID {research_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete research with ID {research_id}"
            )
            
        logger.info(f"Successfully deleted research with ID: {research_id}")
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting research {research_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting research: {str(e)}"
        )
