import asyncio
import json
import logging
from typing import Dict, List, Any
from fastapi import WebSocket

class WebSocketManager:
    """
    WebSocket connection manager for handling real-time communication
    between the FastAPI backend and the frontend.
    """
    
    def __init__(self):
        """Initialize the WebSocket manager with an empty connections list."""
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger(__name__)
        
    async def connect(self, websocket: WebSocket):
        """
        Connect a new WebSocket client.
        
        Args:
            websocket: The WebSocket connection to add
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f'WebSocket client connected. Total connections: {len(self.active_connections)}')
        
    async def disconnect(self, websocket: WebSocket):
        """
        Disconnect a WebSocket client.
        
        Args:
            websocket: The WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f'WebSocket client disconnected. Remaining connections: {len(self.active_connections)}')
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """
        Send a message to a specific client.
        
        Args:
            message: The message to send
            websocket: The WebSocket connection to send to
        """
        if isinstance(message, dict) or isinstance(message, list):
            message = json.dumps(message)
        await websocket.send_text(message)
    
    async def broadcast(self, message: Any):
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: The message to broadcast
        """
        if isinstance(message, dict) or isinstance(message, list):
            message = json.dumps(message)
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                self.logger.error(f'Error sending message to client: {e}')
                disconnected.append(connection)
        
        # Clean up any disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)
    
    async def broadcast_json(self, data: Dict):
        """
        Broadcast a JSON message to all connected clients.
        
        Args:
            data: The JSON data to broadcast
        """
        await self.broadcast(json.dumps(data))
    
    async def send_book_update(self, book_data: Dict):
        """
        Send a book update to all connected clients.
        
        Args:
            book_data: The book data to send
        """
        await self.broadcast_json({
            'type': 'book_update',
            'book': book_data
        })
    
    async def send_chapter_update(self, chapter_data: Dict):
        """
        Send a chapter update to all connected clients.
        
        Args:
            chapter_data: The chapter data to send
        """
        await self.broadcast_json({
            'type': 'chapter_update',
            'chapter': chapter_data
        })
    
    async def send_character_update(self, character_data: Dict):
        """
        Send a character update to all connected clients.
        
        Args:
            character_data: The character data to send
        """
        await self.broadcast_json({
            'type': 'character_update',
            'character': character_data
        })
    
    async def send_generation_progress(self, progress: int):
        """
        Send a generation progress update to all connected clients.
        
        Args:
            progress: The generation progress percentage (0-100)
        """
        await self.broadcast_json({
            'type': 'generation_progress',
            'progress': progress
        })
    
    async def send_research_result(self, research_data: Dict):
        """
        Send a research result to all connected clients.
        
        Args:
            research_data: The research data to send
        """
        await self.broadcast_json({
            'type': 'research_result',
            'research': research_data
        })
    
    async def send_error(self, message: str):
        """
        Send an error message to all connected clients.
        
        Args:
            message: The error message to send
        """
        await self.broadcast_json({
            'type': 'error',
            'message': message
        })
