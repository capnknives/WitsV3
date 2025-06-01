from fastapi import FastAPI, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import WitsV3 modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Use absolute imports
from gui.book_writing_gui.api import book, character, chapter, generation, research
from gui.book_writing_gui.services.book_writing_service import BookWritingService
from gui.book_writing_gui.utils.websocket_manager import WebSocketManager

# Import the encoding utility function directly
from gui.book_writing_gui.utils.encoding_utils import ensure_template_encoding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent

# Create FastAPI app
app = FastAPI(title='Book Writing GUI')

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Mount static files
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'static')), name='static')

# Configure templates
templates = Jinja2Templates(directory=str(BASE_DIR / 'templates'))

# Initialize services
book_service = BookWritingService()
websocket_manager = WebSocketManager()

# Include routers
app.include_router(book.router, prefix='/api/books', tags=['books'])
app.include_router(character.router, prefix='/api/characters', tags=['characters'])
app.include_router(chapter.router, prefix='/api/chapters', tags=['chapters'])
app.include_router(generation.router, prefix='/api/generation', tags=['generation'])
app.include_router(research.router, prefix='/api/research', tags=['research'])

@app.on_event('startup')
async def startup_event():
    """Initialize services on startup"""
    # Fix template file encodings to prevent UnicodeDecodeError
    fixed_count = ensure_template_encoding()
    if fixed_count > 0:
        logging.info(f'Fixed encoding for {fixed_count} template files')
    
    await book_service.initialize()
    logging.info('Book Writing GUI started')

@app.get('/')
async def index(request: Request):
    """Render the index page"""
    return templates.TemplateResponse('index.html', {'request': request})

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket_manager.broadcast(data)
    except Exception as e:
        logging.error(f'WebSocket error: {e}')
    finally:
        await websocket_manager.disconnect(websocket)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
