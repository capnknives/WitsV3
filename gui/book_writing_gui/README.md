# WitsV3 Book Writing GUI

A FastAPI-based GUI for the WitsV3 Book Writing Agent. This GUI provides a web interface for interacting with the Book Writing Agent, allowing users to create and manage books, characters, chapters, and research.

## Features

- Book management (create, read, update, delete)
- Character management (create, read, update, delete)
- Chapter management (create, read, update, delete)
- Research capabilities for book topics
- Content generation for chapters and books
- Real-time updates via WebSockets

## Project Structure

```
book_writing_gui/
├── api/                  # API endpoints
│   ├── book.py           # Book API
│   ├── chapter.py        # Chapter API
│   ├── character.py      # Character API
│   ├── generation.py     # Content generation API
│   └── research.py       # Research API
├── models/               # Database models
│   └── database.py       # Database connection and models
├── schemas/              # Pydantic schemas
│   └── models.py         # Data models
├── services/             # Business logic
│   ├── agent_connector.py # Connector to Book Writing Agent
│   └── book_writing_service.py # Book writing service
├── static/               # Static files
│   ├── css/              # CSS files
│   └── js/               # JavaScript files
├── templates/            # HTML templates
│   └── index.html        # Main page
├── utils/                # Utility functions
│   └── websocket_manager.py # WebSocket manager
├── main.py               # FastAPI application
└── run.py                # Entry point
```

## Requirements

- Python 3.10+
- FastAPI
- Uvicorn
- Pydantic
- WitsV3 (for agent integration)

## Installation

1. Make sure you have WitsV3 installed and configured
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

To start the Book Writing GUI:

```bash
cd gui/book_writing_gui
python run.py
```

This will start the FastAPI server on http://localhost:8000

## API Documentation

Once the server is running, you can access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

The Book Writing GUI is designed to be modular and extensible. To add new features:

1. Add new API endpoints in the `api/` directory
2. Update the schemas in `schemas/models.py`
3. Add business logic in the `services/` directory
4. Update the frontend in `static/` and `templates/`

## License

This project is licensed under the same license as WitsV3.
