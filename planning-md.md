# WitsV3 Planning Document

## 🎯 Project Vision
WitsV3 is a streamlined, LLM-wrapper based AI orchestration system designed for maximum flexibility and LLM-driven decision making. It focuses on a CLI-first approach with a modular design to allow for future expansion.

## 🏗️ Architecture Overview

### Core Components
1. **LLM Interface** (`core/llm_interface.py`)
   - Primary: Ollama integration
   - Supports streaming responses
   - Embedding generation capability
   - Extensible to other providers (OpenAI, etc.)

2. **Agent System** (`agents/`)
   - **BaseAgent**: Abstract base for all agents
   - **WitsControlCenterAgent**: Entry point, parses intent and delegates
   - **LLMDrivenOrchestrator**: ReAct pattern implementation
   - All agents use async patterns

3. **Tool Registry** (`core/tool_registry.py`)
   - Dynamic tool registration
   - LLM-friendly descriptions
   - Async execution pattern
   - Built-in tools: think, calculator, file_ops, datetime

4. **Memory Management** (`core/memory_manager.py`)
   - Persistent storage with embeddings
   - Semantic search capabilities
   - Multiple backend support (basic, FAISS planned)
   - Automatic pruning

5. **Response Parsing** (`core/response_parser.py`)
   - Multiple format support (JSON, function, XML, markdown)
   - ReAct pattern parsing
   - Confidence scoring

## 🛠️ Technology Stack
- **Language**: Python 3.10+
- **Async Framework**: asyncio throughout
- **LLM**: Ollama (default: llama3)
- **Data Validation**: Pydantic v2
- **Configuration**: YAML-based
- **Testing**: pytest with async support
- **Optional**: FAISS for vector storage

## 📋 Design Principles
1. **LLM-First**: Let the LLM drive decision making
2. **Streaming**: Real-time feedback via StreamData
3. **Modular**: Easy to extend with new agents/tools
4. **Configurable**: YAML-driven configuration
5. **Robust**: Comprehensive error handling
6. **CLI-First**: Terminal as primary interface

## 🔧 Key Patterns

### ReAct Loop (Orchestrator)
```
Reason → Act → Observe → (Repeat until goal achieved)
```

### Agent Communication
- StreamData objects for real-time updates
- Types: thinking, action, observation, result, error

### Tool Integration
All tools must:
- Extend BaseTool
- Implement async execute()
- Provide LLM-friendly schema
- Handle errors gracefully

## 🚀 Future Expansion Areas
1. **Web UI**: Browser-based interface
2. **Advanced Book Writing System**: Specialized agent
3. **MCP Integration**: Model Context Protocol tools
4. **Langchain Bridge**: Integration with Langchain ecosystem
5. **FAISS Backends**: GPU-accelerated similarity search
6. **Multi-Model Support**: OpenAI, Anthropic, etc.

## 📏 Constraints & Guidelines
- Files must stay under 500 lines
- All I/O operations must be async
- Comprehensive error handling required
- Tests for all new features
- Backward compatibility important
- Unicode support throughout

## 🎨 Code Style
- PEP8 compliance
- Type hints everywhere
- Google-style docstrings
- Meaningful variable names
- Comments for complex logic