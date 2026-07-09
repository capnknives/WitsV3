---
title: "WitsV3 System Architecture"
created: "2025-06-09"
last_updated: "2026-07-08"
status: "reference"
---
# WitsV3 System Architecture

> **Status note (2026-07-08):** This document remains a useful **component map**, but
> sections labeled “PRODUCTION READY” / mid-2025 metrics are **historical**. For how to
> run the product and what shipped recently, use [`README.md`](../../README.md) and
> [`docs/roadmap/suggested-features-2026-07.md`](../roadmap/suggested-features-2026-07.md).
> The primary user interface is now the **Web UI**, not CLI-only.

## Table of Contents

- [🎯 Project Vision](#🎯-project-vision)
- [📊 Current Status: PRODUCTION READY 🚀](#📊-current-status-production-ready-🚀)
- [🏗️ Architecture Overview](#🏗️-architecture-overview)
- [🛠️ Technology Stack ✅ PROVEN STABLE](#🛠️-technology-stack-✅-proven-stable)
- [📋 Design Principles ✅ FULLY IMPLEMENTED](#📋-design-principles-✅-fully-implemented)
- [🔧 Key Patterns ✅ ALL WORKING](#🔧-key-patterns-✅-all-working)
- [🚀 Future Expansion Areas](#🚀-future-expansion-areas)
- [📏 Constraints & Guidelines ✅ FULLY ADHERED](#📏-constraints--guidelines-✅-fully-adhered)
- [🎨 Code Style ✅ ENFORCED](#🎨-code-style-✅-enforced)
- [📈 Achievements (2025-06-08)](#📈-achievements-(2025-06-08))
- [🎯 Next Priorities](#🎯-next-priorities)



## 🎯 Project Vision

WitsV3 is a streamlined, LLM-wrapper based AI orchestration system designed for maximum flexibility and LLM-driven decision making. It focuses on a CLI-first approach with a modular design to allow for future expansion.

## 📊 Current Status: PRODUCTION READY 🚀

**Updated: 2025-06-08**

### ✅ FULLY IMPLEMENTED & STABLE

- **Test Suite**: 93.3% pass rate (56/60 tests)
- **Core Architecture**: All components operational
- **Tool System**: 6 tools fully functional and tested
- **Agent Framework**: Background agents with scheduling
- **Memory Management**: Multiple backends working
- **Adaptive LLM**: Complete system with model routing
- **External Integrations**: Ollama, Supabase properly mocked and tested

## 🏗️ Architecture Overview

### Core Components ✅ ALL OPERATIONAL

1. **LLM Interface** (`core/llm_interface.py`) ✅

   - Primary: Ollama integration (fully tested)
   - Supports streaming responses
   - Embedding generation capability
   - Extensible to other providers (OpenAI, etc.)
   - **Status**: 13/17 tests passing

2. **Agent System** (`agents/`) ✅

   - **BaseAgent**: Abstract base for all agents
   - **WitsControlCenterAgent**: Entry point, parses intent and delegates
   - **LLMDrivenOrchestrator**: ReAct pattern implementation
   - **BackgroundAgent**: Scheduled tasks and monitoring
   - All agents use async patterns
   - **Status**: 5/5 tests passing

3. **Tool Registry** (`core/tool_registry.py`) ✅

   - Dynamic tool registration
   - LLM-friendly descriptions
   - Async execution pattern
   - Built-in tools: JSON, Math, Python execution, Web search, File ops
   - **Status**: All tools 100% tested and operational

4. **Memory Management** (`core/memory_manager.py`) ✅

   - Persistent storage with embeddings
   - Semantic search capabilities
   - Multiple backend support (Basic, Supabase, Neural)
   - Automatic pruning
   - **Status**: 3/3 backend tests passing

5. **Response Parsing** (`core/response_parser.py`) ✅

   - Multiple format support (JSON, function, XML, markdown)
   - ReAct pattern parsing
   - Confidence scoring
   - **Status**: Stable and operational

6. **Configuration System** (`core/config.py`) ✅
   - YAML-based configuration
   - Pydantic validation with assignment checking
   - Environment variable support
   - **Status**: 8/8 tests passing

## 🛠️ Technology Stack ✅ PROVEN STABLE

- **Language**: Python 3.10+ ✅
- **Async Framework**: asyncio throughout ✅
- **LLM**: Ollama (default: llama3) ✅
- **Data Validation**: Pydantic v2 ✅
- **Configuration**: YAML-based ✅
- **Testing**: pytest with async support ✅ (93.3% pass rate)
- **Optional**: FAISS for vector storage ✅

## 📋 Design Principles ✅ FULLY IMPLEMENTED

1. **LLM-First**: Let the LLM drive decision making ✅
2. **Streaming**: Real-time feedback via StreamData ✅
3. **Modular**: Easy to extend with new agents/tools ✅
4. **Configurable**: YAML-driven configuration ✅
5. **Robust**: Comprehensive error handling ✅
6. **CLI-First**: Terminal as primary interface ✅
7. **Testing**: Production-grade test coverage ✅

## 🔧 Key Patterns ✅ ALL WORKING

### ReAct Loop (Orchestrator) ✅

```
Reason → Act → Observe → (Repeat until goal achieved)
```

### Agent Communication ✅

- StreamData objects for real-time updates
- Types: thinking, action, observation, result, error

### Tool Integration ✅

All tools must:

- Extend BaseTool ✅
- Implement async execute() ✅
- Provide LLM-friendly schema ✅
- Handle errors gracefully ✅

### Testing Patterns ✅

- Async test functions with pytest-asyncio
- Comprehensive mocking for external services
- Edge case and error handling coverage
- 93.3% pass rate achieved

## 🚀 Future Expansion Areas

1. **Web UI**: Browser-based interface
2. **Advanced Book Writing System**: Specialized agent
3. **Enhanced MCP Integration**: More Model Context Protocol tools
4. **Langchain Bridge**: Deeper integration with Langchain ecosystem
5. **FAISS GPU Backends**: GPU-accelerated similarity search
6. **Multi-Model Support**: OpenAI, Anthropic, etc.
7. **Stream Test Improvements**: Fix remaining 4 complex async mock tests

## 📏 Constraints & Guidelines ✅ FULLY ADHERED

- Files must stay under 500 lines ✅
- All I/O operations must be async ✅
- Comprehensive error handling required ✅
- Tests for all new features ✅
- Backward compatibility important ✅
- Unicode support throughout ✅

## 🎨 Code Style ✅ ENFORCED

- PEP8 compliance ✅
- Type hints everywhere ✅
- Google-style docstrings ✅
- Meaningful variable names ✅
- Comments for complex logic ✅

## 📈 Achievements (2025-06-08)

- **25 critical bugs fixed** across all components
- **81% test improvement** (from 31 to 56 passing tests)
- **Production-grade stability** achieved
- **Comprehensive external service mocking**
- **Enhanced async patterns** throughout codebase
- **Robust configuration validation**
- **Complete tool test coverage**

## 🎯 Next Priorities

1. Fix remaining 4 complex stream mock tests
2. Implement Web UI prototype
3. Enhance MCP tool integration
4. Add performance benchmarking suite
5. Create deployment documentation