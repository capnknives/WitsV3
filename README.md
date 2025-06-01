# WitsV3

A streamlined, LLM-wrapper based AI orchestration system.

## Overview

WitsV3 is designed for maximum flexibility and LLM-driven decision making. It focuses on a CLI-first approach with a modular design to allow for future expansion (e.g., Web UI, advanced Book Writing System).

## ✅ Current Status - FULLY OPERATIONAL

WitsV3 has been successfully implemented and tested with the following features working:

- **Tool Registry**: 6 tools properly registered and functional
- **LLM Integration**: Successfully connects to Ollama with llama3 model  
- **Memory Management**: Proper datetime serialization and JSON handling
- **Agent System**: LLM-driven orchestrator with ReAct pattern implementation
- **File Operations**: Read, write, and directory listing capabilities
- **Unicode Support**: Clean text output without character encoding issues
- **Adaptive LLM System**: Dynamic routing to specialized modules based on query complexity and domain

This version prioritizes:

- LLM Wrapper Architecture
- Core agent system (Control Center, Orchestrator)
- Essential memory management (with optional FAISS-GPU)
- Extensible tool system, including MCP (Model Context Protocol) tools
- Langchain integration capabilities
- Adaptive LLM capabilities for optimized resource usage

## Getting Started

(Instructions to be added as core components are implemented)

### Prerequisites

- Python 3.10+
- Ollama (with models like Llama3, CodeLlama, etc.)
- Other dependencies (see `requirements.txt`)

### Installation

(To be detailed)

### Running WitsV3

(To be detailed)

## Adaptive LLM System

WitsV3 includes a sophisticated Adaptive LLM System that dynamically routes queries to specialized modules based on complexity and domain, with semantic caching for improved performance.

Key components:

- **ComplexityAnalyzer**: Analyzes query complexity and domain to route to appropriate modules
- **DynamicModuleLoader**: Manages specialized LLM modules with VRAM/RAM budgeting and quantization
- **SemanticCache**: Stores and retrieves patterns based on semantic similarity
- **AdaptiveLLMInterface**: Main interface that integrates all components

Benefits:

- **Resource Efficiency**: Uses appropriate-sized models based on query complexity
- **Domain Specialization**: Routes to domain-specific modules for better responses
- **Performance Optimization**: Caches common patterns and responses
- **Adaptive Quantization**: Uses lower precision for simple queries, full precision for complex ones

## Project Structure

```
WitsV3/
├── agents/              # Core agent implementations
├── core/                # Core system components (config, llm, memory, schemas, tools)
├── data/                # Data storage (memory files, mcp tools)
├── logs/                # Application logs
├── tools/               # Implementations of various tools
├── config.yaml          # Main configuration file
├── run_v3.py            # Main entry point (to be created)
├── requirements.txt     # Python dependencies (to be created)
└── README.md            # This file
```

## Key Features (Planned/In-Progress)

- **LLM-Driven Orchestration:** Leverages the power of LLMs for task decomposition and execution planning.
- **WitsControlCenterAgent (WCCA):** User interaction and goal clarification.
- **OrchestratorAgent:** ReAct-style loop for managing tasks.
- **Memory Management:** Persistent memory with semantic search capabilities.
- **MCP Tools:** Dynamically created and managed tools based on LLM descriptions.
- **Langchain Integration:** Ability to leverage Langchain tools and functionalities.
- **CLI First:** Robust command-line interface for interaction and development.
- **Adaptive LLM System:** Dynamic routing to specialized modules based on complexity and domain.

## Contributing

(Guidelines to be added)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
