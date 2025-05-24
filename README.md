# WitsV3

A streamlined, LLM-wrapper based AI orchestration system.

## Overview

WitsV3 is designed for maximum flexibility and LLM-driven decision making. It focuses on a CLI-first approach with a modular design to allow for future expansion (e.g., Web UI, advanced Book Writing System).

This version prioritizes:

- LLM Wrapper Architecture
- Core agent system (Control Center, Orchestrator)
- Essential memory management (with optional FAISS-GPU)
- Extensible tool system, including MCP (Model Context Protocol) tools
- Langchain integration capabilities

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

## Contributing

(Guidelines to be added)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
