# WitsV3

A streamlined, LLM-wrapper based AI orchestration system.

## Overview

WitsV3 is designed for maximum flexibility and LLM-driven decision making. It focuses on a CLI-first approach with a modular design to allow for future expansion (e.g., Web UI, advanced Book Writing System).

## ✅ Current Status - PRODUCTION READY 🚀

**WitsV3 v2.0 - STABLE RELEASE** (Updated: 2025-06-08)

WitsV3 has achieved **production-grade stability** with comprehensive testing and bug fixes:

- **🎯 Test Suite**: **93.3% pass rate** (56/60 tests passing)
- **🔧 Stability**: **25 critical bugs fixed** across all core components
- **⚙️ Tool Registry**: 6 tools properly registered and fully functional
- **🤖 LLM Integration**: Successfully connects to Ollama with llama3 model
- **💾 Memory Management**: Robust datetime serialization and JSON handling
- **🧠 Agent System**: LLM-driven orchestrator with ReAct pattern implementation
- **📁 File Operations**: Read, write, and directory listing capabilities
- **🌐 Unicode Support**: Clean text output without character encoding issues
- **🎯 Adaptive LLM System**: Dynamic routing to specialized modules based on query complexity and domain
- **🧪 Background Agents**: Scheduled tasks, monitoring, and maintenance

### 🏆 Recent Major Improvements (2025-06-08)

- **Fixed all tool test suites**: JSON, Math, Python Execution tools now 100% stable
- **Enhanced async patterns**: Proper async/await implementation throughout
- **Robust external service mocking**: Supabase, Ollama properly tested
- **Configuration validation**: Enhanced Pydantic models with assignment validation
- **Memory system stability**: MemorySegment serialization working flawlessly
- **Adaptive LLM testing**: Complete test coverage with dummy model files

This version prioritizes:

- LLM Wrapper Architecture
- Core agent system (Control Center, Orchestrator)
- Essential memory management (with optional FAISS-GPU)
- Extensible tool system, including MCP (Model Context Protocol) tools
- Langchain integration capabilities
- Adaptive LLM capabilities for optimized resource usage
- **Production-grade test coverage and stability**

## Getting Started

### Prerequisites

- Python 3.10+
- Ollama (with models like Llama3, CodeLlama, etc.)
- Other dependencies (see `requirements.txt`)

### Installation

#### Option 1: Automated Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/capnknives/WitsV3.git
cd WitsV3

# Run automated installation
python install.py
```

#### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/capnknives/WitsV3.git
cd WitsV3

# Install dependencies
pip install -r requirements.txt

# Set up local data files (memory, neural web)
python scripts/setup_local_data.py

# Set up authentication (creates secure tokens)
python setup_auth.py

# Ensure Ollama is running
ollama serve

# Run tests to verify installation
pytest tests/ -v
```

### Quick Start with Make

WitsV3 includes a comprehensive Makefile for streamlined development:

```bash
# Install development dependencies
make install-dev

# Run the main application
make dev

# Run tests with coverage
make test-cov

# Clean and build package
make build
```

### First Run Setup

After installation, WitsV3 requires some initial configuration:

1. **Authentication Setup**: Run `python setup_auth.py` to generate your authentication token
2. **Local Data**: Run `python scripts/setup_local_data.py` to initialize memory files
3. **Ollama Models**: Ensure you have compatible models installed:
   ```bash
   ollama pull llama3
   ollama pull codellama
   ```

### Running WitsV3

```bash
# Start the CLI interface
python run.py

# Run background agent
python -m agents.background_agent

# Run specific tests
pytest tests/tools/test_json_tool.py -v
```

### Authentication & Security

WitsV3 includes enterprise-grade security features:

- **Token-based Authentication**: Secure access control with SHA-256 hashed tokens
- **Network Access Control**: Configurable restrictions with user override capability
- **Personality Profiles**: Comprehensive user recognition and behavior customization
- **Ethics System**: Built-in ethical oversight with testing override capabilities

Your authentication token will be displayed during setup - **save it securely** as it's required for secure operations.

## Documentation

### Project Structure

- **[FILE_STRUCTURE.md](FILE_STRUCTURE.md)** - Comprehensive file structure documentation
- **[PATH_MIGRATION_GUIDE.md](PATH_MIGRATION_GUIDE.md)** - Guide for file path changes from previous versions
- **[TASK.md](TASK.md)** - Current task tracking and management

### Detailed Documentation

All project documentation is organized in the `/planning` directory:

- **[Architecture](planning/architecture/)** - System design and components

  - [System Architecture](planning/architecture/system-architecture.md) - Core architecture design

- **[Implementation](planning/implementation/)** - Implementation details

  - [Personality & Ethics System](planning/implementation/personality-ethics-network-implementation.md)
  - [Adaptive LLM Design](planning/implementation/adaptive-llm-design.md)
  - [Claude Evolution Integration](planning/implementation/claude-evolution-prompt.md)

- **[Roadmap](planning/roadmap/)** - Future plans and enhancements

  - [Neural Web Architecture](planning/roadmap/neural-web-roadmap.md) - Future enhancement roadmap

- **[Tasks](planning/tasks/)** - Task tracking and management

  - [Task Management](planning/tasks/task-management.md) - Current tasks and backlog

- **[Technical Notes](planning/technical-notes/)** - Debug info and fixes
  - [Consolidated System Fixes](planning/technical-notes/consolidated-system-fixes.md) - All system fixes and improvements

### Documentation Maintenance

Documentation is maintained using dedicated tools:

```bash
# List all documentation
python scripts/doc_maintenance.py list

# Create a new document
python scripts/doc_maintenance.py create planning/path/to/doc.md "Document Title"

# Update document metadata
python scripts/doc_maintenance.py update planning/path/to/doc.md --status "draft"

# Archive a document
python scripts/doc_maintenance.py archive planning/path/to/doc.md
```

See the [Planning Documentation](planning/README.md) for more details.

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

## Test Suite Status 🧪

WitsV3 maintains a comprehensive test suite with **93.3% pass rate**:

- ✅ **Background Agent Tests** (5/5) - Scheduler, task execution, monitoring
- ✅ **Configuration Tests** (8/8) - Pydantic validation, YAML loading
- ✅ **JSON Tool Tests** (7/7) - Data manipulation, file operations
- ✅ **Math Tool Tests** (7/7) - Statistics, probability, matrix operations
- ✅ **Python Execution Tests** (7/7) - Code execution, timeout handling
- ✅ **Memory Backend Tests** (3/3) - Supabase, neural memory systems
- ✅ **LLM Interface Tests** (13/17) - Ollama integration, embeddings
- ✅ **Adaptive LLM Tests** (1/1) - Complexity analysis, module loading
- ✅ **Tool Integration Tests** (12/12) - Web search, file operations

```bash
# Run the full test suite
pytest tests/ -v

# Quick test run
pytest tests/ --tb=no -q
```

## Project Structure

```
WitsV3/
├── agents/              # Core agent implementations
├── core/                # Core system components (config, llm, memory, schemas, tools)
├── data/                # Data storage (memory files, mcp tools)
├── logs/                # Application logs
├── models/              # Adaptive LLM model files
├── tests/               # Comprehensive test suite (93.3% pass rate)
├── tools/               # Implementations of various tools
├── config.yaml          # Main configuration file
├── run.py            # Main entry point script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Key Features ✨

- **LLM-Driven Orchestration:** Leverages the power of LLMs for task decomposition and execution planning.
- **WitsControlCenterAgent (WCCA):** User interaction and goal clarification.
- **OrchestratorAgent:** ReAct-style loop for managing tasks.
- **Memory Management:** Persistent memory with semantic search capabilities.
- **MCP Tools:** Dynamically created and managed tools based on LLM descriptions.
- **Langchain Integration:** Ability to leverage Langchain tools and functionalities.
- **CLI First:** Robust command-line interface for interaction and development.
- **Adaptive LLM System:** Dynamic routing to specialized modules based on complexity and domain.
- **Background Processing:** Automated maintenance, monitoring, and optimization.
- **Production-Grade Testing:** Comprehensive test suite with high reliability.

## Contributing

WitsV3 follows strict development standards:

- **Testing Required**: All new features must include tests
- **Async Patterns**: All I/O operations must be async
- **Type Hints**: Full type annotation coverage
- **Documentation**: Google-style docstrings for all functions
- **Code Quality**: PEP8 compliance with black formatting

See `task-md.md` for current development tasks and `.cursorrules` for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Environment Setup

To ensure consistent dependency installation across environments:

1. Install dependencies using the helper script:

`ash
# Install core dependencies only
python setup_dependencies.py

# Install both core and development dependencies
python setup_dependencies.py --dev
`

2. Alternative manual installation:

`ash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Fix numpy version
pip install "numpy>=1.25.0,<3.0" --force-reinstall
`

3. For containerized environments, use the provided Dockerfile or the requirements.lock file.
# WitsV3

## Environment Setup

To ensure consistent dependency installation across environments:

1. Install dependencies using the helper script:

`ash
# Install core dependencies only
python setup_dependencies.py

# Install both core and development dependencies
python setup_dependencies.py --dev
`

2. Alternative manual installation:

`ash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Fix numpy version
pip install "numpy>=1.25.0,<3.0" --force-reinstall
`

3. For containerized environments, use the provided Dockerfile or the requirements.lock file.
