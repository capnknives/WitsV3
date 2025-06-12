# WITS Synthetic Brain

## Overview

The WITS Synthetic Brain is an expansion of the WitsV3 system that transitions WITS from a language-model-based assistant to a fully modular synthetic brain. This implementation adds memory systems, sensorimotor input/output, self-modeling, autonomous goal capabilities, emotion modeling, ethical reasoning, and symbolic planning.

## Core Components

### Memory Handling

#### Memory Handler Updated (`core/memory_handler_updated.py`)
The enhanced Memory Handler provides a unified interface for managing various memory systems:
- **Working Memory**: Short-term active memory for current processing
- **Episodic Memory**: Event-based memories with temporal information
- **Semantic Memory**: Conceptual knowledge and facts
- **Procedural Memory**: Action sequences and procedures

#### Memory Handler Base Version (`core/memory_handler.py`)
A base implementation of the memory handler that provides core functionality without all the enhancements.

#### Memory Manager (`core/memory_manager.py`)
An extended memory management system that handles vector storage, embedding generation, and memory operations.

### Cognitive Architecture 

#### Cognitive Architecture Updated (`core/cognitive_architecture_updated.py`)
The enhanced Cognitive Architecture coordinates the operation of all cognitive subsystems:
- **Perception**: Processing and understanding input
- **Reasoning**: Multiple reasoning styles (deductive, inductive, analogical)
- **Metacognition**: Self-reflection and improvement
- **Planning**: Action planning and execution

#### Cognitive Architecture Base Version (`core/cognitive_architecture.py`)
A base implementation of the cognitive architecture that provides core functionality without all the enhancements.

### Configuration (`config/wits_core.yaml`)

The central configuration file that defines:
- Identity and self-model parameters
- Memory system settings
- Cognitive module configuration
- System integration settings

## Implementation Status

The current implementation focuses on Phase 1 (Core Cognitive Layer Integration) of the expansion plan. For detailed status, see [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md).

## Usage

### Basic Usage

```python
from core.cognitive_architecture_updated import CognitiveArchitecture

# Initialize the cognitive architecture
brain = CognitiveArchitecture()

# Process input with the synthetic brain
response = await brain.process_input("How can I optimize my learning process?")

# Response will include processed output and any actions taken
print(response.content)
```

### Working with Memory

```python
from core.memory_handler_updated import MemoryHandler

# Initialize memory handler
memory = MemoryHandler()

# Store information in memory
memory_id = await memory.remember(
    "User expressed interest in machine learning",
    memory_type="episodic",
    metadata={"source": "conversation", "topic": "learning"}
)

# Recall relevant memories
memories = await memory.recall("machine learning interests")

# Process memories
for mem in memories:
    print(f"Memory: {mem['content']}, Relevance: {mem['relevance']}")
```

## Advanced Tools

### Neural Web Visualization (`tools/neural_web_visualization.py`)

Tools for visualizing the neural web and its connections:
- Static graph generation with NetworkX and Matplotlib
- Interactive HTML visualization with D3.js
- Domain-based color coding and filtering

### Neural Web NLP (`tools/neural_web_nlp.py`)

Specialized natural language processing tools for the neural web:
- Hybrid concept extraction (pattern-based + LLM-based)
- Relationship extraction between concepts
- Domain classification with confidence scoring

### Enhanced Reasoning (`tools/enhanced_reasoning.py`)

Advanced reasoning patterns for the synthetic brain:
- Deductive reasoning (general principles → specific conclusions)
- Inductive reasoning (specific observations → general patterns)
- Analogical reasoning (knowledge transfer from similar situations)

## Testing

The synthetic brain components are thoroughly tested:
- **Unit Tests**: Tests for individual components in the `tests/core/` directory
- **Integration Tests**: Tests for component integration in the project root
- **Memory Tests**: Specialized tests for memory pruning and optimization

## Documentation

- [`docs/WITS_Synthetic_Brain_Expansion_Guide_With_Emojis.md`](docs/WITS_Synthetic_Brain_Expansion_Guide_With_Emojis.md): Overall implementation plan
- [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md): Current implementation status
- [`docs/MEMORY_HANDLER_FIXES.md`](docs/MEMORY_HANDLER_FIXES.md): Documentation of fixes for memory handler
- [`docs/SYNTHETIC_BRAIN_NEXT_STEPS.md`](docs/SYNTHETIC_BRAIN_NEXT_STEPS.md): Upcoming work and priorities
- [`docs/REMAINING_TASKS.md`](docs/REMAINING_TASKS.md): Detailed list of remaining implementation tasks
