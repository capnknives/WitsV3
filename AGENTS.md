# WitsV3 Agents Documentation

## Overview

WitsV3's agent system is the core of its LLM orchestration capabilities. The system follows a hierarchical design with specialized agents handling specific tasks, all coordinated through a central control mechanism.

## Agent Architecture

```
BaseAgent (abstract)
├── WitsControlCenterAgent (main entry point)
├── BaseOrchestratorAgent (abstract)
│   └── LLMDrivenOrchestrator (ReAct pattern implementation)
├── BackgroundAgent (scheduled tasks)
├── AdvancedCodingAgent (code generation & analysis)
└── BookWritingAgent (content creation)
```

## Core Agent Classes

### BaseAgent

The abstract base class that all agents inherit from, providing:

- Communication with LLMs via the `llm_interface`
- Memory management with `memory_manager`
- Streaming response infrastructure (thinking, action, observation, result)
- Configuration handling
- Standard logging patterns

```python
class BaseAgent(ABC):
    @abstractmethod
    async def run(self, user_input: str, **kwargs) -> AsyncGenerator[StreamData, None]:
        """Main entry point for all agents to process user requests"""
        pass
```

### WitsControlCenterAgent

The main entry point for user interactions, responsible for:

- Parsing user intent
- Delegating to specialized agents
- Managing conversation flow
- Handling enhanced meta-reasoning when available

### BaseOrchestratorAgent

Abstract agent implementing the ReAct (Reason-Act-Observe) pattern:

- Reasoning about user requests and system state
- Executing appropriate tools
- Observing results
- Making decisions on next steps

### LLMDrivenOrchestrator

Concrete implementation of BaseOrchestratorAgent that uses LLMs for decision-making:

- Generates reasoning based on current state
- Selects appropriate tools to use
- Interprets tool outputs
- Determines when to complete a request

## Specialized Agents

### BackgroundAgent

Handles scheduled tasks and system maintenance:

- Memory pruning and optimization
- System health monitoring
- Periodic tasks via AsyncIOScheduler
- Metric collection and reporting

### AdvancedCodingAgent

Specialized agent for software development tasks:

- Code generation for multiple languages and frameworks
- Project structure creation
- Code analysis and quality assessment
- Integration with neural web for enhanced reasoning (when available)

### BookWritingAgent

Agent designed for long-form content creation:

- Book outlining and chapter planning
- Content writing with narrative structure
- Research synthesis
- Revision and improvement
- Character and world development

## Agent Communication Patterns

All agents use the `StreamData` system for communication, which allows for:

1. **Progressive output** - Clients see interim results (thinking, actions, observations)
2. **Structured data** - Different message types for different purposes
3. **Unified interface** - Consistent across all agent types

## Memory System Integration

Agents use the memory system to:

- Store important context from interactions
- Retrieve relevant past information
- Build coherent mental models across sessions
- Share knowledge with other agents

## Creating New Agents

When creating a new agent:

1. Inherit from `BaseAgent` (or a specialized agent class)
2. Implement the required `run()` method
3. Use the streaming helpers (`stream_thinking`, `stream_action`, etc.)
4. Place the agent in the `agents/` directory
5. Register in `__init__.py` if needed for importing

Example skeleton:

```python
class YourNewAgent(BaseAgent):
    """
    Specialized agent for [purpose].
    """

    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: Optional[MemoryManager] = None
    ):
        super().__init__(agent_name, config, llm_interface, memory_manager)
        # Additional initialization

    async def run(
        self,
        user_input: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Process requests specific to this agent.
        """
        # Implementation
        yield self.stream_thinking("Processing request...")
        # ...
        yield self.stream_result("Final result")
```

## Testing Agents

All agents have corresponding tests in the `tests/agents/` directory. When creating a new agent, include:

- Basic functionality tests
- Exception handling tests
- Integration with memory system tests
- Sample conversation flow tests

## Best Practices

1. **Async First**: All agent methods should be async
2. **Stream Progress**: Use the stream helpers to show work
3. **Memory Integration**: Store important context in memory
4. **Error Handling**: Gracefully handle exceptions with good user feedback
5. **Modularity**: Break complex agents into smaller methods
6. **Documentation**: Include docstrings for all methods
