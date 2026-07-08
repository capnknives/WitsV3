# WitsV3 Agents Documentation

## Overview

WitsV3 agents are the routing and reasoning layer of a local LLM orchestration system.  
Users typically chat through the **Web UI** (`run_web.py`) or CLI (`run.py`); the control center decides intent and delegates.

## Agent architecture

```
BaseAgent (abstract)
├── WitsControlCenterAgent     # main entry — intent + routing
├── BaseOrchestratorAgent      # ReAct loop
│   └── LLMDrivenOrchestrator  # tool calling + synthesis guard
├── BackgroundAgent            # scheduled maintenance (optional / Docker path)
├── AdvancedCodingAgent        # scaffolds + verified file edits
├── SelfRepairAgent            # log/test diagnose → verified fixes
└── BookWritingAgent           # long-form content
```

## Core classes

### BaseAgent

Abstract base providing:

- LLM access via `llm_interface`
- Memory via `memory_manager`
- Streaming helpers (thinking / action / observation / result)
- Config + logging

```python
class BaseAgent(ABC):
    @abstractmethod
    async def run(self, user_input: str, **kwargs) -> AsyncGenerator[StreamData, None]:
        """Main entry point for all agents to process user requests."""
        pass
```

### WitsControlCenterAgent

Entry point for user turns:

- Parses intent
- Routes to specialized agents **before** generic enhanced paths (specialists are not dead code)
- Delegates tool-heavy work to the orchestrator
- Manages conversation flow

### BaseOrchestratorAgent / LLMDrivenOrchestrator

ReAct (Reason → Act → Observe):

- Plan, call tools, read observations
- **Synthesis guard** rejects final answers that ignore usable search observations (retry once, then auto-synthesize when needed)
- JSON robustness for local models (`format=json`, repair-reparse)

### BackgroundAgent

Optional scheduled tasks (memory maintenance, metrics, self-repair parity). Prefer the **in-process** daily self-repair schedule on `run.py` / `run_web.py` for local use; this agent is mainly for the Docker background path.

## Specialized agents

### AdvancedCodingAgent

- Project scaffolds written under `workspace/<name>/` with `py_compile` checks
- Requests that name an existing project file use the same verified-edit pipeline as self-repair (`core/safe_code_editor.py`)

### SelfRepairAgent

1. Target a named file, or scan `logs/witsv3.log` (then failing tests if needed)  
2. Ask the LLM for a full corrected file  
3. Apply via `apply_code_fix` (pytest gate + git commit or full revert)  
4. Optionally restart if `self_repair.restart_after_fix` is enabled  

### BookWritingAgent

Outlining, chapters, research synthesis, revision — content-focused, not the verified filesystem edit path.

## Communication

All agents stream `StreamData` so clients can show progressive thinking, tool actions, and final results.

## Memory

Agents store and retrieve segments through the memory manager so context survives across turns and (depending on backend) sessions.

## Creating a new agent

1. Inherit from `BaseAgent`  
2. Implement `async def run(...)`  
3. Use stream helpers  
4. Place under `agents/`  
5. Wire routing from the control center when it should be user-reachable  

```python
class YourNewAgent(BaseAgent):
    async def run(
        self,
        user_input: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamData, None]:
        yield self.stream_thinking("Processing request...")
        # ...
        yield self.stream_result("Final result")
```

## Testing

Mirror under `tests/agents/`: happy path, errors, memory/tool integration where relevant.

## Best practices

1. Async I/O everywhere  
2. Stream progress generously  
3. Persist important context  
4. Fail gracefully with clear user-facing messages (`core/user_errors.py` for LLM outages)  
5. Keep files under ~500 lines; share pipelines (e.g. safe code editor) instead of forking logic  
