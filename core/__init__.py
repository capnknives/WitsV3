"""
WitsV3 Core Package
Provides core functionality for the WITS v3 LLM wrapper system
"""

from .base_tool import BaseTool
from .config import AgentConfig, LLMConfig, MemoryConfig, ToolConfig, WitsV3Config
from .llm_interface import BaseLLMInterface, OllamaInterface
from .mcp_adapter import MCPAdapter, MCPServer, MCPTool
from .memory_manager import BaseMemoryBackend, BasicMemoryBackend, MemoryManager
from .metrics import MetricsManager
from .response_parser import ParsedResponse, ReactParser, ResponseParser, ResponseType
from .schemas import AgentResponse, ConversationHistory, StreamData, ToolCall, ToolResult
from .tool_registry import ToolRegistry

__all__ = [
    # Configuration
    "WitsV3Config",
    "LLMConfig",
    "MemoryConfig",
    "AgentConfig",
    "ToolConfig",
    # LLM Interface
    "BaseLLMInterface",
    "OllamaInterface",
    # Memory Management
    "BaseMemoryBackend",
    "BasicMemoryBackend",
    "MemoryManager",
    # Schemas
    "StreamData",
    "AgentResponse",
    "ConversationHistory",
    "ToolCall",
    "ToolResult",
    # Tool System
    "BaseTool",
    "ToolRegistry",
    # Response Parsing
    "ResponseParser",
    "ReactParser",
    "ParsedResponse",
    "ResponseType",
    # MCP Integration
    "MCPAdapter",
    "MCPServer",
    "MCPTool",
    "MetricsManager",
]

__version__ = "1.0.0"
