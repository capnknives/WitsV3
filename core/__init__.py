"""
WitsV3 Core Package
Provides core functionality for the WITS v3 LLM wrapper system
"""

from .config import WitsV3Config, LLMConfig, MemoryConfig, AgentConfig, ToolConfig
from .llm_interface import BaseLLMInterface, OllamaInterface
from .memory_manager import BaseMemoryBackend, BasicMemoryBackend, MemoryManager
from .schemas import StreamData, AgentResponse, ConversationHistory, ToolCall, ToolResult
from .base_tool import BaseTool
from .tool_registry import ToolRegistry
from .response_parser import ResponseParser, ReactParser, ParsedResponse, ResponseType
from .mcp_adapter import MCPAdapter, MCPServer, MCPTool
from .metrics import MetricsManager

__all__ = [
    # Configuration
    'WitsV3Config', 'LLMConfig', 'MemoryConfig', 'AgentConfig', 'ToolConfig',

    # LLM Interface
    'BaseLLMInterface', 'OllamaInterface',

    # Memory Management
    'BaseMemoryBackend', 'BasicMemoryBackend', 'MemoryManager',

    # Schemas
    'StreamData', 'AgentResponse', 'ConversationHistory', 'ToolCall', 'ToolResult',

    # Tool System
    'BaseTool', 'ToolRegistry',

    # Response Parsing
    'ResponseParser', 'ReactParser', 'ParsedResponse', 'ResponseType',

    # MCP Integration
    'MCPAdapter', 'MCPServer', 'MCPTool',

    'MetricsManager',
]

__version__ = "1.0.0"
