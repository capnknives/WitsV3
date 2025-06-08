"""
Agents module for WitsV3
This module contains all specialized agents that extend the base agent functionality.
"""

from .base_agent import BaseAgent, AgentCapability, ConversationHistory
from .react_agent import ReactAgent
from .research_agent import ResearchAgent
from .code_agent import CodeAgent
from .writing.book_writing_agent import BookWritingAgent
from .control_center_agent import ControlCenterAgent
from .orchestrator_agent import OrchestratorAgent
from .tool_agent import ToolAgent
from .evaluation_agent import EvaluationAgent
from .basic_agent import BasicAgent
from .coding.advanced_coding_agent import AdvancedCodingAgent

__all__ = [
    'BaseAgent',
    'AgentCapability', 
    'ConversationHistory',
    'ReactAgent',
    'ResearchAgent',
    'CodeAgent',
    'BookWritingAgent',
    'ControlCenterAgent',
    'OrchestratorAgent',
    'ToolAgent',
    'EvaluationAgent',
    'BasicAgent',
    'AdvancedCodingAgent'
]

__version__ = "1.0.0"
