"""
WitsV3 Agents Package
Provides agent implementations for the WITS v3 system
"""

from .background_agent import BackgroundAgent
from .base_agent import BaseAgent
from .base_orchestrator_agent import BaseOrchestratorAgent
from .wits_control_center_agent import WitsControlCenterAgent
from .llm_driven_orchestrator import LLMDrivenOrchestrator

__all__ = [
    'BackgroundAgent',
    'BaseAgent',
    'BaseOrchestratorAgent',
    'WitsControlCenterAgent',
    'LLMDrivenOrchestrator',
]

__version__ = "1.0.0"
