# core/base_tool.py
"""
Base tool class for WitsV3.
This module contains the base class that all tools must inherit from.
"""

import logging
from typing import Any, Dict
from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Base class for all tools in WitsV3.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the tool.
        
        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"WitsV3.Tool.{name}")
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given arguments.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            Tool execution result
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool's schema for LLM consumption.
        
        Returns:
            Tool schema dictionary
        """
        pass
    
    def get_llm_description(self) -> Dict[str, Any]:
        """
        Get tool description for LLM.
        
        Returns:
            Tool description for LLM
        """
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.get_schema()
        }
