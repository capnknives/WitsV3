# core/base_tool.py
"""
Base tool class for WitsV3.
This module contains the base class that all tools must inherit from.
"""

import logging
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from .schemas import ToolSchema, ToolParameter, ToolValidationResult


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
        self._tool_schema: Optional[ToolSchema] = None

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

    def get_enhanced_schema(self) -> ToolSchema:
        """
        Get the enhanced tool schema for validation.

        Returns:
            ToolSchema instance for validation
        """
        if not self._tool_schema:
            # Convert legacy schema to enhanced schema
            legacy_schema = self.get_schema()
            parameters = []

            schema_props = legacy_schema.get("properties", {})
            required_props = legacy_schema.get("required", [])

            for prop_name, prop_def in schema_props.items():
                param = ToolParameter(
                    name=prop_name,
                    type=prop_def.get("type", "string"),
                    description=prop_def.get("description", ""),
                    required=prop_name in required_props,
                    default=prop_def.get("default"),
                    enum_values=prop_def.get("enum"),
                    min_value=prop_def.get("minimum"),
                    max_value=prop_def.get("maximum"),
                    min_length=prop_def.get("minLength"),
                    max_length=prop_def.get("maxLength"),
                    pattern=prop_def.get("pattern")
                )
                parameters.append(param)

            self._tool_schema = ToolSchema(
                name=self.name,
                description=self.description,
                parameters=parameters
            )

        return self._tool_schema

    def validate_arguments(self, arguments: Dict[str, Any]) -> ToolValidationResult:
        """
        Validate tool arguments using enhanced schema.

        Args:
            arguments: Arguments to validate

        Returns:
            ToolValidationResult with validation outcome
        """
        schema = self.get_enhanced_schema()
        return schema.validate_arguments(arguments)

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
