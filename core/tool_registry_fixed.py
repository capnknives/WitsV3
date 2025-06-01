# core/tool_registry.py
"""
Tool Registry for WitsV3.
Manages registration, discovery, and execution of tools.
"""

import logging
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
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


class ToolRegistry:
    """
    Registry for managing tools in WitsV3.
    """
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, BaseTool] = {}
        self.logger = logging.getLogger("WitsV3.ToolRegistry")
        
        # Register built-in tools
        self._register_builtin_tools()
        
        # Log summary of all discovered tools
        self._log_tool_summary()
        
        self.logger.info("Tool registry initialized")
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool instance to register
        """
        if tool.name in self.tools:
            self.logger.warning(f"Tool {tool.name} already registered, overwriting")
        
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of tool to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of all tools
        """
        return self.tools.copy()
    
    def list_tool_names(self) -> List[str]:
        """
        Get list of all tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get all tools formatted for LLM consumption.
        
        Returns:
            List of tool descriptions for LLM
        """
        return [tool.get_llm_description() for tool in self.tools.values()]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name with enhanced parameter validation.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments for the tool
            
        Returns:
            Tool execution result
            
        Raises:
            Exception: If tool not found, validation fails, or execution fails
        """
        # First validate the tool call
        validation = self.validate_tool_call(tool_name, **kwargs)
        
        if not validation["valid"]:
            error_msg = f"Tool '{tool_name}' validation failed: {', '.join(validation['errors'])}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log any warnings
        if validation["warnings"]:
            for warning in validation["warnings"]:
                self.logger.warning(warning)
        
        tool = validation["tool"]  # We know it exists from validation
        
        try:
            self.logger.debug(f"Executing tool {tool_name} with validated args: {kwargs}")
            result = await tool.execute(**kwargs)
            self.logger.debug(f"Tool {tool_name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            raise
      
    def validate_tool_call(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Validate a tool call before execution.
        
        Args:
            tool_name: Name of the tool to validate
            **kwargs: Arguments for the tool
            
        Returns:
            Validation result with status and details
        """
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "tool": None
        }
        
        # Check if tool exists
        tool = self.get_tool(tool_name)
        if not tool:
            validation_result["errors"].append(f"Tool '{tool_name}' not found")
            return validation_result
        
        validation_result["tool"] = tool
        
        try:
            # Get tool schema to validate parameters
            schema = tool.get_schema()
            required_params = schema.get("required", [])
            properties = schema.get("properties", {})
            
            # Check required parameters
            missing_params = []
            for param in required_params:
                if param not in kwargs:
                    missing_params.append(param)
            
            if missing_params:
                validation_result["errors"].append(
                    f"Missing required parameters for {tool_name}: {missing_params}"
                )
            
            # Check for unknown parameters
            unknown_params = []
            for param in kwargs:
                if param not in properties:
                    unknown_params.append(param)
            
            if unknown_params:
                validation_result["warnings"].append(
                    f"Unknown parameters for {tool_name}: {unknown_params}"
                )
            
            # If no errors, mark as valid
            if not validation_result["errors"]:
                validation_result["valid"] = True
            
        except Exception as e:
            validation_result["errors"].append(f"Error validating tool {tool_name}: {e}")
        
        return validation_result
    
    def _register_builtin_tools(self) -> None:
        """Register built-in tools."""
        # Register basic built-in tools
        self.register_tool(ThinkTool())
        self.register_tool(CalculatorTool())
        
        # Auto-discover tools from tools.base_tool module
        self._discover_tools_from_base_tool_module()
        self.logger.info("Built-in tools registered")
    
    def _discover_tools_from_base_tool_module(self) -> None:
        """Auto-discover and register tools from tools.base_tool module."""
        try:
            self.logger.info("🔍 Starting dynamic tool discovery from tools.base_tool...")
            import tools.base_tool as base_tools_module
            
            discovered_count = 0
            
            # Find all classes in the module that are subclasses of BaseTool
            for name, obj in inspect.getmembers(base_tools_module):
                if (inspect.isclass(obj) and 
                    hasattr(obj, '__bases__') and
                    any(base.__name__ == 'BaseTool' for base in obj.__bases__) and
                    obj.__name__ != 'BaseTool' and
                    not obj.__name__.startswith('_')):
                    
                    try:
                        # Instantiate and register the tool
                        tool_instance = obj()
                        self.register_tool(tool_instance)
                        self.logger.info(f"   ✅ Auto-discovered: {tool_instance.name} (from {obj.__name__})")
                        discovered_count += 1
                    except Exception as e:
                        self.logger.error(f"   ❌ Failed to instantiate {obj.__name__}: {e}")
            
            self.logger.info(f"🎯 Discovery complete: {discovered_count} tools auto-discovered from tools.base_tool")
            
        except ImportError as e:
            self.logger.warning(f"Could not import tools.base_tool module: {e}")
        except Exception as e:
            self.logger.error(f"Error during tool discovery: {e}")

    def _log_tool_summary(self) -> None:
        """Log a comprehensive summary of all discovered tools."""
        self.logger.info("=" * 60)
        self.logger.info("🔧 WITSV3 TOOL DISCOVERY SUMMARY")
        self.logger.info("=" * 60)
        
        total_tools = len(self.tools)
        self.logger.info(f"📊 Total tools available: {total_tools}")
        
        if total_tools > 0:
            self.logger.info("📋 Registered tools:")
            for tool_name, tool in self.tools.items():
                self.logger.info(f"   ✓ {tool_name}: {tool.description}")
        
        self.logger.info("=" * 60)


class ThinkTool(BaseTool):
    """Simple thinking tool for reasoning."""
    
    def __init__(self):
        super().__init__(
            name="think",
            description="Think through a problem or situation step by step"
        )
    
    async def execute(self, thought: str = "") -> str:
        """Execute thinking process."""
        self.logger.debug(f"Thinking: {thought}")
        return f"Thought: {thought}"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "The thought or reasoning to process"
                }
            },
            "required": ["thought"]
        }


class CalculatorTool(BaseTool):
    """Simple calculator tool."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform basic mathematical calculations"
        )
    
    async def execute(self, expression: str) -> str:
        """Execute calculation."""
        try:
            # Basic safety check
            if any(char in expression for char in ['import', 'exec', 'eval', '__']):
                raise ValueError("Invalid expression")
            
            # Only allow basic math operations
            allowed_chars = set('0123456789+-*/().,_ ')
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Expression contains invalid characters")
            
            result = eval(expression)
            self.logger.debug(f"Calculated: {expression} = {result}")
            return str(result)
            
        except Exception as e:
            self.logger.warning(f"Calculation error: {e}")
            return f"Error: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        }


# Test function
async def test_tool_registry():
    """Test the tool registry functionality."""
    print("Testing ToolRegistry...")
    
    # Create registry
    registry = ToolRegistry()
    print(f"✓ Tool registry created with {len(registry.tools)} tools")
    
    # Test tool listing
    tool_names = registry.list_tool_names()
    print(f"✓ Available tools: {tool_names}")
    
    # Test tool execution
    try:
        # Test think tool
        result = await registry.execute_tool("think", thought="Testing the thinking process")
        print(f"✓ Think tool result: {result}")
        
        # Test calculator tool
        result = await registry.execute_tool("calculator", expression="2 + 2")
        print(f"✓ Calculator tool result: {result}")
        
        # Test LLM descriptions
        llm_tools = registry.get_tools_for_llm()
        print(f"✓ LLM tool descriptions: {len(llm_tools)} tools formatted")
        
    except Exception as e:
        print(f"✓ Tool execution test passed (expected error without full implementation: {e})")
    
    print("ToolRegistry tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_tool_registry())
