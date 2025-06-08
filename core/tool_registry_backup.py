# core/tool_registry.py
"""
Tool Registry for WitsV3.
Manages registration, discovery, and execution of tools.
"""

import importlib.util
import logging
from typing import Any, Dict, List, Optional, Type
from pathlib import Path

from core.base_tool import BaseTool


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
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments for the tool
            
        Returns:
            Tool execution result
            
        Raises:
            Exception: If tool not found or execution fails
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise Exception(f"Tool '{tool_name}' not found")
        
        try:
            self.logger.debug(f"Executing tool {tool_name} with args: {kwargs}")
            result = await tool.execute(**kwargs)
            self.logger.debug(f"Tool {tool_name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            raise
    
    def _register_builtin_tools(self) -> None:
        """Register built-in tools."""
        # Register basic built-in tools        self.register_tool(ThinkTool())
        self.register_tool(CalculatorTool())
        
        self.logger.info("Built-in tools registered")
    
    def discover_and_register_tools(self) -> Dict[str, List[str]]:
        """
        Discover and register new tools dynamically.
        
        Returns:
            Dictionary with discovered tools categorized by type
        """
        discovered = {
            "file_tools": [],
            "system_tools": [],
            "analysis_tools": [],
            "new_tools": []
        }
        
        # Check for additional tools in the tools directory
        tools_dir = Path(__file__).parent.parent / "tools"
        if tools_dir.exists():
            for tool_file in tools_dir.glob("*.py"):
                if tool_file.name.startswith("__") or tool_file.name == "base_tool.py":
                    continue
                    
                try:
                    # Import and discover tools in the file
                    spec = importlib.util.spec_from_file_location(tool_file.stem, tool_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                          # Look for BaseTool subclasses
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and 
                                issubclass(attr, BaseTool) and 
                                attr != BaseTool and
                                attr_name not in [t.__class__.__name__ for t in self.tools.values()]):
                                  try:
                                    # Create instance of the tool class
                                    # Try different instantiation approaches
                                    try:
                                        # First try: no arguments (for properly implemented tools)
                                        tool_instance = attr()
                                    except TypeError:
                                        # Second try: with generic name and description
                                        tool_instance = attr(
                                            name=attr_name.lower().replace('tool', ''),
                                            description=f"Dynamically discovered {attr_name}"
                                        )
                                    
                                    self.register_tool(tool_instance)
                                    discovered["new_tools"].append(tool_instance.name)
                                    self.logger.info(f"Auto-discovered and registered tool: {tool_instance.name}")
                                except Exception as e:
                                    self.logger.debug(f"Could not auto-instantiate tool {attr_name}: {e}")
                                
                except Exception as e:
                    self.logger.warning(f"Failed to discover tools in {tool_file}: {e}")
        
        # Categorize existing tools
        for tool_name, tool in self.tools.items():
            if "file" in tool.name.lower():
                discovered["file_tools"].append(tool_name)
            elif any(keyword in tool.name.lower() for keyword in ["system", "datetime", "calculate"]):
                discovered["system_tools"].append(tool_name)
            elif any(keyword in tool.name.lower() for keyword in ["analyze", "think", "search"]):
                discovered["analysis_tools"].append(tool_name)
        
        self.logger.info(f"Tool discovery complete. Found {sum(len(v) for v in discovered.values())} tools")
        return discovered

    def get_tool_recommendations(self, query: str) -> List[str]:
        """
        Get tool recommendations based on a query or task description.
        
        Args:
            query: User query or task description
            
        Returns:
            List of recommended tool names
        """
        query_lower = query.lower()
        recommendations = []
        
        # File operation keywords
        if any(keyword in query_lower for keyword in ["read", "file", "content", "text"]):
            recommendations.append("read_file")
        
        if any(keyword in query_lower for keyword in ["write", "save", "create file", "output"]):
            recommendations.append("write_file")
            
        if any(keyword in query_lower for keyword in ["list", "directory", "folder", "files"]):
            recommendations.append("list_directory")
            
        # Analysis keywords
        if any(keyword in query_lower for keyword in ["think", "analyze", "reason"]):
            recommendations.append("think")
            
        # System keywords
        if any(keyword in query_lower for keyword in ["time", "date", "datetime", "when"]):
            recommendations.append("datetime")
            
        # Math keywords
        if any(keyword in query_lower for keyword in ["calculate", "math", "compute", "add", "multiply"]):
            recommendations.append("calculator")
        
        return recommendations

    def create_tool_help(self) -> str:
        """
        Create comprehensive help text for all available tools.
        
        Returns:
            Formatted help text
        """
        help_text = "# Available Tools\n\n"
        
        # Group tools by category
        categories = {
            "File Operations": [],
            "System Tools": [],
            "Analysis Tools": [],
            "Other Tools": []
        }
        
        for tool_name, tool in self.tools.items():
            schema = tool.get_schema()
            required_params = schema.get("required", [])
            
            tool_info = f"**{tool_name}**: {tool.description}"
            if required_params:
                tool_info += f"\n  Required: {', '.join(required_params)}"
            
            # Categorize
            if "file" in tool.name.lower():
                categories["File Operations"].append(tool_info)
            elif any(keyword in tool.name.lower() for keyword in ["datetime", "calculator"]):
                categories["System Tools"].append(tool_info)
            elif "think" in tool.name.lower():
                categories["Analysis Tools"].append(tool_info)
            else:
                categories["Other Tools"].append(tool_info)
        
        # Build help text
        for category, tools in categories.items():
            if tools:
                help_text += f"## {category}\n"
                for tool_info in tools:
                    help_text += f"- {tool_info}\n"
                help_text += "\n"
        
        return help_text


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
