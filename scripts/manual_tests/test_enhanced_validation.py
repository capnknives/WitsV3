#!/usr/bin/env python3
"""
Test script for enhanced tool validation in WitsV3.
This demonstrates the new Pydantic-based validation system.
"""

import asyncio
import logging
from typing import Dict, Any
from core.base_tool import BaseTool
from core.schemas import ToolParameter, ToolSchema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestValidationTool(BaseTool):
    """Test tool to demonstrate enhanced validation capabilities."""

    def __init__(self):
        super().__init__(
            name="test_validation",
            description="A test tool to demonstrate enhanced argument validation"
        )

    async def execute(self, name: str, age: int, email: str = "test@example.com", active: bool = True) -> Dict[str, Any]:
        """Execute the test tool with validated arguments."""
        return {
            "success": True,
            "message": f"Processed user {name}, age {age}, email {email}, active: {active}",
            "validated_data": {
                "name": name,
                "age": age,
                "email": email,
                "active": active
            }
        }

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM consumption."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "User name",
                    "minLength": 2,
                    "maxLength": 50
                },
                "age": {
                    "type": "integer",
                    "description": "User age",
                    "minimum": 0,
                    "maximum": 150
                },
                "email": {
                    "type": "string",
                    "description": "User email address",
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                    "default": "test@example.com"
                },
                "active": {
                    "type": "boolean",
                    "description": "Whether the user is active",
                    "default": True
                }
            },
            "required": ["name", "age"]
        }


async def test_enhanced_validation():
    """Test the enhanced validation system."""
    print("🧪 Testing Enhanced Tool Validation System")
    print("=" * 50)

    tool = TestValidationTool()

    # Test 1: Valid arguments
    print("\n1. Testing valid arguments:")
    valid_args = {"name": "John Doe", "age": 30, "email": "john@example.com", "active": True}
    validation_result = tool.validate_arguments(valid_args)

    print(f"   Arguments: {valid_args}")
    print(f"   Valid: {validation_result.valid}")
    print(f"   Errors: {validation_result.errors}")
    print(f"   Warnings: {validation_result.warnings}")
    print(f"   Validated args: {validation_result.validated_arguments}")

    if validation_result.valid:
        result = await tool.execute(**validation_result.validated_arguments)
        print(f"   Execution result: {result}")

    # Test 2: Missing required parameters
    print("\n2. Testing missing required parameters:")
    invalid_args = {"name": "Jane"}  # Missing age
    validation_result = tool.validate_arguments(invalid_args)

    print(f"   Arguments: {invalid_args}")
    print(f"   Valid: {validation_result.valid}")
    print(f"   Errors: {validation_result.errors}")

    # Test 3: Invalid data types
    print("\n3. Testing invalid data types:")
    invalid_args = {"name": "Bob", "age": "thirty"}  # Age should be int
    validation_result = tool.validate_arguments(invalid_args)

    print(f"   Arguments: {invalid_args}")
    print(f"   Valid: {validation_result.valid}")
    print(f"   Errors: {validation_result.errors}")

    # Test 4: Range validation
    print("\n4. Testing range validation:")
    invalid_args = {"name": "Alice", "age": 200}  # Age out of range
    validation_result = tool.validate_arguments(invalid_args)

    print(f"   Arguments: {invalid_args}")
    print(f"   Valid: {validation_result.valid}")
    print(f"   Errors: {validation_result.errors}")

    # Test 5: String length validation
    print("\n5. Testing string length validation:")
    invalid_args = {"name": "A", "age": 25}  # Name too short
    validation_result = tool.validate_arguments(invalid_args)

    print(f"   Arguments: {invalid_args}")
    print(f"   Valid: {validation_result.valid}")
    print(f"   Errors: {validation_result.errors}")

    # Test 6: Unknown parameters (should generate warnings)
    print("\n6. Testing unknown parameters:")
    args_with_unknown = {"name": "Charlie", "age": 35, "unknown_param": "test"}
    validation_result = tool.validate_arguments(args_with_unknown)

    print(f"   Arguments: {args_with_unknown}")
    print(f"   Valid: {validation_result.valid}")
    print(f"   Warnings: {validation_result.warnings}")
    print(f"   Validated args: {validation_result.validated_arguments}")

    # Test 7: Default values
    print("\n7. Testing default values:")
    minimal_args = {"name": "Dave", "age": 40}  # Should use defaults for email and active
    validation_result = tool.validate_arguments(minimal_args)

    print(f"   Arguments: {minimal_args}")
    print(f"   Valid: {validation_result.valid}")
    print(f"   Validated args: {validation_result.validated_arguments}")

    print("\n✅ Enhanced validation testing completed!")


async def test_tool_registry_integration():
    """Test integration with the tool registry."""
    print("\n🔗 Testing Tool Registry Integration")
    print("=" * 50)

    from core.tool_registry import ToolRegistry

    # Create registry and register our test tool
    registry = ToolRegistry()
    test_tool = TestValidationTool()
    registry.register_tool(test_tool)

    print(f"   Registered tools: {registry.list_tool_names()}")

    # Test execution through registry
    print("\n   Testing execution through registry:")    # Valid execution
    try:
        result = await registry.execute_tool_enhanced("test_validation", name="Registry User", age=25)
        print(f"   Valid execution result: {result.success}")
        print(f"   Execution time: {result.execution_time:.3f}s")
        print(f"   Warnings: {result.warnings}")
    except Exception as e:
        print(f"   Error: {e}")

    # Invalid execution
    try:
        result = await registry.execute_tool_enhanced("test_validation", name="A")  # Missing age
        print(f"   Invalid execution result: {result.success}")
        print(f"   Validation errors: {result.validation_errors}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n✅ Tool registry integration testing completed!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_validation())
    asyncio.run(test_tool_registry_integration())
