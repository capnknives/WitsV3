"""Tests for the Python execution tool."""
import pytest
import asyncio
from tools.python_execution_tool import PythonExecutionTool

@pytest.fixture
def python_execution_tool():
    """Create a PythonExecutionTool instance for testing."""
    return PythonExecutionTool()

@pytest.mark.asyncio
async def test_python_execution_success(python_execution_tool):
    """Test successful Python code execution."""
    code = "print('Hello, World!')"
    result = await python_execution_tool.execute(code=code)
    
    assert result["success"] is True
    assert result["output"] == "Hello, World!\n"
    assert result["error"] == ""
    assert result["return_code"] == 0

@pytest.mark.asyncio
async def test_python_execution_error(python_execution_tool):
    """Test Python code execution with error."""
    code = "print(undefined_variable)"
    result = await python_execution_tool.execute(code=code)
    
    assert result["success"] is False
    assert "NameError" in result["error"]
    assert result["return_code"] != 0

@pytest.mark.asyncio
async def test_python_execution_timeout(python_execution_tool):
    """Test Python code execution timeout."""
    code = "import time; time.sleep(60)"
    result = await python_execution_tool.execute(code=code)
    
    assert result["success"] is False
    assert "timeout" in result["error"].lower()
    assert result["return_code"] != 0

@pytest.mark.asyncio
async def test_python_execution_output_limit(python_execution_tool):
    """Test Python code execution with large output."""
    code = "print('x' * 2000000)"  # Generate 2MB of output
    result = await python_execution_tool.execute(code=code)
    
    assert result["success"] is True
    assert len(result["output"]) <= 1024 * 1024  # Should be truncated to 1MB
    assert result["return_code"] == 0

def test_python_execution_schema(python_execution_tool):
    """Test Python execution tool schema."""
    schema = python_execution_tool.get_schema()
    
    assert schema["name"] == "python_execution"
    assert "code" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["code"]

@pytest.mark.asyncio
async def test_python_execution_imports(python_execution_tool):
    """Test Python code execution with imports."""
    code = """
import math
print(math.pi)
"""
    result = await python_execution_tool.execute(code=code)
    
    assert result["success"] is True
    assert "3.14159" in result["output"]
    assert result["return_code"] == 0

@pytest.mark.asyncio
async def test_python_execution_file_operations(python_execution_tool):
    """Test Python code execution with file operations."""
    code = """
import tempfile
with tempfile.NamedTemporaryFile() as f:
    f.write(b'test')
    print(f.name)
"""
    result = await python_execution_tool.execute(code=code)
    
    assert result["success"] is True
    assert result["return_code"] == 0 