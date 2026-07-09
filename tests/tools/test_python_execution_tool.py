"""Tests for the Python execution tool."""

from types import SimpleNamespace

import pytest

from tools.python_execution_tool import PythonExecutionTool


@pytest.fixture
def python_execution_tool(monkeypatch):
    """Create a PythonExecutionTool using the in-process sandbox path.

    Reason: config.yaml may set security.sandbox_mode=docker; these unit tests
    assert the legacy restricted-script semantics (strip, stderr, truncate).
    """

    def _fake_load_config(*_a, **_k):
        return SimpleNamespace(
            security=SimpleNamespace(
                sandbox_mode="off",
                python_execution_network_access=False,
                python_execution_subprocess_access=False,
            )
        )

    monkeypatch.setattr("core.config.load_config", _fake_load_config)
    monkeypatch.setattr("tools.python_execution_tool.load_config", _fake_load_config)
    return PythonExecutionTool()


@pytest.mark.asyncio
async def test_python_execution_success(python_execution_tool):
    """Test successful Python code execution."""
    code = "print('Hello, World!')"
    result = await python_execution_tool.execute(code=code)

    assert result["success"] is True
    assert result["output"] == "Hello, World!"  # Tool strips the newline
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
    assert "timed out" in result["error"]  # Actual message is "timed out" not "timeout"
    assert result["return_code"] != 0


@pytest.mark.asyncio
async def test_python_execution_output_limit(python_execution_tool):
    """Test Python code execution with large output."""
    code = "print('x' * 2000000)"  # Generate 2MB of output
    result = await python_execution_tool.execute(code=code)

    assert result["success"] is True
    # Tool truncates and adds "... (output truncated)" so check for that
    assert len(result["output"]) > (
        1024 * 1024
    )  # Will be slightly larger due to truncation message
    assert "output truncated" in result["output"]
    assert result["return_code"] == 0


def test_python_execution_schema(python_execution_tool):
    """Test Python execution tool schema."""
    schema = python_execution_tool.get_schema()

    assert schema["name"] == "python_execute"  # Actual tool name
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
