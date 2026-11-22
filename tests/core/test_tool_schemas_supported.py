"""Every built-in tool's schema must be accepted by the C3a tool-call guard."""

import importlib
import inspect
import pkgutil

import pytest

import tools
from core.base_tool import BaseTool
from core.tool_call_guard import normalise_schema


def _built_in_tools() -> list[BaseTool]:
    """Instantiate each concrete BaseTool under tools/ that needs no arguments."""
    found = []
    for info in pkgutil.walk_packages(tools.__path__, "tools."):
        module = importlib.import_module(info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or inspect.isabstract(cls):
                continue
            if not issubclass(cls, BaseTool):
                continue
            try:
                found.append(cls())
            except TypeError:
                # Reason: tools that need arguments (MCPTool) carry schemas supplied at
                # runtime by their server, so there is no fixed schema to check here.
                continue
    return found


TOOLS = _built_in_tools()


def test_enough_tools_are_checked():
    assert len(TOOLS) >= 20


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: type(t).__name__)
def test_tool_schema_is_supported(tool):
    schema = normalise_schema(tool.get_schema())
    assert schema["type"] == "object" and schema["additionalProperties"] is False
