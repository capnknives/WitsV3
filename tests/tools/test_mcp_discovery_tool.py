"""Tests for the search_mcp_tools discovery tool."""

from unittest.mock import AsyncMock, patch

import pytest

from tools.mcp_discovery_tool import SearchMCPToolsTool


@pytest.fixture
def tool():
    t = SearchMCPToolsTool()
    t.set_dependencies(config=None)
    return t


@pytest.mark.asyncio
async def test_execute_requires_query(tool):
    result = await tool.execute(query="")
    assert result["success"] is False
    assert result["results"] == []


@pytest.mark.asyncio
async def test_execute_reports_deep_link_to_mcp_page(tool):
    entries = [
        {
            "name": "com.example/slack",
            "description": "Send Slack messages.",
            "repository": "https://github.com/example/slack",
            "install": {
                "command": ["npx", "-y", "slack-mcp@1.0.0"],
                "env_vars": [{"name": "SLACK_TOKEN", "required": True}],
            },
        }
    ]
    with patch(
        "tools.mcp_discovery_tool.search_registry",
        new=AsyncMock(return_value=entries),
    ):
        result = await tool.execute(query="send a slack message", max_results=5)

    assert result["success"] is True
    assert result["count"] == 1
    assert result["deep_link"] == "/mcp?discover=send%20a%20slack%20message"
    assert result["deep_link"] in result["message"]
    assert result["results"][0]["installable"] is True
    assert result["results"][0]["required_env"] == ["SLACK_TOKEN"]


@pytest.mark.asyncio
async def test_execute_handles_registry_failure(tool):
    with patch(
        "tools.mcp_discovery_tool.search_registry",
        new=AsyncMock(side_effect=RuntimeError("registry returned HTTP 503")),
    ):
        result = await tool.execute(query="postgres")

    assert result["success"] is False
    assert "503" in result["error"]
