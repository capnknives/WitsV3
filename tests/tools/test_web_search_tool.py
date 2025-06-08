"""Tests for the web search tool."""
import pytest
from unittest.mock import patch, AsyncMock
from tools.web_search_tool import WebSearchTool

@pytest.fixture
def web_search_tool():
    """Create a WebSearchTool instance for testing."""
    return WebSearchTool()

@pytest.mark.asyncio
async def test_web_search_success(web_search_tool):
    """Test successful web search."""
    mock_response = {
        "RelatedTopics": [
            {
                "Text": "Test Result",
                "FirstURL": "https://example.com"
            }
        ]
    }

    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_context_manager = AsyncMock()
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json.return_value = mock_response
        mock_context_manager.__aenter__.return_value = mock_response_obj
        mock_get.return_value = mock_context_manager

        result = await web_search_tool.execute(query="test query")

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Test Result"
        assert result["results"][0]["link"] == "https://example.com"
        assert result["results"][0]["snippet"] == "Test Result"

@pytest.mark.asyncio
async def test_web_search_error(web_search_tool):
    """Test web search with error."""
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.side_effect = Exception("API Error")

        result = await web_search_tool.execute(query="test query")

        assert result["success"] is False
        assert "error" in result
        assert "API Error" in result["error"]

@pytest.mark.asyncio
async def test_web_search_max_results(web_search_tool):
    """Test web search with max_results parameter."""
    mock_response = {
        "RelatedTopics": [
            {"Text": f"Result {i}", "FirstURL": f"https://example.com/{i}"}
            for i in range(10)
        ]
    }

    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_context_manager = AsyncMock()
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json.return_value = mock_response
        mock_context_manager.__aenter__.return_value = mock_response_obj
        mock_get.return_value = mock_context_manager

        result = await web_search_tool.execute(query="test query", max_results=3)

        assert result["success"] is True
        assert len(result["results"]) == 3

def test_web_search_schema(web_search_tool):
    """Test web search tool schema."""
    schema = web_search_tool.get_schema()

    assert schema["name"] == "web_search"
    assert "query" in schema["parameters"]["properties"]
    assert "max_results" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["query"]
