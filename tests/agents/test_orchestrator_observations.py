"""Tests for ReAct tool observation formatting in BaseOrchestratorAgent."""

from agents.base_orchestrator_agent import BaseOrchestratorAgent


class _ObsHarness(BaseOrchestratorAgent):
    """Minimal concrete subclass for testing format helpers."""

    def _build_reasoning_prompt(self, state):
        return ""

    def _parse_reasoning_response(self, response):
        return {}


def _harness() -> _ObsHarness:
    """Build harness without running full agent __init__ (not needed here)."""
    return _ObsHarness.__new__(_ObsHarness)


def test_web_search_result_uses_numbered_source_format():
    harness = _harness()
    result = {
        "success": True,
        "provider": "tavily",
        "answer": "Oliver Tree died on June 14, 2026.",
        "answer_provider": "tavily",
        "results": [
            {
                "title": "Obituary",
                "snippet": "Oliver Tree passed away...",
                "link": "https://example.com/obit",
            }
        ],
    }
    text = harness._format_tool_observation("web_search", result)
    assert "SOURCES below" in text
    assert "Oliver Tree died" in text
    assert "[1] Obituary" in text
    assert "https://example.com/obit" in text


def test_document_search_result_stays_plain_dict():
    harness = _harness()
    result = {
        "success": True,
        "query": "audit findings",
        "result_count": 1,
        "results": [
            {
                "file": "audit.pdf",
                "chunk": "1/3",
                "relevance": 0.91,
                "text": "Revenue increased 12% year over year.",
            }
        ],
    }
    text = harness._format_tool_observation("document_search", result)
    assert text.startswith("Tool document_search result:")
    assert "audit.pdf" in text
    assert "Revenue increased" in text
    assert "SOURCES below" not in text


def test_mcp_discovery_results_not_formatted_as_web_search():
    harness = _harness()
    result = {
        "success": True,
        "count": 1,
        "results": [
            {
                "name": "io.github.example/postgres",
                "description": "Postgres MCP server",
                "installable": True,
                "command": "npx -y @modelcontextprotocol/server-postgres",
            }
        ],
    }
    text = harness._format_tool_observation("search_mcp_tools", result)
    assert "SOURCES below" not in text
    assert "postgres" in text.lower()


def test_is_web_search_result_requires_provider_or_web_hit_shape():
    assert BaseOrchestratorAgent._is_web_search_result(
        "web_search",
        {"provider": "tavily", "results": []},
    )
    assert not BaseOrchestratorAgent._is_web_search_result(
        "document_search",
        {"query": "x", "results": [{"file": "a.txt", "text": "hello"}]},
    )
