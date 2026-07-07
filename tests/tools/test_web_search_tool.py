"""Tests for the multi-provider web search tool."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from tools.web_search_tool import WebSearchTool


@pytest.fixture
def web_search_tool():
    return WebSearchTool()


def _response(status=200, json_data=None, text_data=None):
    response = AsyncMock()
    response.status = status
    if json_data is not None:
        response.json.return_value = json_data
    if text_data is not None:
        response.text.return_value = text_data
    cm = AsyncMock()
    cm.__aenter__.return_value = response
    return cm


def _fake_session(*, status=200, json_data=None, text_data=None):
    """aiohttp-style session mock; the same response answers .get() and .post()."""
    request_cm = _response(status=status, json_data=json_data, text_data=text_data)
    session = MagicMock()
    session.get.return_value = request_cm
    session.post.return_value = request_cm
    session_cm = AsyncMock()
    session_cm.__aenter__.return_value = session
    return session_cm


def _routed_session(routes):
    """Session mock that picks a response by matching a substring of the URL.

    `routes` is a list of (url_substring, response_cm_factory). Returns a
    factory suitable for `patch.object(..., side_effect=...)`.
    """
    def make():
        def pick(url, *args, **kwargs):
            for needle, factory in routes:
                if needle in url:
                    return factory()
            return _response(status=404, text_data="")
        session = MagicMock()
        session.get.side_effect = pick
        session.post.side_effect = pick
        session_cm = AsyncMock()
        session_cm.__aenter__.return_value = session
        return session_cm
    return make


DDG_HTML = """
<html><body>
  <div class="result">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x">Result A</a>
    <a class="result__snippet">Snippet A about the topic.</a>
  </div>
  <div class="result">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fb">Result B</a>
    <a class="result__snippet">Snippet B.</a>
  </div>
</body></html>
"""


@pytest.mark.asyncio
async def test_duckduckgo_html_parses_real_results(web_search_tool):
    """Keyless path scrapes the HTML endpoint and unwraps redirect URLs."""
    with patch.object(web_search_tool, "_session", return_value=_fake_session(text_data=DDG_HTML)):
        result = await web_search_tool.execute(query="test query")

    assert result["success"] is True
    assert result["provider"] == "ddg_html"
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Result A"
    assert result["results"][0]["link"] == "https://example.com/a"  # uddg-decoded
    assert result["results"][0]["snippet"] == "Snippet A about the topic."


@pytest.mark.asyncio
async def test_max_results_is_honored(web_search_tool):
    many = "".join(
        f'<div class="result"><a class="result__a" '
        f'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F{i}">R{i}</a>'
        f'<a class="result__snippet">S{i}</a></div>'
        for i in range(10)
    )
    markup = f"<html><body>{many}</body></html>"
    with patch.object(web_search_tool, "_session", return_value=_fake_session(text_data=markup)):
        result = await web_search_tool.execute(query="test query", max_results=3)

    assert result["success"] is True
    assert len(result["results"]) == 3


@pytest.mark.asyncio
async def test_falls_through_to_lite_on_202(web_search_tool):
    """A 202 bot-wall on the HTML endpoint should fall through to Lite."""
    lite_markup = (
        '<html><body><table>'
        '<tr><td><a class="result-link" href="https://example.org/x">Lite Result</a></td></tr>'
        '<tr><td class="result-snippet">Lite snippet text.</td></tr>'
        '</table></body></html>'
    )

    make = _routed_session([
        ("html.duckduckgo.com", lambda: _response(status=202, text_data="")),
        ("lite.duckduckgo.com", lambda: _response(text_data=lite_markup)),
    ])

    with patch.object(web_search_tool, "_session", side_effect=make), \
         patch("tools.web_search_tool.asyncio.sleep", new=AsyncMock()):
        result = await web_search_tool.execute(query="test query")

    assert result["success"] is True
    assert result["provider"] == "ddg_lite"
    assert result["results"][0]["link"] == "https://example.org/x"
    assert result["results"][0]["snippet"] == "Lite snippet text."


@pytest.mark.asyncio
async def test_tavily_used_when_key_present(web_search_tool):
    """When a Tavily key is configured, 'auto' tries Tavily first."""
    tavily_json = {
        "answer": "A concise answer.",
        "results": [
            {"title": "T1", "url": "https://t.example/1", "content": "content one"},
        ],
    }
    with patch.object(web_search_tool, "_tavily_key", return_value="tvly-abc"), \
         patch.object(web_search_tool, "_brave_key", return_value=""), \
         patch.object(web_search_tool, "_session", return_value=_fake_session(json_data=tavily_json)):
        result = await web_search_tool.execute(query="test query")

    assert result["success"] is True
    assert result["provider"] == "tavily"
    assert result["answer"] == "A concise answer."
    assert result["results"][0]["link"] == "https://t.example/1"


@pytest.mark.asyncio
async def test_all_providers_fail_returns_error(web_search_tool):
    with patch.object(web_search_tool, "_session", return_value=_fake_session(status=202, text_data="", json_data={})), \
         patch("tools.web_search_tool.asyncio.sleep", new=AsyncMock()):
        result = await web_search_tool.execute(query="test query")

    assert result["success"] is False
    assert "error" in result
    assert result["results"] == []


@pytest.mark.asyncio
async def test_empty_query_rejected(web_search_tool):
    result = await web_search_tool.execute(query="   ")
    assert result["success"] is False
    assert result["results"] == []


def test_decode_ddg_url_unwraps_redirect():
    decode = WebSearchTool._decode_ddg_url
    assert decode("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fp&rut=z") == "https://example.com/p"
    assert decode("https://example.com/direct") == "https://example.com/direct"
    assert decode("//duckduckgo.com/y.js?ad=1") == ""  # internal ad link dropped


def test_web_search_schema(web_search_tool):
    schema = web_search_tool.get_schema()
    assert schema["name"] == "web_search"
    assert "query" in schema["parameters"]["properties"]
    assert "max_results" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["query"]
