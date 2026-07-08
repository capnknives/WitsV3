"""Tests for the MCP registry search / command-derivation logic."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from core import mcp_registry_search as reg


def test_build_command_npm_injects_yes_flag():
    pkg = {
        "registryType": "npm",
        "identifier": "@modelcontextprotocol/server-memory",
        "version": "1.2.3",
        "runtimeHint": "npx",
        "transport": {"type": "stdio"},
    }
    cmd = reg.build_stdio_command(pkg)
    assert cmd == ["npx", "-y", "@modelcontextprotocol/server-memory@1.2.3"]


def test_build_command_npm_keeps_existing_yes():
    pkg = {
        "registryType": "npm",
        "identifier": "pkg",
        "version": "0.1.0",
        "runtimeHint": "npx",
        "transport": {"type": "stdio"},
        "runtimeArguments": [{"value": "-y", "type": "positional"}],
    }
    cmd = reg.build_stdio_command(pkg)
    assert cmd == ["npx", "-y", "pkg@0.1.0"]  # not doubled


def test_build_command_pypi_uses_uvx_and_pin():
    pkg = {
        "registryType": "pypi",
        "identifier": "mcp-server-git",
        "version": "2.0.0",
        "transport": {"type": "stdio"},
    }
    assert reg.build_stdio_command(pkg) == ["uvx", "mcp-server-git==2.0.0"]


def test_build_command_skips_placeholder_args():
    pkg = {
        "registryType": "npm",
        "identifier": "fs-server",
        "version": "1.0.0",
        "runtimeHint": "npx",
        "packageArguments": [
            {"type": "positional", "description": "root path"},  # no value → skipped
            {"type": "named", "name": "--mode", "value": "ro"},  # concrete → kept
        ],
    }
    assert reg.build_stdio_command(pkg) == ["npx", "-y", "fs-server@1.0.0", "--mode", "ro"]


def test_build_command_non_stdio_returns_none():
    pkg = {"registryType": "npm", "identifier": "x", "transport": {"type": "sse"}}
    assert reg.build_stdio_command(pkg) is None


def test_build_command_unsupported_registry_returns_none():
    pkg = {"registryType": "oci", "identifier": "some/image", "transport": {"type": "stdio"}}
    assert reg.build_stdio_command(pkg) is None


def test_normalize_picks_first_installable_and_env():
    raw = {
        "server": {
            "name": "com.example/db",
            "description": "Query a database.",
            "version": "1.0.0",
            "repository": {"url": "https://github.com/example/db"},
            "packages": [
                {
                    "registryType": "npm",
                    "identifier": "db-mcp",
                    "version": "1.0.0",
                    "runtimeHint": "npx",
                    "transport": {"type": "stdio"},
                    "environmentVariables": [
                        {"name": "DB_URL", "isRequired": True, "description": "connection"},
                        {"name": "DB_DEBUG", "description": "verbose", "isSecret": False},
                    ],
                }
            ],
        },
        "_meta": {"io.modelcontextprotocol.registry/official": {"isLatest": True}},
    }
    norm = reg.normalize_server(raw)
    assert norm["name"] == "com.example/db"
    assert norm["install"]["command"] == ["npx", "-y", "db-mcp@1.0.0"]
    required = [e for e in norm["install"]["env_vars"] if e["required"]]
    assert [e["name"] for e in required] == ["DB_URL"]


def test_dedupe_keeps_latest_version():
    entries = [
        {"name": "a", "is_latest": False, "install": {}},
        {"name": "a", "is_latest": True, "install": {}},
        {"name": "b", "is_latest": True, "install": {}},
    ]
    deduped = {e["name"]: e for e in reg._dedupe_latest(entries)}
    assert deduped["a"]["is_latest"] is True
    assert len(deduped) == 2


def _fake_session(json_data):
    response = AsyncMock()
    response.status = 200
    response.json.return_value = json_data
    request_cm = AsyncMock()
    request_cm.__aenter__.return_value = response
    session = MagicMock()
    session.get.return_value = request_cm
    session_cm = AsyncMock()
    session_cm.__aenter__.return_value = session
    return session_cm


@pytest.mark.asyncio
async def test_search_registry_normalizes_and_orders_installable_first():
    payload = {
        "servers": [
            {  # remote-only → not installable, should sort last
                "server": {
                    "name": "remote/thing",
                    "description": "hosted",
                    "remotes": [{"type": "sse", "url": "https://x/sse"}],
                    "packages": [],
                },
                "_meta": {"io.modelcontextprotocol.registry/official": {"isLatest": True}},
            },
            {  # installable
                "server": {
                    "name": "local/thing",
                    "description": "local",
                    "packages": [{
                        "registryType": "npm", "identifier": "local-thing",
                        "version": "1.0.0", "runtimeHint": "npx",
                        "transport": {"type": "stdio"},
                    }],
                },
                "_meta": {"io.modelcontextprotocol.registry/official": {"isLatest": True}},
            },
        ]
    }
    with patch("core.mcp_registry_search.aiohttp.ClientSession", return_value=_fake_session(payload)):
        results = await reg.search_registry("thing", limit=10)

    assert [r["name"] for r in results] == ["local/thing", "remote/thing"]
    assert results[0]["install"] is not None
    assert results[1]["install"] is None


def test_candidate_queries_falls_back_to_keywords():
    cands = reg._candidate_queries("send a slack message")
    assert cands[0] == "send a slack message"      # full phrase first
    assert "message" in cands and "slack" in cands  # significant words follow
    assert "a" not in cands                         # stopword/short dropped


@pytest.mark.asyncio
async def test_search_registry_uses_keyword_when_phrase_empty():
    """Full phrase returns nothing → falls back to a single keyword that hits."""
    hit = {
        "servers": [{
            "server": {
                "name": "com.example/slack",
                "description": "Slack integration",
                "packages": [{
                    "registryType": "npm", "identifier": "slack-mcp",
                    "version": "1.0.0", "runtimeHint": "npx",
                    "transport": {"type": "stdio"},
                }],
            },
            "_meta": {"io.modelcontextprotocol.registry/official": {"isLatest": True}},
        }]
    }

    def response_for(url, *, params=None, **kwargs):
        term = (params or {}).get("search", "")
        payload = hit if term == "slack" else {"servers": []}
        response = AsyncMock()
        response.status = 200
        response.json.return_value = payload
        cm = AsyncMock()
        cm.__aenter__.return_value = response
        return cm

    session = MagicMock()
    session.get.side_effect = response_for
    session_cm = AsyncMock()
    session_cm.__aenter__.return_value = session

    with patch("core.mcp_registry_search.aiohttp.ClientSession", return_value=session_cm):
        results = await reg.search_registry("send a slack message", limit=5)

    assert [r["name"] for r in results] == ["com.example/slack"]


@pytest.mark.asyncio
async def test_search_registry_raises_on_http_error():
    response = AsyncMock()
    response.status = 503
    request_cm = AsyncMock()
    request_cm.__aenter__.return_value = response
    session = MagicMock()
    session.get.return_value = request_cm
    session_cm = AsyncMock()
    session_cm.__aenter__.return_value = session

    with patch("core.mcp_registry_search.aiohttp.ClientSession", return_value=session_cm):
        with pytest.raises(RuntimeError, match="503"):
            await reg.search_registry("thing")
