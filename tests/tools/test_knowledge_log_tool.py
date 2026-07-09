"""Tests for the owner-only knowledge log tools (tools/knowledge_log_tool.py)."""

from __future__ import annotations

import pytest

from core.knowledge_log import KnowledgeLogStore
from tools.knowledge_log_tool import KnowledgeLogAddFactTool, KnowledgeLogSummaryTool


@pytest.mark.asyncio
async def test_summary_tool_blocks_guests():
    tool = KnowledgeLogSummaryTool()
    result = await tool.execute(user_role="guest")
    assert "only available to the owner" in result


@pytest.mark.asyncio
async def test_summary_tool_reports_store_contents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    tool = KnowledgeLogSummaryTool()
    tool.store = KnowledgeLogStore(tmp_path / "knowledge_log.json")
    tool.store.add_fact("Runs on an RTX 3070", source="owner")

    result = await tool.execute(user_role="owner")
    assert "Runs on an RTX 3070" in result


@pytest.mark.asyncio
async def test_add_fact_tool_blocks_guests():
    tool = KnowledgeLogAddFactTool()
    result = await tool.execute(text="secret", user_role="guest")
    assert "only available to the owner" in result


@pytest.mark.asyncio
async def test_add_fact_tool_saves_and_dedupes(tmp_path):
    tool = KnowledgeLogAddFactTool()
    tool.store = KnowledgeLogStore(tmp_path / "knowledge_log.json")

    first = await tool.execute(text="The launch code changed", user_role="owner")
    assert "Saved fact" in first

    second = await tool.execute(text="The launch code changed", user_role="owner")
    assert "already recorded" in second

    data = tool.store._load()
    assert len(data["facts"]) == 1


@pytest.mark.asyncio
async def test_add_fact_tool_rejects_blank_text():
    tool = KnowledgeLogAddFactTool()
    result = await tool.execute(text="   ", user_role="owner")
    assert "No fact text provided" in result
