"""Tools for tiered core memory promotion and archival search."""

from __future__ import annotations

from typing import Any

from core.base_tool import BaseTool


class PromoteToCoreMemoryTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="promote_to_core_memory",
            description="Promote a durable fact into always-in-prompt core memory",
        )

    async def execute(self, fact: str, **kwargs) -> str:
        from core.config import load_config
        from core.core_memory import get_core_memory_store

        config = load_config()
        store = get_core_memory_store(config)
        if store.promote_fact(fact, source="tool"):
            return f"Promoted to core memory: {fact[:200]}"
        return "Fact was not promoted (too short or duplicate)."

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "Short durable fact to remember"},
            },
            "required": ["fact"],
        }


class SearchArchivalMemoryTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="search_archival_memory",
            description="Search long-term archival memory (FAISS/basic backend)",
        )
        self.memory_manager = None

    def set_dependencies(self, config, llm_interface=None, memory_manager=None, **kwargs) -> None:
        self.memory_manager = memory_manager

    async def execute(self, query: str, limit: int = 5, **kwargs) -> str:
        if self.memory_manager is None:
            return "Archival memory is not initialized."
        segments = await self.memory_manager.search_memory(query_text=query, limit=limit)
        if not segments:
            return "No archival memory matches found."
        lines = []
        for seg in segments:
            text = (seg.content.text or "")[:400]
            lines.append(f"- [{seg.type}] {text}")
        return "\n".join(lines)

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }
