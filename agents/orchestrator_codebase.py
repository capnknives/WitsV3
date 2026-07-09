"""Codebase bootstrap and document inventory helpers for the orchestrator."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from core.schemas import StreamData

from agents.routing_classifier import needs_codebase_intro

CODEBASE_BOOTSTRAP_FILES = (
    "README.md",
    "AGENTS.md",
    "docs/architecture/system-architecture.md",
)


class OrchestratorCodebaseMixin:
    """Mixin: pre-load project files and document inventory for ReAct context."""

    async def _bootstrap_codebase_intro(
        self, state: dict[str, Any], session_id: str | None
    ) -> AsyncGenerator[StreamData, None]:
        """Pre-load key project files for codebase introspection goals."""
        if not needs_codebase_intro(state.get("goal", "")):
            return
        if self._has_codebase_file_observation(state.get("observations", [])):
            return

        yield self.stream_action("Reading project structure for codebase overview...")
        try:
            listing = await self._call_tool("list_directory", {"directory_path": "."}, state)
            list_obs = self._format_tool_observation("list_directory", listing)
        except Exception as e:
            list_obs = f"Tool list_directory failed: {e}"

        state["observations"].append(list_obs)
        yield self.stream_observation(list_obs)

        for rel_path in CODEBASE_BOOTSTRAP_FILES:
            yield self.stream_action(f"Reading {rel_path}...")
            try:
                content = await self._call_tool("read_file", {"file_path": rel_path}, state)
                read_obs = self._format_tool_observation("read_file", content)
            except Exception as e:
                read_obs = f"Tool read_file failed: {e}"
            if not read_obs.startswith("Tool read_file result: Error"):
                state["observations"].append(read_obs)
                yield self.stream_observation(read_obs)

    async def _get_document_inventory(self) -> dict[str, int]:
        """File path -> chunk count for every ingested document (empty if none)."""
        if not self.memory_manager:
            return {}
        try:
            segments = await self.memory_manager.get_recent_memory(
                limit=getattr(self.config.orchestrator, "document_inventory_limit", 5000),
                filter_dict={"type": "DOCUMENT_CHUNK"},
            )
        except Exception as e:
            self.logger.warning(f"Could not list ingested documents: {e}")
            return {}
        counts: dict[str, int] = {}
        for seg in segments:
            fp = seg.metadata.get("file_path")
            if fp:
                counts[fp] = counts.get(fp, 0) + 1
        return counts

    @staticmethod
    def _format_documents_context(inventory: dict[str, int]) -> str:
        """Prompt block listing ingested documents the orchestrator can search."""
        if not inventory:
            return "No user documents are currently ingested."
        listing = "\n".join(
            f"- {name} ({count} chunks)" for name, count in sorted(inventory.items())
        )
        return (
            "These user documents are ALREADY ingested and searchable via "
            "document_search. Never claim you cannot access them:\n" + listing
        )
