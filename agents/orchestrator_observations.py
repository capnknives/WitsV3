"""Observation formatting helpers for the ReAct orchestrator."""

from __future__ import annotations

from typing import Any


class OrchestratorObservationsMixin:
    """Format tool results into ReAct observations."""

    def _format_tool_observation(self, tool_name: str, result: Any) -> str:
        if isinstance(result, dict) and self._is_web_search_result(tool_name, result):
            return self._format_search_observation(tool_name, result)
        if isinstance(result, dict) and self._is_document_search_result(tool_name, result):
            return self._format_document_observation(tool_name, result)
        return f"Tool {tool_name} result: {result}"

    @staticmethod
    def _is_web_search_result(tool_name: str, result: dict[str, Any]) -> bool:
        results = result.get("results")
        if not isinstance(results, list):
            return False
        if result.get("provider") or result.get("answer_provider"):
            return True
        if results and isinstance(results[0], dict):
            sample = results[0]
            if "link" in sample or "snippet" in sample:
                return tool_name == "web_search"
            if "file" in sample or "text" in sample or "installable" in sample:
                return False
        return False

    @staticmethod
    def _is_document_search_result(tool_name: str, result: dict[str, Any]) -> bool:
        if tool_name != "document_search" or not isinstance(result, dict):
            return False
        if "query" in result or result.get("success") is not None:
            return True
        results = result.get("results")
        return isinstance(results, list) and (
            not results or isinstance(results[0], dict) and "text" in results[0]
        )

    @staticmethod
    def _format_document_observation(tool_name: str, result: dict[str, Any]) -> str:
        lines = [f"{tool_name} results (base your answer on the EXCERPTS below):"]
        if result.get("success") is False:
            lines.append(f"(search failed: {result.get('error', 'unknown error')})")
            return "\n".join(lines)
        excerpts = result.get("results") or []
        if not excerpts:
            lines.append("(no matching passages — try a broader query before giving up)")
            return "\n".join(lines)
        for i, r in enumerate(excerpts, 1):
            file_name = (r.get("file") or "unknown").strip()
            chunk = (r.get("chunk") or "").strip()
            text = (r.get("text") or "").strip()
            rel = r.get("relevance")
            header = f"[{i}] {file_name}"
            if chunk:
                header += f" ({chunk})"
            if rel is not None:
                header += f" relevance={rel}"
            lines.append(f"{header}\n    {text}")
        return "\n".join(lines)

    @staticmethod
    def _format_search_observation(tool_name: str, result: dict[str, Any]) -> str:
        lines = [f"{tool_name} results (base your answer on the SOURCES below):"]
        answer = result.get("answer")
        if answer:
            provider = result.get("answer_provider", "search engine")
            lines.append(
                f"{provider} summary (usually accurate — use it, but trust the "
                f"sources below if any clearly contradicts it): {answer}"
            )
        sources = result.get("results") or []
        max_sources = 5
        max_snippet = 220
        for i, r in enumerate(sources[:max_sources], 1):
            title = (r.get("title") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            if len(snippet) > max_snippet:
                snippet = snippet[:max_snippet] + "…"
            link = (r.get("link") or "").strip()
            lines.append(f"[{i}] {title}\n    {snippet}\n    source: {link}")
        if len(sources) > max_sources:
            lines.append(f"(+ {len(sources) - max_sources} more sources omitted)")
        if not sources and not answer:
            lines.append("(no results found)")
        return "\n".join(lines)
