"""Utilities for summarizing conversation memory segments."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from .llm_interface import BaseLLMInterface
from .memory_manager import MemoryManager, MemorySegment

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """Summarizes conversation history stored in the memory manager."""

    def __init__(self, memory_manager: MemoryManager, llm_interface: BaseLLMInterface) -> None:
        self.memory_manager = memory_manager
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(__name__)

    async def summarize_conversation(
        self,
        time_window_minutes: int | None = None,
        segment_count: int | None = None,
        filter_dict: dict[str, Any] | None = None,
        summary_type: str = "CONVERSATION_SUMMARY",
        max_tokens: int = 2000,
        model: str | None = None,
    ) -> tuple[str, str]:
        """Summarize recent conversation segments and store the summary.

        Returns:
            Tuple of (summary text, id of the stored summary segment)
        """
        segments = await self._get_recent_segments(time_window_minutes, segment_count, filter_dict)
        transcript = await self._create_transcript(segments)
        summary = await self.llm_interface.generate_text(
            prompt=(
                "Please provide a concise summary of the following conversation:\n" f"{transcript}"
            ),
            model=model,
            max_tokens=max_tokens,
        )
        segment_id = await self.memory_manager.add_memory(
            type=summary_type,
            source="conversation_summarizer",
            content_text=summary,
            importance=0.8,
            metadata={
                "summary_timestamp": datetime.now().isoformat(),
                "summarized_segments": [s.id for s in segments],
                "time_window_minutes": time_window_minutes,
            },
        )
        return summary, segment_id

    async def summarize_agent_interaction(
        self,
        agent_name: str,
        time_window_minutes: int | None = None,
        segment_count: int | None = None,
        summary_type: str = "AGENT_INTERACTION_SUMMARY",
        max_tokens: int = 2000,
        model: str | None = None,
    ) -> tuple[str, str]:
        """Summarize interactions with a specific agent.

        Args:
            agent_name: The agent (segment source) whose interactions to summarize

        Returns:
            Tuple of (summary text, id of the stored summary segment)
        """
        filter_dict = {"source": agent_name}
        segments = await self._get_recent_segments(time_window_minutes, segment_count, filter_dict)
        transcript = await self._create_transcript(segments)
        summary = await self.llm_interface.generate_text(
            prompt=(
                f"Please provide a concise summary of the following interactions with "
                f"the agent '{agent_name}':\n{transcript}"
            ),
            model=model,
            max_tokens=max_tokens,
        )
        segment_id = await self.memory_manager.add_memory(
            type=summary_type,
            source="conversation_summarizer",
            content_text=summary,
            importance=0.8,
            metadata={
                "summary_timestamp": datetime.now().isoformat(),
                "summarized_segments": [s.id for s in segments],
                "agent_name": agent_name,
                "time_window_minutes": time_window_minutes,
            },
        )
        return summary, segment_id

    async def summarize_topics(
        self,
        time_window_minutes: int | None = None,
        segment_count: int | None = None,
        filter_dict: dict[str, Any] | None = None,
        summary_type: str = "TOPIC_SUMMARY",
        max_topics: int = 5,
        model: str | None = None,
    ) -> tuple[dict[str, str], list[str]]:
        """Create topic summaries for recent conversation."""
        segments = await self._get_recent_segments(time_window_minutes, segment_count, filter_dict)
        transcript = await self._create_transcript(segments)
        topics = await self._identify_topics(transcript, max_topics, model)
        summaries: dict[str, str] = {}
        ids: list[str] = []
        for topic in topics:
            summary = await self.llm_interface.generate_text(
                prompt=f"Summarize topic '{topic}':\n{transcript}",
                model=model,
                max_tokens=500,
            )
            seg_id = await self.memory_manager.add_memory(
                type=summary_type,
                source="topic_summarizer",
                content_text=summary,
                importance=0.7,
                metadata={"topic": topic, "summary_timestamp": datetime.now().isoformat()},
            )
            summaries[topic] = summary
            ids.append(seg_id)
        return summaries, ids

    async def _get_recent_segments(
        self,
        time_window_minutes: int | None,
        segment_count: int | None,
        filter_dict: dict[str, Any] | None,
    ) -> list[MemorySegment]:
        if time_window_minutes is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)
            segments = await self.memory_manager.get_recent_memory(
                limit=1000, filter_dict=filter_dict
            )
            recent = []
            for s in segments:
                ts = s.timestamp
                # Tolerate naive timestamps from older memory files
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    recent.append(s)
            return recent
        count = segment_count if segment_count is not None else 100
        return await self.memory_manager.get_recent_memory(limit=count, filter_dict=filter_dict)

    async def _create_transcript(self, segments: list[MemorySegment]) -> str:
        """Create a human-readable transcript from memory segments."""
        lines = []
        for seg in sorted(segments, key=lambda s: s.timestamp):
            text = seg.content.text or seg.content.tool_output or ""
            if not text:
                continue

            if seg.type == "USER_INPUT":
                lines.append(f"User: {text}")
            elif seg.type == "AGENT_THOUGHT":
                lines.append(f"{seg.source} (thought): {text}")
            elif seg.type == "TOOL_CALL":
                tool_name = seg.content.tool_name or "tool"
                lines.append(f"{seg.source} (tool call: {tool_name}): {text}")
            elif seg.type == "TOOL_RESPONSE":
                lines.append(f"Tool response: {text}")
            else:
                lines.append(f"{seg.source}: {text}")
        return "\n".join(lines)

    async def _identify_topics(
        self,
        transcript: str,
        max_topics: int = 5,
        model: str | None = None,
    ) -> list[str]:
        """Identify the key topics discussed in a transcript."""
        prompt = f"Identify up to {max_topics} key topics as a JSON list:\n{transcript}"
        response = await self.llm_interface.generate_text(
            prompt=prompt, model=model, max_tokens=200
        )
        try:
            topics = json.loads(response)
            if isinstance(topics, list):
                return [str(t) for t in topics][:max_topics]
        except json.JSONDecodeError:
            pass
        # Fall back to manual extraction for non-JSON responses
        return self._extract_topics_manually(response, max_topics=max_topics)

    def _extract_topics_manually(self, text: str, max_topics: int = 5) -> list[str]:
        """Extract topics from a free-form LLM response.

        Handles numbered lists ("1. Topic"), dashed lists ("- Topic") and
        quoted strings ('"Topic"'). Falls back to ["Conversation"] when no
        recognizable format is found.
        """
        topics: list[str] = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Numbered format: "1. Topic" or "1) Topic"
            match = re.match(r"^\d+[.)]\s*(.+)$", line)
            if match:
                candidate = match.group(1).strip().strip('"').strip()
                if candidate:
                    topics.append(candidate)
                continue

            # Dashed/bulleted format: "- Topic" or "* Topic"
            match = re.match(r"^[-*]\s*(.+)$", line)
            if match:
                candidate = match.group(1).strip().strip('"').strip()
                if candidate:
                    topics.append(candidate)
                continue

            # Quoted format: '"Topic"'
            match = re.match(r'^"(.+)"$', line)
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    topics.append(candidate)
                continue

        if not topics:
            return ["Conversation"]

        return topics[:max_topics]


async def summarize_memory_segment(text: str) -> str:
    """Simplified summary helper used by tests."""
    # In the real system this would call the LLM; for tests we shorten the text.
    return text[:200]
