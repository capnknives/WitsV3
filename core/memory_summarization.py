"""Utilities for summarizing conversation memory segments."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .llm_interface import BaseLLMInterface
from .memory_manager import MemoryManager, MemorySegment

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """Simplified conversation summarizer used in tests."""

    def __init__(self, memory_manager: MemoryManager, llm_interface: BaseLLMInterface) -> None:
        self.memory_manager = memory_manager
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(__name__)

    async def summarize_conversation(
        self,
        time_window_minutes: Optional[int] = None,
        segment_count: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        summary_type: str = "CONVERSATION_SUMMARY",
        max_tokens: int = 2000,
        model: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Return a simple summary of recent conversation segments."""
        segments = await self._get_recent_segments(time_window_minutes, segment_count, filter_dict)
        transcript = await self._create_transcript(segments)
        summary = await self.llm_interface.generate_text(
            prompt=f"Summarize:\n{transcript}", model=model, max_tokens=max_tokens
        )
        segment_id = await self.memory_manager.add_memory(
            type=summary_type,
            source="conversation_summarizer",
            content_text=summary,
            importance=0.8,
            metadata={"summary_timestamp": datetime.now().isoformat()},
        )
        return summary, segment_id

    async def summarize_topics(
        self,
        time_window_minutes: Optional[int] = None,
        segment_count: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        summary_type: str = "TOPIC_SUMMARY",
        max_topics: int = 5,
        model: Optional[str] = None,
    ) -> Tuple[Dict[str, str], List[str]]:
        """Create topic summaries for recent conversation."""
        segments = await self._get_recent_segments(time_window_minutes, segment_count, filter_dict)
        transcript = await self._create_transcript(segments)
        topics = await self._identify_topics(transcript, max_topics, model)
        summaries: Dict[str, str] = {}
        ids: List[str] = []
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
        time_window_minutes: Optional[int],
        segment_count: Optional[int],
        filter_dict: Optional[Dict[str, Any]],
    ) -> List[MemorySegment]:
        if time_window_minutes is not None:
            cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
            segments = await self.memory_manager.get_recent_memory(limit=1000, filter_dict=filter_dict)
            return [s for s in segments if s.timestamp >= cutoff]
        count = segment_count if segment_count is not None else 100
        return await self.memory_manager.get_recent_memory(limit=count, filter_dict=filter_dict)

    async def _create_transcript(self, segments: List[MemorySegment]) -> str:
        lines = []
        for seg in sorted(segments, key=lambda s: s.timestamp):
            text = seg.content.text or seg.content.tool_output or ""
            if not text:
                continue
            ts = seg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"[{ts}] {seg.source}: {text}")
        return "\n".join(lines)

    async def _identify_topics(self, transcript: str, max_topics: int, model: Optional[str]) -> List[str]:
        prompt = f"Identify up to {max_topics} key topics as a JSON list:\n{transcript}"
        response = await self.llm_interface.generate_text(prompt=prompt, model=model, max_tokens=200)
        try:
            topics = json.loads(response)
            if isinstance(topics, list):
                return [str(t) for t in topics][:max_topics]
        except json.JSONDecodeError:
            pass
        return ["Conversation"]


async def summarize_memory_segment(text: str) -> str:
    """Simplified summary helper used by tests."""
    # In the real system this would call the LLM; for tests we shorten the text.
    return text[:200]


