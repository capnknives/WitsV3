"""
Memory summarization functionality for WitsV3.

This module provides utilities for summarizing memory segments to create
compact representations of conversations and interactions.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from .memory_manager import MemorySegment, MemoryManager, MemorySegmentContent
from .config import WitsV3Config
from .llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)

class ConversationSummarizer:
    """Utility for summarizing conversations stored in memory."""

    def __init__(self, memory_manager: MemoryManager, llm_interface: BaseLLMInterface):
        """Initialize the conversation summarizer.

        Args:
            memory_manager: Memory manager for retrieving and storing segments
            llm_interface: LLM interface for generating summaries
        """
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
        model: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """Summarize a conversation within a time window or segment count.

        Args:
            time_window_minutes: Time window to summarize (last N minutes)
            segment_count: Number of recent segments to summarize
            filter_dict: Optional filter for memory segments
            summary_type: Type to assign to the summary segment
            max_tokens: Maximum tokens for the summary
            model: Optional specific model to use for summarization

        Returns:
            Tuple of (summary_text, summary_segment_id or None if no summary created)
        """
        if time_window_minutes is None and segment_count is None:
            # Default to last 30 minutes
            time_window_minutes = 30

        if time_window_minutes is not None:
            # Get segments from the last N minutes
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            segments = await self._get_segments_since(cutoff_time, filter_dict)
        else:
            # Get the last N segments
            # Use a default value if segment_count is None
            actual_segment_count = segment_count if segment_count is not None else 50
            segments = await self.memory_manager.get_recent_memory(
                limit=actual_segment_count,
                filter_dict=filter_dict
            )

        if not segments:
            self.logger.warning("No segments found to summarize")
            return "No conversation found to summarize.", None

        # Create a conversation transcript
        transcript = await self._create_transcript(segments)

        # Generate summary
        summary = await self._generate_summary(transcript, max_tokens, model)

        # Store the summary as a memory segment
        segment_id = await self.memory_manager.add_memory(
            type=summary_type,
            source="conversation_summarizer",
            content_text=summary,
            importance=0.8,  # Summaries are important
            metadata={
                "summarized_segments": len(segments),
                "time_window_minutes": time_window_minutes,
                "segment_count": segment_count,
                "filter": filter_dict,
                "summary_timestamp": datetime.now().isoformat()
            }
        )

        self.logger.info(f"Created conversation summary with {len(segments)} segments: {segment_id}")
        return summary, segment_id

    async def summarize_agent_interaction(
        self,
        agent_name: str,
        time_window_minutes: Optional[int] = None,
        segment_count: Optional[int] = None,
        summary_type: str = "AGENT_INTERACTION_SUMMARY",
        max_tokens: int = 2000,
        model: Optional[str] = None
    ) -> Tuple[str, str]:
        """Summarize interactions with a specific agent.

        Args:
            agent_name: Name of the agent to summarize interactions with
            time_window_minutes: Time window to summarize (last N minutes)
            segment_count: Number of recent segments to summarize
            summary_type: Type to assign to the summary segment
            max_tokens: Maximum tokens for the summary
            model: Optional specific model to use for summarization

        Returns:
            Tuple of (summary_text, summary_segment_id)
        """
        # Use source filter to get segments from this agent
        filter_dict = {"source": agent_name}

        summary, segment_id = await self.summarize_conversation(
            time_window_minutes=time_window_minutes,
            segment_count=segment_count,
            filter_dict=filter_dict,
            summary_type=summary_type,
            max_tokens=max_tokens,
            model=model
        )

        return summary, segment_id

        async def summarize_topics(
        self,
        time_window_minutes: Optional[int] = None,
        segment_count: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        summary_type: str = "TOPIC_SUMMARY",
        max_topics: int = 5,
        model: Optional[str] = None
    ) -> Tuple[Dict[str, str], List[str]]:
        """Generate topic-based summaries of conversation content.

        Args:
            time_window_minutes: Time window to summarize (last N minutes)
            segment_count: Number of recent segments to summarize
            filter_dict: Optional filter for memory segments
            summary_type: Type to assign to the summary segments
            max_topics: Maximum number of topics to identify
            model: Optional specific model to use for summarization

        Returns:
            Tuple of (topic_summaries_dict, segment_ids)
        """
        if time_window_minutes is None and segment_count is None:
            # Default to last 60 minutes for topic analysis
            time_window_minutes = 60

        if time_window_minutes is not None:
            # Get segments from the last N minutes
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            segments = await self._get_segments_since(cutoff_time, filter_dict)
        else:
            # Get the last N segments
            # Use a default value if segment_count is None
            actual_segment_count = segment_count if segment_count is not None else 100
            segments = await self.memory_manager.get_recent_memory(
                limit=actual_segment_count,
                filter_dict=filter_dict
            )

        if not segments:
            self.logger.warning("No segments found for topic summarization")
            return {}, []

        # Create a conversation transcript
        transcript = await self._create_transcript(segments)

        # Identify topics
        topics = await self._identify_topics(transcript, max_topics, model)

        # Generate a summary for each topic
        topic_summaries = {}
        segment_ids = []

        for topic in topics:
            # Generate summary for this topic
            prompt = f"""
            Based on the following conversation transcript, provide a concise summary of the topic: {topic}

            {transcript}

            Summary of '{topic}':
            """

            summary = await self.llm_interface.generate_text(
                prompt=prompt,
                model=model,
                max_tokens=500  # Shorter summaries for topics
            )

            # Store the topic summary
            segment_id = await self.memory_manager.add_memory(
                type=summary_type,
                source="topic_summarizer",
                content_text=summary,
                importance=0.7,
                metadata={
                    "topic": topic,
                    "summarized_segments": len(segments),
                    "time_window_minutes": time_window_minutes,
                    "segment_count": segment_count,
                    "summary_timestamp": datetime.now().isoformat()
                }
            )

            topic_summaries[topic] = summary
            segment_ids.append(segment_id)

        self.logger.info(f"Created {len(topic_summaries)} topic summaries")
        return topic_summaries, segment_ids

    async def _get_segments_since(
        self,
        cutoff_time: datetime,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        """Get memory segments since a cutoff time.

        Args:
            cutoff_time: Timestamp to filter segments by
            filter_dict: Additional filters to apply

        Returns:
            List of matching MemorySegments
        """
        # Get a large number of recent segments
        segments = await self.memory_manager.get_recent_memory(
            limit=1000,  # Large limit to get all recent segments
            filter_dict=filter_dict
        )

        # Filter by timestamp
        return [s for s in segments if s.timestamp >= cutoff_time]

    async def _create_transcript(self, segments: List[MemorySegment]) -> str:
        """Create a readable transcript from memory segments.

        Args:
            segments: List of memory segments to include in transcript

        Returns:
            Formatted transcript string
        """
        # Sort segments by timestamp
        sorted_segments = sorted(segments, key=lambda s: s.timestamp)

        transcript_lines = []
        for segment in sorted_segments:
            # Format timestamp
            timestamp = segment.timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Get content text or tool output
            content = segment.content.text or segment.content.tool_output or ""
            if not content:
                continue

            # Format based on segment type
            if segment.type == "USER_INPUT":
                transcript_lines.append(f"[{timestamp}] User: {content}")
            elif segment.type == "AGENT_RESPONSE":
                transcript_lines.append(f"[{timestamp}] {segment.source}: {content}")
            elif segment.type == "AGENT_THOUGHT":
                transcript_lines.append(f"[{timestamp}] {segment.source} (thought): {content}")
            elif segment.type == "TOOL_CALL":
                tool_name = segment.content.tool_name or "unknown_tool"
                transcript_lines.append(f"[{timestamp}] {segment.source} used tool '{tool_name}': {content}")
            elif segment.type == "TOOL_RESPONSE":
                transcript_lines.append(f"[{timestamp}] Tool response: {content}")
            else:
                # Generic format for other types
                transcript_lines.append(f"[{timestamp}] {segment.source} ({segment.type}): {content}")

        return "\n".join(transcript_lines)

    async def _generate_summary(
        self,
        transcript: str,
        max_tokens: int = 2000,
        model: Optional[str] = None
    ) -> str:
        """Generate a summary of a conversation transcript.

        Args:
            transcript: Conversation transcript to summarize
            max_tokens: Maximum tokens for the summary
            model: Optional specific model to use for summarization

        Returns:
            Generated summary
        """
        prompt = f"""
        Please provide a comprehensive summary of the following conversation.
        Focus on the main topics discussed, key decisions made, and important information shared.

        {transcript}

        Summary:
        """

        try:
            summary = await self.llm_interface.generate_text(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens
            )
            return summary.strip()
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"

    async def _identify_topics(
        self,
        transcript: str,
        max_topics: int = 5,
        model: Optional[str] = None
    ) -> List[str]:
        """Identify main topics in a conversation transcript.

        Args:
            transcript: Conversation transcript to analyze
            max_topics: Maximum number of topics to identify
            model: Optional specific model to use for topic identification

        Returns:
            List of identified topics
        """
        prompt = f"""
        Please analyze the following conversation and identify the main topics discussed.
        Return a JSON array of topic strings (e.g., ["Topic 1", "Topic 2"]).
        Identify between 1 and {max_topics} topics.

        {transcript}

        Topics (JSON array):
        """

        try:
            response = await self.llm_interface.generate_text(
                prompt=prompt,
                model=model,
                max_tokens=500
            )

            # Try to parse the response as JSON
            try:
                # Extract JSON array if it's within a larger response
                start_idx = response.find('[')
                end_idx = response.rfind(']') + 1

                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    topics = json.loads(json_str)

                    # Validate that we got a list of strings
                    if isinstance(topics, list) and all(isinstance(t, str) for t in topics):
                        return topics[:max_topics]  # Limit to max_topics

                self.logger.warning(f"Could not parse topics from response: {response}")
                # Fall back to manual extraction
                return self._extract_topics_manually(response, max_topics)

            except json.JSONDecodeError:
                self.logger.warning(f"Could not parse topics as JSON: {response}")
                # Fall back to manual extraction
                return self._extract_topics_manually(response, max_topics)

        except Exception as e:
            self.logger.error(f"Error identifying topics: {e}")
            return ["Conversation"]  # Default topic

    def _extract_topics_manually(self, response: str, max_topics: int = 5) -> List[str]:
        """Extract topics from a non-JSON response.

        Args:
            response: LLM response to extract topics from
            max_topics: Maximum number of topics to extract

        Returns:
            List of extracted topics
        """
        topics = []

        # Look for numbered lists like "1. Topic"
        lines = response.split('\n')
        for line in lines:
            line = line.strip()

            # Check for numbered format: "1. Topic"
            if line and (line[0].isdigit() and line[1:].startswith('. ')):
                topic = line[line.find('.')+1:].strip()
                if topic and len(topics) < max_topics:
                    topics.append(topic)

            # Check for quoted format: "Topic"
            elif line.startswith('"') and line.endswith('"') and len(line) > 2:
                topic = line[1:-1].strip()
                if topic and len(topics) < max_topics:
                    topics.append(topic)

            # Check for dash format: "- Topic"
            elif line.startswith('- '):
                topic = line[2:].strip()
                if topic and len(topics) < max_topics:
                    topics.append(topic)

        if not topics:
            # If we couldn't find structured topics, use the whole response
            topics = ["Conversation"]

        return topics
