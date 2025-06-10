"""Tests for memory summarization functionality."""

import pytest
import asyncio
from typing import AsyncGenerator, List
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone, timedelta
import json

from core.memory_summarization import ConversationSummarizer
from core.memory_manager import MemorySegment, MemorySegmentContent, MemoryManager
from core.llm_interface import BaseLLMInterface

class DummyLLM(BaseLLMInterface):
    def __init__(self):
        # Don't call super() since we're just using this for testing
        pass

    async def generate_text(self, prompt: str, **kwargs) -> str:
        # For testing, just return a simple summary based on the input
        if "summary" in prompt.lower():
            return "This is a test summary of the conversation."
        elif "topic" in prompt.lower() and "json" in prompt.lower():
            return '["Topic 1", "Topic 2", "Topic 3"]'
        else:
            return "Test response"

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield "dummy"
        yield " stream"

    async def get_embedding(self, text, model=None):
        return [0.1] * 384

@pytest.fixture
def test_segments() -> List[MemorySegment]:
    """Create a list of test memory segments for a conversation."""
    now = datetime.now(timezone.utc)
    return [
        # User messages
        MemorySegment(
            id="user-1",
            timestamp=now - timedelta(minutes=30),
            type="USER_INPUT",
            source="user",
            content=MemorySegmentContent(text="Hello, I need help with my project."),
            importance=0.7
        ),
        MemorySegment(
            id="agent-1",
            timestamp=now - timedelta(minutes=29),
            type="AGENT_RESPONSE",
            source="assistant",
            content=MemorySegmentContent(text="I'd be happy to help with your project. What do you need?"),
            importance=0.6
        ),
        MemorySegment(
            id="user-2",
            timestamp=now - timedelta(minutes=28),
            type="USER_INPUT",
            source="user",
            content=MemorySegmentContent(text="I need to implement a memory system."),
            importance=0.7
        ),
        MemorySegment(
            id="agent-2",
            timestamp=now - timedelta(minutes=27),
            type="AGENT_THOUGHT",
            source="assistant",
            content=MemorySegmentContent(text="User wants to implement a memory system. I should suggest options."),
            importance=0.5
        ),
        MemorySegment(
            id="agent-3",
            timestamp=now - timedelta(minutes=26),
            type="AGENT_RESPONSE",
            source="assistant",
            content=MemorySegmentContent(text="For a memory system, you could use FAISS for vector storage, or a simple JSON-based solution."),
            importance=0.8
        ),
        MemorySegment(
            id="tool-1",
            timestamp=now - timedelta(minutes=25),
            type="TOOL_CALL",
            source="assistant",
            content=MemorySegmentContent(
                text="Looking up documentation",
                tool_name="search_docs",
                tool_args={"query": "memory systems"}
            ),
            importance=0.4
        ),
        MemorySegment(
            id="tool-response-1",
            timestamp=now - timedelta(minutes=24),
            type="TOOL_RESPONSE",
            source="search_docs",
            content=MemorySegmentContent(
                tool_output="Found 3 relevant documents about memory systems"
            ),
            importance=0.4
        ),
        MemorySegment(
            id="user-3",
            timestamp=now - timedelta(minutes=20),
            type="USER_INPUT",
            source="user",
            content=MemorySegmentContent(text="Thanks, I'll try FAISS."),
            importance=0.6
        )
    ]

@pytest.fixture
def mock_memory_manager(test_segments):
    """Create a mock memory manager with test segments."""
    manager = MagicMock(spec=MemoryManager)

    # Mock get_recent_memory to return test segments
    async def mock_get_recent_memory(limit=100, filter_dict=None):
        segments = test_segments

        # Apply filter if provided
        if filter_dict:
            for key, value in filter_dict.items():
                if key == "source":
                    segments = [s for s in segments if s.source == value]
                # Add more filter options as needed

        # Apply limit
        return segments[:limit]

    # Mock add_memory to return a fake segment ID
    async def mock_add_memory(type, source, content_text=None, **kwargs):
        return f"summary-{type}-{datetime.now().timestamp()}"

    manager.get_recent_memory.side_effect = mock_get_recent_memory
    manager.add_memory.side_effect = mock_add_memory

    return manager

@pytest.mark.asyncio
async def test_summarize_conversation(mock_memory_manager):
    """Test summarizing a conversation."""
    llm = DummyLLM()
    summarizer = ConversationSummarizer(mock_memory_manager, llm)

    # Test with default parameters
    summary, segment_id = await summarizer.summarize_conversation()

    assert summary == "This is a test summary of the conversation."
    assert segment_id is not None
    assert segment_id.startswith("summary-CONVERSATION_SUMMARY")

    # Check that add_memory was called with the right parameters
    mock_memory_manager.add_memory.assert_called_once()
    call_args = mock_memory_manager.add_memory.call_args[1]
    assert call_args["type"] == "CONVERSATION_SUMMARY"
    assert call_args["source"] == "conversation_summarizer"
    assert call_args["content_text"] == "This is a test summary of the conversation."
    assert call_args["importance"] == 0.8
    assert "summarized_segments" in call_args["metadata"]
    assert "time_window_minutes" in call_args["metadata"]

@pytest.mark.asyncio
async def test_summarize_agent_interaction(mock_memory_manager):
    """Test summarizing interactions with a specific agent."""
    llm = DummyLLM()
    summarizer = ConversationSummarizer(mock_memory_manager, llm)

    # Test summarizing assistant interactions
    summary, segment_id = await summarizer.summarize_agent_interaction("assistant")

    assert summary == "This is a test summary of the conversation."
    assert segment_id is not None
    assert segment_id.startswith("summary-AGENT_INTERACTION_SUMMARY")

    # Check that filter was applied correctly
    filter_dict = mock_memory_manager.get_recent_memory.call_args[1]["filter_dict"]
    assert filter_dict["source"] == "assistant"

@pytest.mark.asyncio
async def test_summarize_topics(mock_memory_manager):
    """Test summarizing conversation topics."""
    llm = DummyLLM()
    summarizer = ConversationSummarizer(mock_memory_manager, llm)

    # Mock the _identify_topics method to return fixed topics
    with patch.object(summarizer, '_identify_topics',
                    new=AsyncMock(return_value=["Memory Systems", "Project Help"])):

        # Test topic summarization
        topic_summaries, segment_ids = await summarizer.summarize_topics()

        assert len(topic_summaries) == 2
        assert "Memory Systems" in topic_summaries
        assert "Project Help" in topic_summaries
        assert len(segment_ids) == 2

        # Check that add_memory was called with the right parameters
        assert mock_memory_manager.add_memory.call_count == 2

        # Check one of the calls
        for call in mock_memory_manager.add_memory.call_args_list:
            args = call[1]
            assert args["type"] == "TOPIC_SUMMARY"
            assert args["source"] == "topic_summarizer"
            assert args["importance"] == 0.7
            assert "topic" in args["metadata"]

@pytest.mark.asyncio
async def test_create_transcript(mock_memory_manager, test_segments):
    """Test creating a readable transcript from memory segments."""
    llm = DummyLLM()
    summarizer = ConversationSummarizer(mock_memory_manager, llm)

    # Test transcript creation
    transcript = await summarizer._create_transcript(test_segments)

    # Check that transcript contains expected content
    assert "Hello, I need help with my project" in transcript
    assert "User: " in transcript
    assert "assistant: " in transcript
    assert "assistant (thought): " in transcript
    assert "Tool response: " in transcript

    # Check chronological order
    assert transcript.index("Hello, I need help") < transcript.index("I need to implement")
    assert transcript.index("I need to implement") < transcript.index("Thanks, I'll try FAISS")

@pytest.mark.asyncio
async def test_identify_topics():
    """Test identifying topics in a conversation."""
    llm = DummyLLM()
    mock_manager = MagicMock(spec=MemoryManager)
    summarizer = ConversationSummarizer(mock_manager, llm)

    # Test with JSON response
    topics = await summarizer._identify_topics("Test transcript")
    assert topics == ["Topic 1", "Topic 2", "Topic 3"]

    # Test with non-JSON response that needs manual extraction
    with patch.object(llm, 'generate_text',
                    new=AsyncMock(return_value="1. First Topic\n2. Second Topic\n- Third Topic")):
        topics = await summarizer._identify_topics("Test transcript")
        assert "First Topic" in topics
        assert "Second Topic" in topics
        assert "Third Topic" in topics

@pytest.mark.asyncio
async def test_extract_topics_manually():
    """Test manually extracting topics from LLM responses."""
    mock_manager = MagicMock(spec=MemoryManager)
    mock_llm = MagicMock(spec=BaseLLMInterface)
    summarizer = ConversationSummarizer(mock_manager, mock_llm)

    # Test numbered format
    topics = summarizer._extract_topics_manually("1. First Topic\n2. Second Topic")
    assert topics == ["First Topic", "Second Topic"]

    # Test quoted format
    topics = summarizer._extract_topics_manually('"Memory Systems"\n"FAISS Implementation"')
    assert topics == ["Memory Systems", "FAISS Implementation"]

    # Test dash format
    topics = summarizer._extract_topics_manually("- Topic One\n- Topic Two\n- Topic Three")
    assert topics == ["Topic One", "Topic Two", "Topic Three"]

    # Test mixed format
    topics = summarizer._extract_topics_manually('1. First\n"Second"\n- Third')
    assert topics == ["First", "Second", "Third"]

    # Test with limit
    topics = summarizer._extract_topics_manually("1. First\n2. Second\n3. Third\n4. Fourth", max_topics=2)
    assert len(topics) == 2
    assert topics == ["First", "Second"]

    # Test with no recognizable format
    topics = summarizer._extract_topics_manually("This is not in any recognized format.")
    assert topics == ["Conversation"]
