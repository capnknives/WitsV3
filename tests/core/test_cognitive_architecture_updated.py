"""
Tests for the cognitive architecture module in the WITS Synthetic Brain.

These tests verify the functionality of the cognitive architecture's
integration with various cognitive systems and its processing capabilities.
"""

import asyncio
import pytest
import os
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure we can import from the root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from the updated module
from core.cognitive_architecture_updated import CognitiveState, CognitiveArchitecture
from core.schemas import StreamData


@pytest.fixture
def cognitive_architecture():
    """Fixture for a cognitive architecture instance with mocked dependencies."""
    with patch('core.cognitive_architecture_updated.MemoryHandler', return_value=AsyncMock()) as mock_memory_handler, \
         patch('core.cognitive_architecture_updated.get_enhanced_llm_interface', return_value=AsyncMock()) as mock_llm, \
         patch('core.cognitive_architecture_updated.ToolRegistry', return_value=MagicMock()) as mock_tool_registry, \
         patch('core.cognitive_architecture_updated.KnowledgeGraph', return_value=MagicMock()) as mock_knowledge_graph:

        # Override _load_config method
        CognitiveArchitecture._load_config = MagicMock(return_value={
             "identity": {"name": "TestWITS"},
             "cognitive_modules": {
                 "perception": {"enabled": True, "input_processors": ["text_processor"]},
                 "reasoning": {"enabled": True, "modules": ["deductive_reasoning"]},
                 "metacognition": {"enabled": True}
             }
        })

        # Create cognitive architecture
        architecture = CognitiveArchitecture()

        # Configure mocks
        architecture.memory_handler.remember = AsyncMock(return_value="memory-123")
        architecture.memory_handler.recall = AsyncMock(return_value=[
            {"id": "memory-1", "content": "Relevant memory", "relevance": 0.9}
        ])
        architecture.memory_handler.get_current_context = AsyncMock(return_value={
            "context_id": "context-123",
            "working_memory": {"test": "value"},
            "active_concepts": ["concept1"],
            "recent_memories": [{"id": "memory-1", "summary": "Memory summary"}]
        })

        architecture.llm_interface.generate_text = AsyncMock(
            return_value="Generated reasoning and response text."
        )

        architecture.knowledge_graph.get_active_concepts = MagicMock(
            return_value=["concept1", "concept2"]
        )

        yield architecture


@pytest.mark.asyncio
async def test_process_input(cognitive_architecture):
    """Test processing input through the cognitive architecture."""
    # Process test input
    results = []
    async for result in cognitive_architecture.process("Test input"):
        results.append(result)

    # Verify results structure
    assert len(results) >= 3  # At least thinking, processing, and result

    # Check for expected stream types
    thinking_results = [r for r in results if r.type == "thinking"]
    assert len(thinking_results) >= 1

    result_data = [r for r in results if r.type == "result"]
    assert len(result_data) == 1
    assert result_data[0].content == "Generated reasoning and response text."

    # Verify memory handler was called
    cognitive_architecture.memory_handler.remember.assert_called()
    cognitive_architecture.memory_handler.get_current_context.assert_called()

    # Verify LLM was called
    cognitive_architecture.llm_interface.generate_text.assert_called()


@pytest.mark.asyncio
async def test_perception_module(cognitive_architecture):
    """Test the perception module."""
    # Run perception directly
    perception_result = await cognitive_architecture._run_perception(
        "What is the capital of France?", "process-123"
    )

    # Verify perception result structure
    assert "raw_input" in perception_result
    assert perception_result["raw_input"] == "What is the capital of France?"
    assert "process_id" in perception_result
    assert "timestamp" in perception_result


@pytest.mark.asyncio
async def test_reasoning_module(cognitive_architecture):
    """Test the reasoning module."""
    # Create test inputs
    perception_result = {
        "raw_input": "Test input",
        "intent": "question",
        "domains": ["general"]
    }

    memory_context = {
        "current_context": {"test": "value"},
        "relevant_memories": [{"content": "Relevant memory"}],
        "active_concepts": ["concept1"]
    }

    # Run reasoning directly
    reasoning_result = await cognitive_architecture._run_reasoning(
        "Test input", perception_result, memory_context, "process-123"
    )

    # Verify reasoning result structure
    assert "process_id" in reasoning_result
    assert "timestamp" in reasoning_result
    assert "conclusion" in reasoning_result
    assert reasoning_result["conclusion"] == "Generated reasoning and response text."

    # Verify LLM was called with appropriate prompt
    cognitive_architecture.llm_interface.generate_text.assert_called_once()
    prompt = cognitive_architecture.llm_interface.generate_text.call_args[0][0]
    assert "Test input" in prompt
    assert "Intent: question" in prompt


@pytest.mark.asyncio
async def test_cognitive_state_model():
    """Test the CognitiveState model."""
    # Create a cognitive state
    state = CognitiveState(
        state_id="test-state",
        timestamp=time.time(),
        identity={"name": "TestWITS"},
        active_goals=[{"name": "Answer user question", "priority": 1}],
        current_context={"focus": "user query"},
        reasoning_pathway="deductive",
        attention_focus={"topic": "test topic"}
    )

    # Verify attributes
    assert state.state_id == "test-state"
    assert state.identity["name"] == "TestWITS"
    assert state.active_goals[0]["name"] == "Answer user question"
    assert state.reasoning_pathway == "deductive"
    assert state.attention_focus["topic"] == "test topic"


@pytest.mark.asyncio
async def test_error_handling(cognitive_architecture):
    """Test error handling in the cognitive architecture."""
    # Force an error by making LLM interface raise an exception
    cognitive_architecture.llm_interface.generate_text = AsyncMock(
        side_effect=Exception("Test error")
    )

    # Process input and collect results
    results = []
    async for result in cognitive_architecture.process("Test input"):
        results.append(result)

    # Find error message
    error_results = [r for r in results if r.type == "error"]
    assert len(error_results) == 1
    assert "Test error" in error_results[0].content
    assert error_results[0].error_code == "COGNITIVE_ERROR"
    assert error_results[0].error_category == "execution"
