"""
Tests for the integration between KnowledgeGraph, WorkingMemory, and NeuralWeb components.
"""

import asyncio
import os
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.knowledge_graph import KnowledgeGraph, Entity, Relation
from core.working_memory import WorkingMemory, MemoryItem
from core.neural_web_core import NeuralWeb, ConceptNode, Connection


class MockLLMInterface(BaseLLMInterface):
    """Mock LLM interface for testing."""

    def __init__(self, config: WitsV3Config):
        """Initialize the mock LLM interface."""
        super().__init__(config)

    async def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Mock embedding generation."""
        # Return a simple embedding based on the hash of the text
        hash_value = hash(text) % 10000
        return [hash_value / 10000] * 384  # 384-dim embedding

    async def generate_text(self, prompt: str, **kwargs):
        """Mock text generation."""
        return f"Generated response for: {prompt[:20]}..."

    # Implement other required methods with mock behavior
    async def stream_response(self, prompt: str, **kwargs):
        yield "Mock response"

    async def stream_chat_response(self, messages: List[Dict[str, str]], **kwargs):
        yield "Mock chat response"

    async def get_token_count(self, text: str, **kwargs):
        return 10

    async def stream_text(self, prompt: str, **kwargs):
        """Mock streaming text."""
        yield "Mock streaming text"


@pytest.fixture
def temp_config():
    """Create a temporary configuration for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = WitsV3Config()

        # Set temporary file paths for neural web
        config.memory_manager.neural_web_path = os.path.join(temp_dir, "neural_web.json")

        # Neural web settings
        config.memory_manager.neural_web_settings = type('NeuralWebSettings', (), {
            'activation_threshold': 0.3,
            'decay_rate': 0.1,
            'auto_connect': True,
            'reasoning_patterns': ["modus_ponens", "analogy", "chain", "contradiction"],
            'max_concept_connections': 50,
            'connection_strength_threshold': 0.2
        })

        # Ollama settings
        config.ollama_settings.embedding_model = "llama2"

        yield config


@pytest.fixture
def mock_llm_interface(temp_config):
    """Create a mock LLM interface for testing."""
    return MockLLMInterface(temp_config)


@pytest.fixture
async def neural_web(temp_config, mock_llm_interface):
    """Create a NeuralWeb instance for testing."""
    neural_web = NeuralWeb(
        activation_threshold=temp_config.memory_manager.neural_web_settings.activation_threshold,
        decay_rate=temp_config.memory_manager.neural_web_settings.decay_rate
    )

    # Add some test concepts
    neural_web.concepts["concept1"] = ConceptNode(
        id="concept1",
        content="Python is a programming language",
        concept_type="fact"
    )

    neural_web.concepts["concept2"] = ConceptNode(
        id="concept2",
        content="Programming languages are used to create software",
        concept_type="fact"
    )

    neural_web.concepts["concept3"] = ConceptNode(
        id="concept3",
        content="Software development requires testing",
        concept_type="fact"
    )

    # Add nodes to graph
    neural_web.graph.add_node("concept1")
    neural_web.graph.add_node("concept2")
    neural_web.graph.add_node("concept3")

    # Connect concepts
    await neural_web.connect_concepts(
        source_id="concept1",
        target_id="concept2",
        relationship_type="enables",
        strength=0.8
    )

    await neural_web.connect_concepts(
        source_id="concept2",
        target_id="concept3",
        relationship_type="enables",
        strength=0.7
    )

    return neural_web


@pytest.fixture
async def knowledge_graph(temp_config, mock_llm_interface):
    """Create a KnowledgeGraph instance for testing."""
    kg = KnowledgeGraph(temp_config, mock_llm_interface)

    # Add some test entities
    entity1 = await kg.add_entity(
        name="Python",
        entity_type="programming_language",
        observations=["Python is a high-level programming language", "Python is widely used"]
    )

    entity2 = await kg.add_entity(
        name="Software Development",
        entity_type="process",
        observations=["Software development is the process of creating software", "It involves coding, testing, and deployment"]
    )

    entity3 = await kg.add_entity(
        name="Testing",
        entity_type="activity",
        observations=["Testing verifies software works correctly", "Unit testing checks individual components"]
    )

    # Connect entities
    await kg.add_relation(
        source_id=entity1.id,
        target_id=entity2.id,
        relation_type="usedIn",
        bidirectional=True
    )

    await kg.add_relation(
        source_id=entity2.id,
        target_id=entity3.id,
        relation_type="includes"
    )

    return kg


@pytest.fixture
def working_memory_config():
    """Create working memory configuration for testing."""
    return {
        'max_items': 50,
        'activation_threshold': 0.2,
        'decay_rate': 0.1,
        'default_ttl_seconds': 3600,
        'decay_interval_seconds': 60
    }


@pytest.fixture
async def working_memory(temp_config, mock_llm_interface, knowledge_graph, neural_web, working_memory_config):
    """Create a WorkingMemory instance for testing."""
    memory = WorkingMemory(
        config=temp_config,
        llm_interface=mock_llm_interface,
        knowledge_graph=knowledge_graph,
        neural_web=neural_web
    )

    # Manually set the working memory properties
    memory.max_items = working_memory_config['max_items']
    memory.activation_threshold = working_memory_config['activation_threshold']
    memory.decay_rate = working_memory_config['decay_rate']
    memory.default_ttl = timedelta(seconds=working_memory_config['default_ttl_seconds'])

    return memory


@pytest.mark.asyncio
async def test_working_memory_creation(working_memory):
    """Test that working memory is created successfully."""
    assert working_memory is not None
    assert working_memory.knowledge_graph is not None
    assert working_memory.neural_web is not None
    assert isinstance(working_memory.items, dict)
    assert len(working_memory.items) == 0


@pytest.mark.asyncio
async def test_add_memory_item(working_memory):
    """Test adding items to working memory."""
    # Add a simple memory item
    item = await working_memory.add_item(
        content="Need to learn Python for the project",
        item_type="goal",
        source="user",
        importance=0.8
    )

    assert item.id in working_memory.items
    assert working_memory.items[item.id].content == "Need to learn Python for the project"
    assert working_memory.items[item.id].item_type == "goal"
    assert working_memory.items[item.id].importance == 0.8


@pytest.mark.asyncio
async def test_connect_to_knowledge_graph(working_memory, knowledge_graph):
    """Test connecting memory items to knowledge graph entities."""
    # Add a memory item
    item = await working_memory.add_item(
        content="Python is a great language for data science",
        item_type="fact",
        source="user"
    )

    # Find the Python entity in the knowledge graph
    python_entities = await knowledge_graph.find_entities_by_name("Python", exact_match=True)
    assert len(python_entities) == 1
    python_entity = python_entities[0]

    # Connect memory item to entity
    connected = await working_memory.connect_to_knowledge_graph(item.id, python_entity.id)
    assert connected is True

    # Verify connection
    item = await working_memory.get_item(item.id)
    assert item.entity_id == python_entity.id


@pytest.mark.asyncio
async def test_connect_to_neural_web(working_memory, neural_web):
    """Test connecting memory items to neural web concepts."""
    # Add a memory item
    item = await working_memory.add_item(
        content="I need to learn how to code in Python",
        item_type="goal",
        source="user"
    )

    # Connect to Python concept in neural web
    connected = await working_memory.connect_to_neural_web(item.id, "concept1")
    assert connected is True

    # Verify connection
    item = await working_memory.get_item(item.id)
    assert item.concept_id == "concept1"

    # Check that the concept was activated
    assert neural_web.concepts["concept1"].activation_level > 0


@pytest.mark.asyncio
async def test_search_memory_items(working_memory):
    """Test searching memory items."""
    # Add several memory items
    await working_memory.add_item(
        content="Python is a great language for beginners",
        item_type="fact",
        source="agent"
    )

    await working_memory.add_item(
        content="Software testing is essential for quality",
        item_type="fact",
        source="agent"
    )

    await working_memory.add_item(
        content="The project deadline is next week",
        item_type="reminder",
        source="user",
        importance=0.9
    )

    # Search for Python-related items
    results = await working_memory.search_items("Python programming language")
    assert len(results) > 0

    # First result should be about Python
    assert "Python" in results[0][0].content

    # Search for deadline
    results = await working_memory.search_items("deadline project")
    assert len(results) > 0
    assert "deadline" in results[0][0].content.lower()


@pytest.mark.asyncio
async def test_neural_web_activation_propagation(working_memory, neural_web):
    """Test that activating concepts in the neural web propagates through connections."""
    # Reset activation levels for all concepts to ensure a clean test state
    for concept in neural_web.concepts.values():
        concept.activation_level = 0.0

    # Verify initial state
    assert neural_web.concepts["concept1"].activation_level == 0.0
    assert neural_web.concepts["concept2"].activation_level == 0.0
    assert neural_web.concepts["concept3"].activation_level == 0.0

    # Add a memory item connected to concept1
    item = await working_memory.add_item(
        content="Python is very versatile",
        item_type="fact",
        source="user",
        concept_id="concept1"
    )

    # Activate concept1 through the neural web
    activated = await neural_web.activate_concept("concept1", 1.0)

    # Activation should have propagated to concept2
    assert neural_web.concepts["concept2"].activation_level > 0

    # If concept2 is activated strongly enough, activation might reach concept3
    await neural_web.activate_concept("concept2", 1.0)
    assert neural_web.concepts["concept3"].activation_level > 0


@pytest.mark.asyncio
async def test_knowledge_graph_entity_retrieval(working_memory, knowledge_graph):
    """Test retrieval of entities from knowledge graph via working memory."""
    # Add a memory item connected to a knowledge graph entity
    python_entities = await knowledge_graph.find_entities_by_name("Python", exact_match=True)
    python_entity = python_entities[0]

    item = await working_memory.add_item(
        content="Need to use Python for the project",
        item_type="goal",
        source="user",
        entity_id=python_entity.id
    )

    # Test context summary generation
    summary = await working_memory.get_context_summary()
    assert "Python" in summary

    # Test streaming context
    stream_data = []
    async for data in working_memory.stream_context_thinking():
        stream_data.append(data)

    # There should be mention of the Python entity in the stream
    entity_mentions = [data for data in stream_data if "Python" in data.content]
    assert len(entity_mentions) > 0


@pytest.mark.asyncio
async def test_decay_and_pruning(working_memory):
    """Test that memory items decay over time and are pruned when over capacity."""
    # Add many items to trigger pruning
    for i in range(working_memory.max_items + 10):
        await working_memory.add_item(
            content=f"Test item {i}",
            item_type="fact",
            source="agent",
            importance=0.1 + (i / 100)  # Increasing importance
        )

    # Memory should have been pruned to max capacity
    assert len(working_memory.items) <= working_memory.max_items

    # Test decay
    await working_memory.decay_items()

    # All items should have reduced activation
    for item in working_memory.items.values():
        assert item.activation < 1.0


@pytest.mark.asyncio
async def test_integrated_context_building(working_memory, knowledge_graph, neural_web):
    """Test building an integrated context using all three components."""
    # Add entities in knowledge graph
    project_entity = await knowledge_graph.add_entity(
        name="Project X",
        entity_type="project",
        observations=["A software development project", "Requires Python and testing"]
    )

    # Add concepts in neural web
    neural_web.concepts["project_concept"] = ConceptNode(
        id="project_concept",
        content="Software projects require planning and execution",
        concept_type="pattern"
    )
    neural_web.graph.add_node("project_concept")

    await neural_web.connect_concepts(
        source_id="concept1",  # Python concept
        target_id="project_concept",
        relationship_type="usedIn"
    )

    # Add memory items connected to both
    item1 = await working_memory.add_item(
        content="Project X requires Python expertise",
        item_type="requirement",
        source="user",
        entity_id=project_entity.id
    )

    item2 = await working_memory.add_item(
        content="Need to create a project plan",
        item_type="task",
        source="agent",
        concept_id="project_concept"
    )

    # Relate the items
    await working_memory.relate_items(item1.id, item2.id)

    # Get statistics
    stats = await working_memory.get_statistics()
    assert stats["knowledge_graph_connections"] >= 1
    assert stats["neural_web_connections"] >= 1

    # Test context summary - should include both items
    summary = await working_memory.get_context_summary()
    assert "Project X" in summary
    assert "project plan" in summary
