"""
Tests for the cross-domain learning functionality in the Neural Web system.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch

from core.cross_domain_learning import CrossDomainLearning, DomainClassifier
from core.neural_web_core import NeuralWeb, ConceptNode
from core.config import WitsV3Config


@pytest.fixture
def config():
    """Create a test configuration."""
    return WitsV3Config()


@pytest.fixture
def neural_web():
    """Create a test neural web."""
    web = NeuralWeb()

    # Add test concepts from different domains
    concepts = [
        ("c1", "Gravity", "physics", "The force that attracts objects with mass"),
        ("c2", "Democracy", "politics", "A system of government by the whole population"),
        ("c3", "Photosynthesis", "biology", "Process by which plants use sunlight to create energy"),
        ("c4", "Algorithm", "computer_science", "A step-by-step procedure for calculations"),
        ("c5", "Symphony", "music", "An elaborate musical composition for orchestra"),
    ]

    for c_id, concept, domain, desc in concepts:
        node = ConceptNode(
            id=c_id,
            concept=concept,
            metadata={
                "domain": domain,
                "description": desc
            }
        )
        web.add_node(node)

    # Add some connections
    web.connect_nodes("c1", "c2", "related_to", 0.5)
    web.connect_nodes("c3", "c4", "used_in", 0.7)
    web.connect_nodes("c2", "c5", "inspires", 0.6)

    return web


@pytest.fixture
def mock_llm_interface():
    """Create a mock LLM interface for testing."""
    with patch("core.llm_interface.LLMInterface") as mock:
        instance = mock.return_value

        # Mock response for domain classification
        async def mock_generate_domain_response(messages):
            prompt = messages[0].content
            if "Classify" in prompt:
                if "Gravity" in prompt:
                    return MagicMock(content="physics")
                elif "Democracy" in prompt:
                    return MagicMock(content="politics")
                elif "Photosynthesis" in prompt:
                    return MagicMock(content="biology")
                elif "Algorithm" in prompt:
                    return MagicMock(content="computer_science")
                elif "Symphony" in prompt:
                    return MagicMock(content="music")
                else:
                    return MagicMock(content="general")
            elif "similarity" in prompt.lower():
                # Return mock similarity scores for domain pairs
                if "physics" in prompt and "politics" in prompt:
                    return MagicMock(content="0.2")
                elif "biology" in prompt and "computer_science" in prompt:
                    return MagicMock(content="0.6")
                elif "music" in prompt:
                    return MagicMock(content="0.4")
                else:
                    return MagicMock(content="0.5")
            elif "Translate" in prompt:
                # Return translations between domains
                if "physics" in prompt and "politics" in prompt:
                    return MagicMock(content="Power")
                elif "biology" in prompt and "computer_science" in prompt:
                    return MagicMock(content="Data Processing")
                else:
                    return MagicMock(content="Related Concept")
            elif "Find analogies" in prompt:
                # Return analogies
                return MagicMock(content="1, 3")
            elif "description" in prompt.lower():
                return MagicMock(content="Description of domain")
            else:
                return MagicMock(content="Default response")

        instance.generate_response.side_effect = mock_generate_domain_response
        yield instance


@pytest.mark.asyncio
async def test_domain_classifier_initialization(config, mock_llm_interface):
    """Test initialization of the domain classifier."""
    classifier = DomainClassifier(config)
    classifier.llm_interface = mock_llm_interface

    await classifier.initialize()

    # Check that we have embeddings for all known domains
    assert len(classifier.domain_embeddings) == len(classifier.known_domains)
    for domain in classifier.known_domains:
        assert domain in classifier.domain_embeddings


@pytest.mark.asyncio
async def test_domain_classification(config, mock_llm_interface):
    """Test classification of concepts into domains."""
    classifier = DomainClassifier(config)
    classifier.llm_interface = mock_llm_interface

    # Test classification
    domain = await classifier.classify_domain("Gravity affects objects with mass")
    assert domain == "physics"

    domain = await classifier.classify_domain("Democracy is a system of government")
    assert domain == "politics"


@pytest.mark.asyncio
async def test_domain_similarity(config, mock_llm_interface):
    """Test domain similarity calculation."""
    classifier = DomainClassifier(config)
    classifier.llm_interface = mock_llm_interface

    # Test similarity calculations
    similarity = await classifier.get_domain_similarity("physics", "politics")
    assert similarity == 0.2

    similarity = await classifier.get_domain_similarity("biology", "computer_science")
    assert similarity == 0.6


@pytest.mark.asyncio
async def test_cross_domain_learning_initialization(config, neural_web, mock_llm_interface):
    """Test initialization of cross-domain learning."""
    cdl = CrossDomainLearning(config, neural_web)
    cdl.llm_interface = mock_llm_interface
    cdl.domain_classifier.llm_interface = mock_llm_interface

    # Test that the object was created successfully
    assert cdl.neural_web == neural_web
    assert cdl.config == config
    assert cdl.domain_classifier is not None


@pytest.mark.asyncio
async def test_classify_concept_domain(config, neural_web, mock_llm_interface):
    """Test classification of a concept's domain."""
    cdl = CrossDomainLearning(config, neural_web)
    cdl.llm_interface = mock_llm_interface
    cdl.domain_classifier.llm_interface = mock_llm_interface

    # Test with a concept that already has domain metadata
    domain = await cdl.classify_concept_domain("c1")
    assert domain == "physics"

    # Test with a concept that doesn't exist
    domain = await cdl.classify_concept_domain("nonexistent")
    assert domain == "unknown"


@pytest.mark.asyncio
async def test_find_cross_domain_analogies(config, neural_web, mock_llm_interface):
    """Test finding analogies across domains."""
    cdl = CrossDomainLearning(config, neural_web)
    cdl.llm_interface = mock_llm_interface
    cdl.domain_classifier.llm_interface = mock_llm_interface

    # Test finding analogies
    analogies = await cdl.find_cross_domain_analogies("c1", "politics")
    assert len(analogies) > 0

    # Test with nonexistent concept
    analogies = await cdl.find_cross_domain_analogies("nonexistent", "politics")
    assert len(analogies) == 0


@pytest.mark.asyncio
async def test_transfer_knowledge(config, neural_web, mock_llm_interface):
    """Test knowledge transfer between domains."""
    cdl = CrossDomainLearning(config, neural_web)
    cdl.llm_interface = mock_llm_interface
    cdl.domain_classifier.llm_interface = mock_llm_interface

    # Test transferring knowledge
    initial_node_count = len(neural_web.nodes)
    mapping = await cdl.transfer_knowledge("physics", "politics", ["c1"])

    # Check that a new node was created
    assert len(neural_web.nodes) == initial_node_count + 1
    assert "c1" in mapping

    # Check that connections were created
    new_node_id = mapping["c1"]
    source_node = neural_web.get_node("c1")
    assert new_node_id in source_node.connections

    # Check metadata
    new_node = neural_web.get_node(new_node_id)
    assert new_node.metadata["domain"] == "politics"
    assert new_node.metadata["source_concept_id"] == "c1"
    assert new_node.metadata["source_domain"] == "physics"


@pytest.mark.asyncio
async def test_propagate_cross_domain_activation(config, neural_web, mock_llm_interface):
    """Test propagation of activation across domains."""
    cdl = CrossDomainLearning(config, neural_web)
    cdl.llm_interface = mock_llm_interface
    cdl.domain_classifier.llm_interface = mock_llm_interface

    # Create some cross-domain connections for testing
    # Connect physics concept to politics concept
    neural_web.connect_nodes("c1", "c2", "cross_domain", 0.8)

    # Test activation propagation
    activations = await cdl.propagate_cross_domain_activation("c1", 1.0)

    # Should have propagated to c2
    assert "c2" in activations
    assert 0 < activations["c2"] <= 1.0


@pytest.mark.asyncio
async def test_get_domain_concepts(config, neural_web, mock_llm_interface):
    """Test getting concepts from a specific domain."""
    cdl = CrossDomainLearning(config, neural_web)
    cdl.llm_interface = mock_llm_interface

    # Test getting physics concepts
    physics_concepts = await cdl.get_domain_concepts("physics")
    assert "c1" in physics_concepts

    # Test with limit
    limited_concepts = await cdl.get_domain_concepts("physics", limit=1)
    assert len(limited_concepts) <= 1


@pytest.mark.asyncio
async def test_analyze_domain_relationships(config, neural_web, mock_llm_interface):
    """Test analyzing relationships between domains."""
    cdl = CrossDomainLearning(config, neural_web)
    cdl.llm_interface = mock_llm_interface
    cdl.domain_classifier.llm_interface = mock_llm_interface

    # Test domain relationship analysis
    relationships = await cdl.analyze_domain_relationships()

    # Should have relationships between each domain pair
    assert len(relationships) > 0

    # Check a specific relationship
    domains = set()
    for node in neural_web.nodes.values():
        if "domain" in node.metadata:
            domains.add(node.metadata["domain"])

    # Calculate expected number of domain pairs (n choose 2)
    n = len(domains)
    expected_pairs = (n * (n - 1)) // 2

    # Check that we have the right number of relationships
    assert len(relationships) <= expected_pairs


@pytest.mark.asyncio
async def test_create_cross_domain_knowledge_graph(config, neural_web, mock_llm_interface):
    """Test creation of cross-domain knowledge graph."""
    cdl = CrossDomainLearning(config, neural_web)
    cdl.llm_interface = mock_llm_interface
    cdl.domain_classifier.llm_interface = mock_llm_interface

    # Test graph creation
    graph = await cdl.create_cross_domain_knowledge_graph()

    # Should have nodes for each domain
    domains = set()
    for node in neural_web.nodes.values():
        if "domain" in node.metadata:
            domains.add(node.metadata["domain"])

    for domain in domains:
        assert domain in graph.nodes

    # Should have edges between domains
    assert len(graph.edges) > 0


if __name__ == "__main__":
    # Run the tests
    asyncio.run(pytest.main(["-xvs", __file__]))
