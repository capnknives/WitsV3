#!/usr/bin/env python3
"""
Test suite for Neural Web NLP Tools

This test suite validates the NLP capabilities for concept extraction,
relationship identification, and domain classification in the Neural Web.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from tools.neural_web_nlp import (
    ConceptExtractor,
    RelationshipExtractor,
    DomainClassifier,
    NeuralWebNLPTool
)
from core.config import WitsV3Config
from core.neural_web_core import NeuralWeb, ConceptNode
from core.memory_manager import MemoryManager


class TestConceptExtractor:
    """Test the core ConceptExtractor functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    def extractor(self, config):
        """Create ConceptExtractor instance."""
        return ConceptExtractor(config)

    @pytest.mark.asyncio
    async def test_extract_concepts_from_text(self, extractor):
        """Test concept extraction from text."""
        text = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms.
        Neural networks are computational models inspired by biological neural networks.
        Deep learning uses multiple layers to model complex patterns in data.
        """

        concepts = await extractor.extract_concepts_from_text(text)

        assert len(concepts) > 0
        assert any("machine learning" in concept["text"].lower() for concept in concepts)
        assert any("artificial intelligence" in concept["text"].lower() for concept in concepts)

        # Verify concept structure
        for concept in concepts:
            assert "id" in concept
            assert "text" in concept
            assert "type" in concept
            assert "confidence" in concept
            assert 0 <= concept["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_pattern_based_extraction(self, extractor):
        """Test pattern-based concept extraction."""
        text = "Python is a programming language. HTTP is a protocol. JSON is a data format."

        concepts = await extractor._pattern_based_extraction(text)

        assert len(concepts) >= 3
        # Should find "Python", "HTTP", "JSON" concepts
        concept_texts = [c["text"] for c in concepts]
        assert "Python" in concept_texts
        assert "HTTP" in concept_texts
        assert "JSON" in concept_texts

    @pytest.mark.asyncio
    async def test_llm_based_extraction(self, extractor):
        """Test LLM-based concept extraction."""
        with patch.object(extractor.llm_interface, 'chat_completion') as mock_llm:
            mock_llm.return_value = {
                "choices": [{
                    "message": {
                        "content": """[
                            {"text": "quantum computing", "type": "technology", "confidence": 0.9},
                            {"text": "superposition", "type": "concept", "confidence": 0.8}
                        ]"""
                    }
                }]
            }

            text = "Quantum computing leverages superposition to process information."
            concepts = await extractor._llm_based_extraction(text)

            assert len(concepts) == 2
            assert concepts[0]["text"] == "quantum computing"
            assert concepts[1]["text"] == "superposition"

    @pytest.mark.asyncio
    async def test_concept_deduplication(self, extractor):
        """Test concept deduplication and merging."""
        concepts = [
            {"text": "AI", "type": "technology", "confidence": 0.8},
            {"text": "artificial intelligence", "type": "technology", "confidence": 0.9},
            {"text": "machine learning", "type": "technology", "confidence": 0.85}
        ]

        deduplicated = await extractor._deduplicate_concepts(concepts)

        # AI and artificial intelligence should be merged
        assert len(deduplicated) == 2
        merged_concept = next((c for c in deduplicated if "artificial intelligence" in c["text"]), None)
        assert merged_concept is not None
        assert merged_concept["confidence"] >= 0.8


class TestRelationshipExtractor:
    """Test the RelationshipExtractor functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    def extractor(self, config):
        """Create RelationshipExtractor instance."""
        return RelationshipExtractor(config)

    @pytest.mark.asyncio
    async def test_extract_relationships(self, extractor):
        """Test relationship extraction between concepts."""
        text = "Machine learning is a subset of artificial intelligence."
        concepts = [
            {"id": "c1", "text": "machine learning", "type": "technology"},
            {"id": "c2", "text": "artificial intelligence", "type": "technology"}
        ]

        relationships = await extractor.extract_relationships(text, concepts)

        assert len(relationships) > 0
        rel = relationships[0]
        assert "source_id" in rel
        assert "target_id" in rel
        assert "relationship_type" in rel
        assert "confidence" in rel
        assert 0 <= rel["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_pattern_based_relationships(self, extractor):
        """Test pattern-based relationship extraction."""
        text = "Python causes better productivity. JavaScript enables web development."
        concepts = [
            {"id": "c1", "text": "Python", "type": "technology"},
            {"id": "c2", "text": "productivity", "type": "concept"},
            {"id": "c3", "text": "JavaScript", "type": "technology"},
            {"id": "c4", "text": "web development", "type": "domain"}
        ]

        relationships = await extractor._pattern_based_relationships(text, concepts)

        assert len(relationships) >= 2

        # Should find "causes" relationship
        causes_rel = next((r for r in relationships if r["relationship_type"] == "causes"), None)
        assert causes_rel is not None

        # Should find "enables" relationship
        enables_rel = next((r for r in relationships if r["relationship_type"] == "enables"), None)
        assert enables_rel is not None

    @pytest.mark.asyncio
    async def test_semantic_relationships(self, extractor):
        """Test semantic relationship detection."""
        with patch.object(extractor.llm_interface, 'chat_completion') as mock_llm:
            mock_llm.return_value = {
                "choices": [{
                    "message": {
                        "content": """[
                            {"source_id": "c1", "target_id": "c2", "relationship_type": "implements", "confidence": 0.9}
                        ]"""
                    }
                }]
            }

            text = "Neural networks implement machine learning algorithms."
            concepts = [
                {"id": "c1", "text": "neural networks", "type": "technology"},
                {"id": "c2", "text": "machine learning", "type": "technology"}
            ]

            relationships = await extractor._semantic_relationships(text, concepts)

            assert len(relationships) == 1
            assert relationships[0]["relationship_type"] == "implements"


class TestDomainClassifier:
    """Test the DomainClassifier functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    def classifier(self, config):
        """Create DomainClassifier instance."""
        return DomainClassifier(config)

    @pytest.mark.asyncio
    async def test_classify_concept_domain(self, classifier):
        """Test domain classification for concepts."""
        concept = {"text": "machine learning algorithm", "type": "technology"}

        domain = await classifier.classify_concept_domain(concept)

        assert "domain" in domain
        assert "confidence" in domain
        assert 0 <= domain["confidence"] <= 1
        assert domain["domain"] in classifier.known_domains

    @pytest.mark.asyncio
    async def test_classify_text_domain(self, classifier):
        """Test domain classification for text."""
        text = "The neural network achieved 95% accuracy on the image classification task."

        domain = await classifier.classify_text_domain(text)

        assert "domain" in domain
        assert "confidence" in domain
        # Should classify as computer science or machine learning
        assert domain["domain"] in ["computer_science", "machine_learning", "technology"]

    def test_keyword_based_classification(self, classifier):
        """Test keyword-based domain classification."""
        concept = {"text": "DNA sequencing", "type": "process"}

        domain = classifier._keyword_based_classification(concept)

        assert domain["domain"] == "biology"
        assert domain["confidence"] > 0

    @pytest.mark.asyncio
    async def test_llm_based_classification(self, classifier):
        """Test LLM-based domain classification."""
        with patch.object(classifier.llm_interface, 'chat_completion') as mock_llm:
            mock_llm.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"domain": "physics", "confidence": 0.92}'
                    }
                }]
            }

            concept = {"text": "quantum entanglement", "type": "phenomenon"}
            domain = await classifier._llm_based_classification(concept)

            assert domain["domain"] == "physics"
            assert domain["confidence"] == 0.92


class TestNeuralWebNLPTool:
    """Test the tool interface for Neural Web NLP."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    def tool(self, config):
        """Create NeuralWebNLPTool instance."""
        return NeuralWebNLPTool(config)

    @pytest.fixture
    def mock_neural_web(self):
        """Create mock Neural Web."""
        neural_web = MagicMock()
        neural_web.add_concept = AsyncMock()
        neural_web.connect_concepts = AsyncMock()
        return neural_web

    def test_tool_metadata(self, tool):
        """Test tool metadata and schema."""
        assert tool.name == "neural_web_nlp"
        assert "Neural Web NLP" in tool.description

        schema = tool.get_schema()
        assert "type" in schema
        assert "properties" in schema
        assert "text" in schema["properties"]

    @pytest.mark.asyncio
    async def test_extract_only_mode(self, tool):
        """Test concept extraction without Neural Web updates."""
        with patch.object(tool, '_get_neural_web') as mock_get_neural_web:
            text = "Python is a programming language."

            result = await tool.execute(text=text, extract_only=True)

            assert result.success
            assert "concepts" in result.result
            assert "relationships" in result.result
            # Should not call neural web methods in extract-only mode
            mock_get_neural_web.assert_not_called()

    @pytest.mark.asyncio
    async def test_neural_web_integration(self, tool, mock_neural_web):
        """Test integration with Neural Web for concept storage."""
        with patch.object(tool, '_get_neural_web', return_value=mock_neural_web):
            with patch.object(tool, '_save_neural_web') as mock_save:
                text = "Machine learning uses neural networks."

                result = await tool.execute(
                    text=text,
                    extract_only=False,
                    update_neural_web=True
                )

                assert result.success
                # Should have called neural web methods
                mock_neural_web.add_concept.assert_called()
                mock_neural_web.connect_concepts.assert_called()
                mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_domain_filtering(self, tool):
        """Test domain-specific concept extraction."""
        text = "Machine learning algorithms process biological data for DNA analysis."

        result = await tool.execute(
            text=text,
            extract_only=True,
            filter_domains=["computer_science"]
        )

        assert result.success
        concepts = result.result["concepts"]
        # Should filter to computer science concepts only
        for concept in concepts:
            domain = concept.get("domain", "")
            if domain:
                assert domain in ["computer_science", "machine_learning", "technology"]

    @pytest.mark.asyncio
    async def test_error_handling(self, tool):
        """Test error handling in tool execution."""
        # Test with empty text
        result = await tool.execute(text="", extract_only=True)
        assert not result.success
        assert "error" in result.error

        # Test with invalid parameters
        result = await tool.execute(text="test", invalid_param=True)
        # Should still work, just ignore invalid params
        assert result.success or "error" in result.error


class TestIntegrationNeuralWebNLP:
    """Integration tests for Neural Web NLP tools."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    def neural_web(self):
        """Create real Neural Web instance for integration testing."""
        return NeuralWeb()

    @pytest.mark.asyncio
    async def test_full_nlp_pipeline(self, config, neural_web):
        """Test the complete NLP pipeline with real Neural Web."""
        # Create NLP tool
        nlp_tool = NeuralWebNLPTool(config)

        # Mock the neural web access methods
        with patch.object(nlp_tool, '_get_neural_web', return_value=neural_web):
            with patch.object(nlp_tool, '_save_neural_web') as mock_save:
                text = """
                Artificial intelligence is a broad field that includes machine learning.
                Machine learning algorithms use neural networks to process data.
                Deep learning is a subset of machine learning using deep neural networks.
                """

                result = await nlp_tool.execute(
                    text=text,
                    extract_only=False,
                    update_neural_web=True,
                    min_confidence=0.6
                )

                assert result.success

                # Verify concepts were added to neural web
                assert len(neural_web.concepts) > 0

                # Verify connections were created
                assert len(neural_web.connections) > 0

                # Check for key concepts
                concept_contents = [concept.content for concept in neural_web.concepts.values()]
                assert any("artificial intelligence" in content.lower() for content in concept_contents)
                assert any("machine learning" in content.lower() for content in concept_contents)

    @pytest.mark.asyncio
    async def test_concept_extraction_accuracy(self, config):
        """Test accuracy of concept extraction on known text."""
        extractor = ConceptExtractor(config)

        text = """
        The Python programming language is widely used in data science.
        TensorFlow and PyTorch are popular machine learning frameworks.
        Natural language processing involves analyzing human language.
        """

        concepts = await extractor.extract_concepts_from_text(text)

        # Should extract key technical concepts
        concept_texts = [c["text"].lower() for c in concepts]
        expected_concepts = ["python", "tensorflow", "pytorch", "machine learning", "natural language processing"]

        found_concepts = sum(1 for expected in expected_concepts
                           if any(expected in text for text in concept_texts))

        # Should find at least 60% of expected concepts
        assert found_concepts >= len(expected_concepts) * 0.6

    @pytest.mark.asyncio
    async def test_relationship_extraction_accuracy(self, config):
        """Test accuracy of relationship extraction."""
        extractor = RelationshipExtractor(config)

        text = "Python enables rapid development. Machine learning requires data preprocessing."
        concepts = [
            {"id": "c1", "text": "Python", "type": "language"},
            {"id": "c2", "text": "rapid development", "type": "process"},
            {"id": "c3", "text": "machine learning", "type": "technology"},
            {"id": "c4", "text": "data preprocessing", "type": "process"}
        ]

        relationships = await extractor.extract_relationships(text, concepts)

        # Should find "enables" and "requires" relationships
        relationship_types = [r["relationship_type"] for r in relationships]
        assert "enables" in relationship_types
        assert "requires" in relationship_types


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
