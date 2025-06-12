#!/usr/bin/env python3
"""
Test suite for Enhanced Reasoning Tools

This test suite validates the enhanced reasoning patterns including
deductive, inductive, and analogical reasoning in the Neural Web.
"""

import pytest
import asyncio
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch

from tools.enhanced_reasoning import (
    BaseReasoningPattern,
    DeductiveReasoning,
    InductiveReasoning,
    AnalogicalReasoning,
    EnhancedReasoningEngine,
    EnhancedReasoningTool
)
from core.config import WitsV3Config
from core.neural_web_core import NeuralWeb, ConceptNode, Connection


class TestBaseReasoningPattern:
    """Test the base reasoning pattern functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    def neural_web(self):
        """Create test Neural Web with sample data."""
        web = NeuralWeb()
        return web

    def test_base_pattern_initialization(self, config, neural_web):
        """Test base reasoning pattern initialization."""
        pattern = BaseReasoningPattern(config, neural_web)

        assert pattern.config == config
        assert pattern.neural_web == neural_web
        assert pattern.confidence_threshold == 0.5

    @pytest.mark.asyncio
    async def test_base_pattern_abstract_method(self, config, neural_web):
        """Test that base pattern raises NotImplementedError."""
        pattern = BaseReasoningPattern(config, neural_web)

        with pytest.raises(NotImplementedError):
            await pattern.reason("test question", {"test": "context"})


class TestDeductiveReasoning:
    """Test deductive reasoning pattern."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    async def neural_web_with_rules(self):
        """Create Neural Web with deductive rules."""
        web = NeuralWeb()

        # Add general rules and specific facts
        await web.add_concept("all_mammals_breathe", "All mammals breathe air", "rule")
        await web.add_concept("humans_are_mammals", "Humans are mammals", "rule")
        await web.add_concept("john_is_human", "John is a human", "fact")

        # Connect the reasoning chain
        await web.connect_concepts("all_mammals_breathe", "humans_are_mammals", "applies_to", 0.9)
        await web.connect_concepts("humans_are_mammals", "john_is_human", "applies_to", 0.95)

        return web

    @pytest.fixture
    def deductive_reasoner(self, config, neural_web_with_rules):
        """Create deductive reasoning instance."""
        return DeductiveReasoning(config, neural_web_with_rules)

    @pytest.mark.asyncio
    async def test_deductive_reasoning_basic(self, deductive_reasoner):
        """Test basic deductive reasoning."""
        question = "Does John breathe air?"
        context = {"subject": "John", "property": "breathing"}

        result = await deductive_reasoner.reason(question, context)

        assert result["reasoning_type"] == "deductive"
        assert result["confidence"] > 0.7
        assert len(result["reasoning_steps"]) > 0
        assert "conclusion" in result

        # Should conclude that John breathes air
        assert "breathe" in result["conclusion"].lower()

    @pytest.mark.asyncio
    async def test_find_applicable_rules(self, deductive_reasoner):
        """Test finding applicable rules for deduction."""
        context = {"subject": "human", "domain": "biology"}

        rules = await deductive_reasoner._find_applicable_rules(context)

        assert len(rules) > 0
        # Should find rules about mammals and humans
        rule_contents = [rule["content"] for rule in rules]
        assert any("mammal" in content.lower() for content in rule_contents)

    @pytest.mark.asyncio
    async def test_apply_modus_ponens(self, deductive_reasoner):
        """Test modus ponens application."""
        premises = [
            {"content": "All mammals breathe air", "type": "rule"},
            {"content": "Humans are mammals", "type": "rule"}
        ]

        conclusion = await deductive_reasoner._apply_modus_ponens(premises)

        assert conclusion is not None
        assert "humans" in conclusion.lower()
        assert "breathe" in conclusion.lower()

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, deductive_reasoner):
        """Test confidence calculation for deductive reasoning."""
        reasoning_steps = [
            {"premise": "All mammals breathe", "confidence": 0.95},
            {"premise": "Humans are mammals", "confidence": 0.9},
            {"conclusion": "Humans breathe", "confidence": 0.85}
        ]

        confidence = deductive_reasoner._calculate_confidence(reasoning_steps)

        assert 0 <= confidence <= 1
        # Should be lower than individual confidences due to chaining
        assert confidence < 0.95


class TestInductiveReasoning:
    """Test inductive reasoning pattern."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    async def neural_web_with_observations(self):
        """Create Neural Web with observations for induction."""
        web = NeuralWeb()

        # Add specific observations
        await web.add_concept("dog1_loyal", "My dog Rex is loyal", "observation")
        await web.add_concept("dog2_loyal", "Neighbor's dog Buddy is loyal", "observation")
        await web.add_concept("dog3_loyal", "Friend's dog Max is loyal", "observation")
        await web.add_concept("dog4_aggressive", "Stray dog was aggressive", "observation")

        # Add entity types
        await web.add_concept("rex_is_dog", "Rex is a dog", "classification")
        await web.add_concept("buddy_is_dog", "Buddy is a dog", "classification")
        await web.add_concept("max_is_dog", "Max is a dog", "classification")

        # Connect observations to entities
        await web.connect_concepts("rex_is_dog", "dog1_loyal", "exhibits", 0.9)
        await web.connect_concepts("buddy_is_dog", "dog2_loyal", "exhibits", 0.9)
        await web.connect_concepts("max_is_dog", "dog3_loyal", "exhibits", 0.9)

        return web

    @pytest.fixture
    def inductive_reasoner(self, config, neural_web_with_observations):
        """Create inductive reasoning instance."""
        return InductiveReasoning(config, neural_web_with_observations)

    @pytest.mark.asyncio
    async def test_inductive_reasoning_basic(self, inductive_reasoner):
        """Test basic inductive reasoning."""
        question = "Are dogs generally loyal?"
        context = {"category": "dogs", "property": "loyalty"}

        result = await inductive_reasoner.reason(question, context)

        assert result["reasoning_type"] == "inductive"
        assert result["confidence"] > 0.6
        assert len(result["observations"]) > 0
        assert "pattern" in result

        # Should identify loyalty pattern in dogs
        assert "loyal" in result["pattern"].lower()

    @pytest.mark.asyncio
    async def test_collect_observations(self, inductive_reasoner):
        """Test observation collection for induction."""
        context = {"category": "dogs", "property": "loyalty"}

        observations = await inductive_reasoner._collect_observations(context)

        assert len(observations) >= 3  # Should find multiple loyalty observations
        for obs in observations:
            assert "content" in obs
            assert "confidence" in obs

    @pytest.mark.asyncio
    async def test_identify_patterns(self, inductive_reasoner):
        """Test pattern identification from observations."""
        observations = [
            {"content": "Rex is loyal", "confidence": 0.9},
            {"content": "Buddy is loyal", "confidence": 0.85},
            {"content": "Max is loyal", "confidence": 0.8}
        ]

        patterns = await inductive_reasoner._identify_patterns(observations)

        assert len(patterns) > 0
        # Should identify loyalty as a common pattern
        loyalty_pattern = next((p for p in patterns if "loyal" in p["pattern"].lower()), None)
        assert loyalty_pattern is not None
        assert loyalty_pattern["strength"] > 0.5

    @pytest.mark.asyncio
    async def test_generalize_from_pattern(self, inductive_reasoner):
        """Test generalization from identified patterns."""
        pattern = {"pattern": "dogs are loyal", "strength": 0.8, "support": 3}

        generalization = await inductive_reasoner._generalize_from_pattern(pattern)

        assert "generalization" in generalization
        assert "confidence" in generalization
        assert "dogs" in generalization["generalization"].lower()
        assert "loyal" in generalization["generalization"].lower()


class TestAnalogicalReasoning:
    """Test analogical reasoning pattern."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    async def neural_web_with_analogies(self):
        """Create Neural Web with analogical structures."""
        web = NeuralWeb()

        # Add source domain (atoms)
        await web.add_concept("atom", "Atom has nucleus and electrons", "concept")
        await web.add_concept("nucleus", "Nucleus is center of atom", "concept")
        await web.add_concept("electrons", "Electrons orbit nucleus", "concept")

        # Add target domain (solar system)
        await web.add_concept("solar_system", "Solar system has sun and planets", "concept")
        await web.add_concept("sun", "Sun is center of solar system", "concept")
        await web.add_concept("planets", "Planets orbit sun", "concept")

        # Create structural relationships
        await web.connect_concepts("atom", "nucleus", "contains", 0.9)
        await web.connect_concepts("nucleus", "electrons", "attracts", 0.8)
        await web.connect_concepts("solar_system", "sun", "contains", 0.9)
        await web.connect_concepts("sun", "planets", "attracts", 0.8)

        return web

    @pytest.fixture
    def analogical_reasoner(self, config, neural_web_with_analogies):
        """Create analogical reasoning instance."""
        return AnalogicalReasoning(config, neural_web_with_analogies)

    @pytest.mark.asyncio
    async def test_analogical_reasoning_basic(self, analogical_reasoner):
        """Test basic analogical reasoning."""
        question = "How is an atom like a solar system?"
        context = {"source_domain": "physics", "target_domain": "astronomy"}

        result = await analogical_reasoner.reason(question, context)

        assert result["reasoning_type"] == "analogical"
        assert result["confidence"] > 0.5
        assert "analogies" in result
        assert len(result["analogies"]) > 0

        # Should find structural similarities
        analogy = result["analogies"][0]
        assert "source" in analogy
        assert "target" in analogy
        assert "similarity" in analogy

    @pytest.mark.asyncio
    async def test_find_structural_similarities(self, analogical_reasoner):
        """Test finding structural similarities between domains."""
        source_concepts = ["atom", "nucleus", "electrons"]
        target_concepts = ["solar_system", "sun", "planets"]

        similarities = await analogical_reasoner._find_structural_similarities(
            source_concepts, target_concepts
        )

        assert len(similarities) > 0
        # Should map nucleus to sun, electrons to planets
        for similarity in similarities:
            assert "source_concept" in similarity
            assert "target_concept" in similarity
            assert "similarity_score" in similarity

    @pytest.mark.asyncio
    async def test_map_relationships(self, analogical_reasoner):
        """Test relationship mapping between analogous domains."""
        source_relations = [("nucleus", "electrons", "attracts")]
        target_relations = [("sun", "planets", "attracts")]

        mappings = await analogical_reasoner._map_relationships(
            source_relations, target_relations
        )

        assert len(mappings) > 0
        mapping = mappings[0]
        assert "source_relation" in mapping
        assert "target_relation" in mapping
        assert "confidence" in mapping

    @pytest.mark.asyncio
    async def test_generate_predictions(self, analogical_reasoner):
        """Test prediction generation from analogies."""
        analogy = {
            "source": "atom",
            "target": "solar_system",
            "mappings": [
                {"source": "nucleus", "target": "sun", "confidence": 0.9},
                {"source": "electrons", "target": "planets", "confidence": 0.8}
            ]
        }

        predictions = await analogical_reasoner._generate_predictions(analogy)

        assert len(predictions) > 0
        for prediction in predictions:
            assert "prediction" in prediction
            assert "confidence" in prediction


class TestEnhancedReasoningEngine:
    """Test the enhanced reasoning engine."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    def neural_web(self):
        """Create test Neural Web."""
        return NeuralWeb()

    @pytest.fixture
    def reasoning_engine(self, config, neural_web):
        """Create enhanced reasoning engine."""
        return EnhancedReasoningEngine(config, neural_web)

    @pytest.mark.asyncio
    async def test_engine_initialization(self, reasoning_engine):
        """Test reasoning engine initialization."""
        assert len(reasoning_engine.reasoning_patterns) == 3
        assert "deductive" in reasoning_engine.reasoning_patterns
        assert "inductive" in reasoning_engine.reasoning_patterns
        assert "analogical" in reasoning_engine.reasoning_patterns

    @pytest.mark.asyncio
    async def test_multi_pattern_reasoning(self, reasoning_engine):
        """Test reasoning with multiple patterns."""
        question = "What can we infer about learning?"
        context = {"domain": "education", "type": "analysis"}

        result = await reasoning_engine.reason_with_multiple_patterns(
            question, context, ["deductive", "inductive"]
        )

        assert "combined_results" in result
        assert "confidence" in result
        assert len(result["pattern_results"]) <= 2

    @pytest.mark.asyncio
    async def test_best_pattern_selection(self, reasoning_engine):
        """Test automatic selection of best reasoning pattern."""
        question = "Are all programming languages object-oriented?"
        context = {"domain": "computer_science", "type": "generalization"}

        pattern = await reasoning_engine._select_best_pattern(question, context)

        # Should select inductive for generalization questions
        assert pattern == "inductive"

    @pytest.mark.asyncio
    async def test_result_synthesis(self, reasoning_engine):
        """Test synthesis of results from multiple patterns."""
        pattern_results = {
            "deductive": {
                "reasoning_type": "deductive",
                "confidence": 0.8,
                "conclusion": "Specific conclusion"
            },
            "inductive": {
                "reasoning_type": "inductive",
                "confidence": 0.7,
                "pattern": "General pattern"
            }
        }

        synthesized = await reasoning_engine._synthesize_results(pattern_results)

        assert "synthesis" in synthesized
        assert "confidence" in synthesized
        assert 0 <= synthesized["confidence"] <= 1


class TestEnhancedReasoningTool:
    """Test the enhanced reasoning tool interface."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    def tool(self, config):
        """Create enhanced reasoning tool."""
        return EnhancedReasoningTool(config)

    @pytest.fixture
    def mock_neural_web(self):
        """Create mock Neural Web."""
        neural_web = MagicMock()
        neural_web.concepts = {}
        neural_web.connections = {}
        return neural_web

    def test_tool_metadata(self, tool):
        """Test tool metadata and schema."""
        assert tool.name == "enhanced_reasoning"
        assert "Enhanced Reasoning" in tool.description

        schema = tool.get_schema()
        assert "type" in schema
        assert "properties" in schema
        assert "question" in schema["properties"]
        assert "reasoning_type" in schema["properties"]

    @pytest.mark.asyncio
    async def test_single_pattern_reasoning(self, tool, mock_neural_web):
        """Test reasoning with a single pattern."""
        with patch.object(tool, '_get_neural_web', return_value=mock_neural_web):
            question = "Are all birds able to fly?"

            result = await tool.execute(
                question=question,
                reasoning_type="inductive",
                context={"domain": "biology"}
            )

            assert result.success
            assert "reasoning_type" in result.result
            assert result.result["reasoning_type"] == "inductive"

    @pytest.mark.asyncio
    async def test_automatic_pattern_selection(self, tool, mock_neural_web):
        """Test automatic reasoning pattern selection."""
        with patch.object(tool, '_get_neural_web', return_value=mock_neural_web):
            question = "If all mammals are warm-blooded and whales are mammals, are whales warm-blooded?"

            result = await tool.execute(
                question=question,
                reasoning_type="auto"
            )

            assert result.success
            # Should automatically select deductive reasoning
            assert result.result["reasoning_type"] == "deductive"

    @pytest.mark.asyncio
    async def test_multi_pattern_reasoning(self, tool, mock_neural_web):
        """Test reasoning with multiple patterns."""
        with patch.object(tool, '_get_neural_web', return_value=mock_neural_web):
            question = "What can we say about intelligence in animals?"

            result = await tool.execute(
                question=question,
                reasoning_type="multi",
                patterns=["deductive", "inductive", "analogical"]
            )

            assert result.success
            assert "combined_results" in result.result
            assert len(result.result["pattern_results"]) > 1

    @pytest.mark.asyncio
    async def test_confidence_filtering(self, tool, mock_neural_web):
        """Test filtering results by confidence threshold."""
        with patch.object(tool, '_get_neural_web', return_value=mock_neural_web):
            question = "Test question"

            result = await tool.execute(
                question=question,
                reasoning_type="inductive",
                min_confidence=0.8
            )

            assert result.success
            # Results should meet minimum confidence threshold
            if "confidence" in result.result:
                assert result.result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_error_handling(self, tool):
        """Test error handling in tool execution."""
        # Test with empty question
        result = await tool.execute(question="", reasoning_type="deductive")
        assert not result.success
        assert "error" in result.error

        # Test with invalid reasoning type
        result = await tool.execute(question="test", reasoning_type="invalid")
        assert not result.success
        assert "error" in result.error


class TestIntegrationEnhancedReasoning:
    """Integration tests for enhanced reasoning tools."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WitsV3Config()

    @pytest.fixture
    async def populated_neural_web(self):
        """Create Neural Web with rich test data."""
        web = NeuralWeb()

        # Add concepts for testing all reasoning types
        await web.add_concept("bird1", "Robin can fly", "observation")
        await web.add_concept("bird2", "Eagle can fly", "observation")
        await web.add_concept("bird3", "Penguin cannot fly", "observation")
        await web.add_concept("all_birds_rule", "All birds have feathers", "rule")
        await web.add_concept("robin_is_bird", "Robin is a bird", "classification")

        # Create connections
        await web.connect_concepts("all_birds_rule", "robin_is_bird", "applies_to", 0.9)
        await web.connect_concepts("robin_is_bird", "bird1", "instance_of", 0.95)

        return web

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(self, config, populated_neural_web):
        """Test the complete reasoning pipeline with real data."""
        reasoning_tool = EnhancedReasoningTool(config)

        with patch.object(reasoning_tool, '_get_neural_web', return_value=populated_neural_web):
            # Test deductive reasoning
            deductive_result = await reasoning_tool.execute(
                question="Does Robin have feathers?",
                reasoning_type="deductive",
                context={"subject": "Robin", "property": "feathers"}
            )

            assert deductive_result.success
            assert deductive_result.result["reasoning_type"] == "deductive"

            # Test inductive reasoning
            inductive_result = await reasoning_tool.execute(
                question="Can most birds fly?",
                reasoning_type="inductive",
                context={"category": "birds", "property": "flight"}
            )

            assert inductive_result.success
            assert inductive_result.result["reasoning_type"] == "inductive"

    @pytest.mark.asyncio
    async def test_reasoning_quality_metrics(self, config, populated_neural_web):
        """Test quality metrics for reasoning results."""
        engine = EnhancedReasoningEngine(config, populated_neural_web)

        question = "What properties do birds have?"
        context = {"domain": "biology", "type": "analysis"}

        result = await engine.reason_with_multiple_patterns(
            question, context, ["deductive", "inductive"]
        )

        # Check quality metrics
        assert result["confidence"] > 0
        assert len(result["pattern_results"]) > 0

        # Each pattern result should have confidence score
        for pattern_name, pattern_result in result["pattern_results"].items():
            assert "confidence" in pattern_result
            assert 0 <= pattern_result["confidence"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
