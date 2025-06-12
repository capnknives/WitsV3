"""
Enhanced Reasoning Patterns for Neural Web

Provides domain-specific reasoning patterns and enhanced cognitive capabilities
for the Neural Web system, enabling more sophisticated problem-solving and
knowledge synthesis across different domains.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import json

from core.base_tool import BaseTool, ToolResult
from core.config import WitsV3Config
from core.neural_web_core import NeuralWeb
from core.llm_interface import BaseLLMInterface, LLMMessage
from core.cross_domain_learning import CrossDomainLearning

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning patterns available."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    ETHICAL = "ethical"
    CREATIVE = "creative"
    SYSTEMATIC = "systematic"


@dataclass
class ReasoningContext:
    """Context for reasoning operations."""
    domain: str
    goal: str
    constraints: List[str]
    available_concepts: List[str]
    confidence_threshold: float = 0.5
    reasoning_depth: int = 3
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""
    reasoning_type: ReasoningType
    conclusion: str
    confidence: float
    reasoning_path: List[str]
    supporting_evidence: List[str]
    assumptions: List[str]
    domain: str
    metadata: Optional[Dict[str, Any]] = None


class BaseReasoningPattern(ABC):
    """Abstract base class for reasoning patterns."""

    def __init__(self, config: WitsV3Config, neural_web: NeuralWeb, llm_interface: BaseLLMInterface):
        self.config = config
        self.neural_web = neural_web
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Execute the reasoning pattern."""
        pass

    @abstractmethod
    def get_reasoning_type(self) -> ReasoningType:
        """Get the type of reasoning this pattern implements."""
        pass

    async def _get_relevant_concepts(self, query: str, limit: int = 10) -> List[str]:
        """Get concepts relevant to the reasoning query."""
        try:
            relevant_concepts = await self.neural_web._find_relevant_concepts(query)
            return relevant_concepts[:limit] if relevant_concepts else []
        except Exception as e:
            self.logger.error(f"Error finding relevant concepts: {e}")
            return []

    async def _activate_concept_network(self, concept_ids: List[str]) -> Dict[str, float]:
        """Activate a network of concepts and return activation levels."""
        activations = {}
        for concept_id in concept_ids:
            try:
                activated = await self.neural_web.activate_concept(concept_id, 0.8)
                for activated_id in activated:
                    if activated_id in self.neural_web.concepts:
                        activations[activated_id] = self.neural_web.concepts[activated_id].activation_level
            except Exception as e:
                self.logger.error(f"Error activating concept {concept_id}: {e}")

        return activations


class DeductiveReasoning(BaseReasoningPattern):
    """Implements deductive reasoning patterns."""

    def get_reasoning_type(self) -> ReasoningType:
        return ReasoningType.DEDUCTIVE

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """
        Implement deductive reasoning: From general principles to specific conclusions.
        Pattern: If A implies B, and A is true, then B is true.
        """
        try:
            # Find general principles (rules, laws, axioms) relevant to the goal
            principles = await self._find_general_principles(context)

            # Apply principles to specific case
            conclusions = []
            reasoning_path = []
            supporting_evidence = []

            for principle in principles:
                if await self._can_apply_principle(principle, context):
                    conclusion = await self._apply_principle(principle, context)
                    if conclusion:
                        conclusions.append(conclusion)
                        reasoning_path.append(f"Applied principle: {principle}")
                        supporting_evidence.append(principle)

            # Synthesize final conclusion
            final_conclusion = await self._synthesize_deductive_conclusions(conclusions, context)
            confidence = self._calculate_deductive_confidence(principles, conclusions)

            return ReasoningResult(
                reasoning_type=ReasoningType.DEDUCTIVE,
                conclusion=final_conclusion,
                confidence=confidence,
                reasoning_path=reasoning_path,
                supporting_evidence=supporting_evidence,
                assumptions=[f"Principle validity in domain: {context.domain}"],
                domain=context.domain,
                metadata={"principles_used": len(principles)}
            )

        except Exception as e:
            self.logger.error(f"Error in deductive reasoning: {e}")
            return self._create_error_result(ReasoningType.DEDUCTIVE, str(e))

    async def _find_general_principles(self, context: ReasoningContext) -> List[str]:
        """Find general principles relevant to the reasoning context."""
        principles = []

        # Look for concepts marked as principles, laws, rules
        for concept_id, concept in self.neural_web.concepts.items():
            if any(keyword in concept.content.lower() for keyword in
                   ['law', 'principle', 'rule', 'axiom', 'theorem', 'theory']):
                if context.domain in concept.metadata.get('domain', ''):
                    principles.append(concept.content)

        return principles

    async def _can_apply_principle(self, principle: str, context: ReasoningContext) -> bool:
        """Check if a principle can be applied to the current context."""
        # Simple heuristic: check if principle keywords match context
        principle_words = set(principle.lower().split())
        context_words = set(context.goal.lower().split())
        overlap = len(principle_words.intersection(context_words))
        return overlap > 0

    async def _apply_principle(self, principle: str, context: ReasoningContext) -> Optional[str]:
        """Apply a principle to derive a conclusion."""
        try:
            prompt = f"""
            Given this principle: {principle}
            And this goal: {context.goal}
            In the domain of: {context.domain}

            What specific conclusion can be deduced? Provide a clear, logical conclusion.
            """

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_interface.generate_response(messages=messages)
            return response.content.strip()

        except Exception as e:
            self.logger.error(f"Error applying principle: {e}")
            return None

    async def _synthesize_deductive_conclusions(self, conclusions: List[str], context: ReasoningContext) -> str:
        """Synthesize multiple conclusions into a final result."""
        if not conclusions:
            return f"No deductive conclusions could be drawn for: {context.goal}"

        if len(conclusions) == 1:
            return conclusions[0]

        try:
            prompt = f"""
            Synthesize these deductive conclusions into a final, coherent conclusion:

            {chr(10).join(f"- {c}" for c in conclusions)}

            Goal: {context.goal}
            Domain: {context.domain}

            Provide a unified conclusion that logically follows from the premises.
            """

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_interface.generate_response(messages=messages)
            return response.content.strip()

        except Exception as e:
            self.logger.error(f"Error synthesizing conclusions: {e}")
            return f"Multiple conclusions derived: {'; '.join(conclusions)}"

    def _calculate_deductive_confidence(self, principles: List[str], conclusions: List[str]) -> float:
        """Calculate confidence in deductive reasoning."""
        base_confidence = 0.8  # Deductive reasoning is generally high confidence

        # Adjust based on number of supporting principles
        principle_bonus = min(0.15, len(principles) * 0.05)

        # Adjust based on consistency of conclusions
        consistency_bonus = 0.1 if len(conclusions) > 1 else 0.0

        # Domain-specific adjustment (some domains have more reliable principles)
        domain_factor = 1.0  # Could be adjusted based on domain reliability

        final_confidence = (base_confidence + principle_bonus + consistency_bonus) * domain_factor
        return min(1.0, max(0.1, final_confidence))

    def _create_error_result(self, reasoning_type: ReasoningType, error_msg: str) -> ReasoningResult:
        """Create an error result for failed reasoning."""
        return ReasoningResult(
            reasoning_type=reasoning_type,
            conclusion=f"Reasoning failed: {error_msg}",
            confidence=0.0,
            reasoning_path=[],
            supporting_evidence=[],
            assumptions=[],
            domain="unknown",
            metadata={"error": True}
        )


class InductiveReasoning(BaseReasoningPattern):
    """Implements inductive reasoning patterns."""

    def get_reasoning_type(self) -> ReasoningType:
        return ReasoningType.INDUCTIVE

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """
        Implement inductive reasoning: From specific observations to general patterns.
        """
        try:
            # Gather specific observations and examples
            observations = await self._gather_observations(context)

            # Identify patterns in the observations
            patterns = await self._identify_patterns(observations, context)

            # Generalize from patterns to broader principles
            generalizations = await self._create_generalizations(patterns, context)

            # Assess strength of inductive inference
            confidence = self._calculate_inductive_confidence(observations, patterns, generalizations)

            # Build reasoning path
            reasoning_path = [
                f"Gathered {len(observations)} observations",
                f"Identified {len(patterns)} patterns",
                f"Generated {len(generalizations)} generalizations"
            ]

            final_conclusion = await self._synthesize_inductive_conclusion(generalizations, context)

            return ReasoningResult(
                reasoning_type=ReasoningType.INDUCTIVE,
                conclusion=final_conclusion,
                confidence=confidence,
                reasoning_path=reasoning_path,
                supporting_evidence=observations,
                assumptions=["Pattern regularity", "Sample representativeness"],
                domain=context.domain,
                metadata={"observations": len(observations), "patterns": len(patterns)}
            )

        except Exception as e:
            self.logger.error(f"Error in inductive reasoning: {e}")
            return self._create_error_result(ReasoningType.INDUCTIVE, str(e))

    async def _gather_observations(self, context: ReasoningContext) -> List[str]:
        """Gather specific observations related to the reasoning goal."""
        observations = []

        # Look for specific instances, examples, cases in the neural web
        for concept_id, concept in self.neural_web.concepts.items():
            if any(keyword in concept.content.lower() for keyword in
                   ['example', 'case', 'instance', 'observation', 'data']):
                if self._is_relevant_to_context(concept.content, context):
                    observations.append(concept.content)

        return observations[:20]  # Limit to prevent overwhelming

    async def _identify_patterns(self, observations: List[str], context: ReasoningContext) -> List[str]:
        """Identify patterns in the observations."""
        if not observations:
            return []

        try:
            prompt = f"""
            Analyze these observations and identify recurring patterns:

            {chr(10).join(f"- {obs}" for obs in observations)}

            Context: {context.goal}
            Domain: {context.domain}

            List the key patterns you observe. Focus on commonalities, trends, and recurring themes.
            """

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_interface.generate_response(messages=messages)

            # Simple parsing - could be enhanced
            patterns = [line.strip('- ').strip() for line in response.content.split('\n')
                       if line.strip() and not line.startswith('#')]

            return patterns[:10]  # Limit patterns

        except Exception as e:
            self.logger.error(f"Error identifying patterns: {e}")
            return []

    async def _create_generalizations(self, patterns: List[str], context: ReasoningContext) -> List[str]:
        """Create generalizations from identified patterns."""
        if not patterns:
            return []

        try:
            prompt = f"""
            Based on these patterns, create general principles or rules:

            {chr(10).join(f"- {pattern}" for pattern in patterns)}

            Context: {context.goal}
            Domain: {context.domain}

            What general principles can be inferred? Express them as broad, applicable rules.
            """

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_interface.generate_response(messages=messages)

            generalizations = [line.strip('- ').strip() for line in response.content.split('\n')
                             if line.strip() and not line.startswith('#')]

            return generalizations[:5]  # Limit generalizations

        except Exception as e:
            self.logger.error(f"Error creating generalizations: {e}")
            return []

    def _is_relevant_to_context(self, content: str, context: ReasoningContext) -> bool:
        """Check if content is relevant to the reasoning context."""
        content_words = set(content.lower().split())
        goal_words = set(context.goal.lower().split())
        return len(content_words.intersection(goal_words)) > 0

    def _calculate_inductive_confidence(self, observations: List[str],
                                      patterns: List[str],
                                      generalizations: List[str]) -> float:
        """Calculate confidence in inductive reasoning."""
        base_confidence = 0.6  # Inductive reasoning is inherently less certain

        # More observations increase confidence
        observation_bonus = min(0.2, len(observations) * 0.01)

        # Clear patterns increase confidence
        pattern_bonus = min(0.15, len(patterns) * 0.03)

        # Successful generalizations increase confidence
        generalization_bonus = min(0.1, len(generalizations) * 0.02)

        final_confidence = base_confidence + observation_bonus + pattern_bonus + generalization_bonus
        return min(1.0, max(0.1, final_confidence))

    async def _synthesize_inductive_conclusion(self, generalizations: List[str],
                                             context: ReasoningContext) -> str:
        """Synthesize generalizations into a final conclusion."""
        if not generalizations:
            return f"Insufficient data to draw inductive conclusions about: {context.goal}"

        if len(generalizations) == 1:
            return f"Based on observed patterns: {generalizations[0]}"

        try:
            prompt = f"""
            Synthesize these inductive generalizations into a coherent conclusion:

            {chr(10).join(f"- {g}" for g in generalizations)}

            Goal: {context.goal}
            Domain: {context.domain}

            Provide a unified conclusion that captures the essence of the inductive reasoning.
            """

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_interface.generate_response(messages=messages)
            return response.content.strip()

        except Exception as e:
            self.logger.error(f"Error synthesizing inductive conclusion: {e}")
            return f"Multiple patterns suggest: {'; '.join(generalizations)}"


class AnalogicalReasoning(BaseReasoningPattern):
    """Implements analogical reasoning patterns."""

    def get_reasoning_type(self) -> ReasoningType:
        return ReasoningType.ANALOGICAL

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """
        Implement analogical reasoning: Transfer knowledge from similar situations.
        """
        try:
            # Find analogous situations or domains
            analogies = await self._find_analogies(context)

            # Map relationships between source and target
            mappings = await self._create_analogical_mappings(analogies, context)

            # Transfer insights from analogies
            insights = await self._transfer_insights(mappings, context)

            # Generate conclusion based on analogical reasoning
            conclusion = await self._generate_analogical_conclusion(insights, context)

            confidence = self._calculate_analogical_confidence(analogies, mappings, insights)

            reasoning_path = [
                f"Found {len(analogies)} relevant analogies",
                f"Created {len(mappings)} analogical mappings",
                f"Transferred {len(insights)} insights"
            ]

            return ReasoningResult(
                reasoning_type=ReasoningType.ANALOGICAL,
                conclusion=conclusion,
                confidence=confidence,
                reasoning_path=reasoning_path,
                supporting_evidence=[f"Analogy: {a}" for a in analogies],
                assumptions=["Structural similarity", "Knowledge transferability"],
                domain=context.domain,
                metadata={"analogies_used": len(analogies)}
            )

        except Exception as e:
            self.logger.error(f"Error in analogical reasoning: {e}")
            return self._create_error_result(ReasoningType.ANALOGICAL, str(e))

    async def _find_analogies(self, context: ReasoningContext) -> List[str]:
        """Find analogous situations or concepts."""
        analogies = []

        # Use neural web's analogical reasoning capability
        try:
            relevant_concepts = await self._get_relevant_concepts(context.goal, 15)
            reasoning_result = await self.neural_web.reason(context.goal, 'analogy')

            if 'results' in reasoning_result:
                for result in reasoning_result['results']:
                    if 'source' in result and 'target' in result:
                        analogy = f"{result['source']} is analogous to {result['target']}"
                        analogies.append(analogy)

        except Exception as e:
            self.logger.error(f"Error finding analogies: {e}")

        return analogies[:10]

    async def _create_analogical_mappings(self, analogies: List[str], context: ReasoningContext) -> List[Dict[str, Any]]:
        """Create mappings between analogical elements."""
        mappings = []

        for analogy in analogies:
            try:
                prompt = f"""
                For this analogy: {analogy}
                In the context of: {context.goal}
                Domain: {context.domain}

                Create a mapping showing:
                1. Source domain elements
                2. Target domain elements
                3. Relationship mappings
                4. Key similarities

                Format as structured text.
                """

                messages = [LLMMessage(role="user", content=prompt)]
                response = await self.llm_interface.generate_response(messages=messages)

                mappings.append({
                    "analogy": analogy,
                    "mapping": response.content.strip(),
                    "confidence": 0.7  # Default confidence
                })

            except Exception as e:
                self.logger.error(f"Error creating mapping for analogy {analogy}: {e}")

        return mappings

    async def _transfer_insights(self, mappings: List[Dict[str, Any]], context: ReasoningContext) -> List[str]:
        """Transfer insights from analogical mappings."""
        insights = []

        for mapping in mappings:
            try:
                prompt = f"""
                Based on this analogical mapping:
                {mapping['mapping']}

                What insights can be transferred to solve: {context.goal}
                In domain: {context.domain}

                Provide specific, actionable insights.
                """

                messages = [LLMMessage(role="user", content=prompt)]
                response = await self.llm_interface.generate_response(messages=messages)

                # Extract insights from response
                insight_lines = [line.strip() for line in response.content.split('\n')
                               if line.strip() and not line.startswith('#')]
                insights.extend(insight_lines[:3])  # Limit insights per mapping

            except Exception as e:
                self.logger.error(f"Error transferring insights: {e}")

        return insights[:10]  # Limit total insights

    async def _generate_analogical_conclusion(self, insights: List[str], context: ReasoningContext) -> str:
        """Generate conclusion from analogical insights."""
        if not insights:
            return f"No analogical insights found for: {context.goal}"

        try:
            prompt = f"""
            Synthesize these analogical insights into a conclusion:

            {chr(10).join(f"- {insight}" for insight in insights)}

            Goal: {context.goal}
            Domain: {context.domain}

            What conclusion can be drawn from these analogical insights?
            """

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_interface.generate_response(messages=messages)
            return response.content.strip()

        except Exception as e:
            self.logger.error(f"Error generating analogical conclusion: {e}")
            return f"Analogical insights suggest: {'; '.join(insights[:3])}"

    def _calculate_analogical_confidence(self, analogies: List[str],
                                       mappings: List[Dict[str, Any]],
                                       insights: List[str]) -> float:
        """Calculate confidence in analogical reasoning."""
        base_confidence = 0.5  # Analogical reasoning has moderate base confidence

        # More analogies increase confidence
        analogy_bonus = min(0.2, len(analogies) * 0.02)

        # Quality mappings increase confidence
        mapping_bonus = min(0.15, len(mappings) * 0.03)

        # Actionable insights increase confidence
        insight_bonus = min(0.15, len(insights) * 0.02)

        final_confidence = base_confidence + analogy_bonus + mapping_bonus + insight_bonus
        return min(1.0, max(0.1, final_confidence))


class EnhancedReasoningEngine:
    """Main engine for enhanced reasoning patterns."""

    def __init__(self, config: WitsV3Config, neural_web: NeuralWeb,
                 llm_interface: BaseLLMInterface, cross_domain_learning: Optional[CrossDomainLearning] = None):
        self.config = config
        self.neural_web = neural_web
        self.llm_interface = llm_interface
        self.cross_domain_learning = cross_domain_learning
        self.logger = logging.getLogger(__name__)

        # Initialize reasoning patterns
        self.reasoning_patterns = {
            ReasoningType.DEDUCTIVE: DeductiveReasoning(config, neural_web, llm_interface),
            ReasoningType.INDUCTIVE: InductiveReasoning(config, neural_web, llm_interface),
            ReasoningType.ANALOGICAL: AnalogicalReasoning(config, neural_web, llm_interface),
        }

    async def reason(self, goal: str, domain: str,
                   reasoning_types: Optional[List[ReasoningType]] = None,
                   **kwargs) -> List[ReasoningResult]:
        """
        Execute enhanced reasoning using multiple patterns.

        Args:
            goal: The reasoning goal or question
            domain: The knowledge domain
            reasoning_types: Specific reasoning types to use (default: all available)
            **kwargs: Additional reasoning parameters

        Returns:
            List of reasoning results from different patterns
        """
        try:
            # Create reasoning context
            context = ReasoningContext(
                domain=domain,
                goal=goal,
                constraints=kwargs.get('constraints', []),
                available_concepts=kwargs.get('available_concepts', []),
                confidence_threshold=kwargs.get('confidence_threshold', 0.5),
                reasoning_depth=kwargs.get('reasoning_depth', 3)
            )

            # Determine which reasoning patterns to use
            if reasoning_types is None:
                reasoning_types = list(self.reasoning_patterns.keys())

            # Execute reasoning patterns
            results = []
            for reasoning_type in reasoning_types:
                if reasoning_type in self.reasoning_patterns:
                    try:
                        result = await self.reasoning_patterns[reasoning_type].reason(context)
                        if result.confidence >= context.confidence_threshold:
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error in {reasoning_type} reasoning: {e}")

            # Sort results by confidence
            results.sort(key=lambda x: x.confidence, reverse=True)

            self.logger.info(f"Completed reasoning with {len(results)} valid results")
            return results

        except Exception as e:
            self.logger.error(f"Error in enhanced reasoning: {e}")
            return []

    async def synthesize_reasoning_results(self, results: List[ReasoningResult], goal: str) -> str:
        """Synthesize multiple reasoning results into a unified conclusion."""
        if not results:
            return f"No valid reasoning results found for: {goal}"

        if len(results) == 1:
            return f"{results[0].reasoning_type.value.title()} reasoning: {results[0].conclusion}"

        try:
            prompt = f"""
            Synthesize these reasoning results into a unified conclusion:

            {chr(10).join(f"- {r.reasoning_type.value.title()}: {r.conclusion} (confidence: {r.confidence:.2f})" for r in results)}

            Goal: {goal}

            Provide a coherent synthesis that integrates the different reasoning approaches.
            """

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_interface.generate_response(messages=messages)
            return response.content.strip()

        except Exception as e:
            self.logger.error(f"Error synthesizing reasoning results: {e}")
            return f"Multiple reasoning approaches suggest: {results[0].conclusion}"


class EnhancedReasoningTool(BaseTool):
    """Tool for enhanced reasoning with domain-specific patterns."""

    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        super().__init__(config)
        self.llm_interface = llm_interface
        self.name = "enhanced_reasoning"
        self.description = "Apply enhanced reasoning patterns with domain-specific knowledge"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "The reasoning goal or question to address"
                        },
                        "domain": {
                            "type": "string",
                            "description": "The knowledge domain (e.g., 'technology', 'science', 'business')"
                        },
                        "reasoning_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["deductive", "inductive", "analogical", "causal", "creative"]
                            },
                            "description": "Specific reasoning types to apply"
                        },
                        "confidence_threshold": {
                            "type": "number",
                            "description": "Minimum confidence threshold for results (0.0-1.0)"
                        },
                        "synthesize_results": {
                            "type": "boolean",
                            "description": "Whether to synthesize multiple reasoning results"
                        }
                    },
                    "required": ["goal", "domain"]
                }
            }
        }

    async def execute(self, **kwargs) -> ToolResult:
        try:
            goal = kwargs.get("goal", "")
            domain = kwargs.get("domain", "general")
            reasoning_type_names = kwargs.get("reasoning_types", [])
            confidence_threshold = kwargs.get("confidence_threshold", 0.5)
            synthesize_results = kwargs.get("synthesize_results", True)

            if not goal.strip():
                return ToolResult(
                    success=False,
                    error="No reasoning goal provided"
                )

            # Convert reasoning type names to enums
            reasoning_types = []
            for name in reasoning_type_names:
                try:
                    reasoning_types.append(ReasoningType(name))
                except ValueError:
                    self.logger.warning(f"Unknown reasoning type: {name}")

            # Create reasoning engine (simplified for demo)
            neural_web = NeuralWeb()  # In production, get from system
            reasoning_engine = EnhancedReasoningEngine(
                self.config, neural_web, self.llm_interface
            )

            # Execute reasoning
            results = await reasoning_engine.reason(
                goal=goal,
                domain=domain,
                reasoning_types=reasoning_types if reasoning_types else None,
                confidence_threshold=confidence_threshold
            )

            # Prepare response
            if not results:
                return ToolResult(
                    success=True,
                    result="No reasoning results met the confidence threshold",
                    metadata={"goal": goal, "domain": domain, "threshold": confidence_threshold}
                )

            # Synthesize results if requested
            if synthesize_results and len(results) > 1:
                synthesis = await reasoning_engine.synthesize_reasoning_results(results, goal)

                return ToolResult(
                    success=True,
                    result=synthesis,
                    metadata={
                        "goal": goal,
                        "domain": domain,
                        "reasoning_results": [
                            {
                                "type": r.reasoning_type.value,
                                "conclusion": r.conclusion,
                                "confidence": round(r.confidence, 3),
                                "supporting_evidence": len(r.supporting_evidence)
                            }
                            for r in results
                        ],
                        "synthesis": synthesis
                    }
                )
            else:
                # Return best single result
                best_result = results[0]
                return ToolResult(
                    success=True,
                    result=f"{best_result.reasoning_type.value.title()} reasoning: {best_result.conclusion}",
                    metadata={
                        "goal": goal,
                        "domain": domain,
                        "reasoning_type": best_result.reasoning_type.value,
                        "confidence": round(best_result.confidence, 3),
                        "reasoning_path": best_result.reasoning_path,
                        "supporting_evidence": best_result.supporting_evidence
                    }
                )

        except Exception as e:
            logger.error(f"Error in enhanced reasoning tool: {e}")
            return ToolResult(
                success=False,
                error=f"Error during enhanced reasoning: {str(e)}"
            )
