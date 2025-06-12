# agents/neural_orchestrator_agent.py
"""
Neural Web Enhanced Orchestrator Agent for WitsV3
Integrates graph-based reasoning with traditional orchestration
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime

from .llm_driven_orchestrator import LLMDrivenOrchestrator
from core.neural_web_core import NeuralWeb
from core.memory_manager import MemorySegment, MemorySegmentContent
from core.schemas import StreamData
from core.cross_domain_learning import CrossDomainLearning

logger = logging.getLogger(__name__)


class NeuralOrchestratorAgent(LLMDrivenOrchestrator):
    """
    Enhanced orchestrator that uses neural web reasoning for
    improved planning, decision making, and knowledge synthesis
    """

    def __init__(self, *args, neural_web: Optional[NeuralWeb] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_web = neural_web or NeuralWeb()
        self.reasoning_depth = getattr(self.config.agents, 'reasoning_depth', 3)
        self.enable_neural_reasoning = getattr(self.config.agents, 'enable_neural_reasoning', True)

        # Initialize cross-domain learning if enabled
        self.cross_domain_learning = None
        if hasattr(self.config.memory_manager, 'neural_web_settings') and \
           getattr(self.config.memory_manager.neural_web_settings, 'enable_cross_domain_learning', False):
            self.cross_domain_learning = CrossDomainLearning(
                config=self.config,
                neural_web=self.neural_web
            )
            logger.info("Cross-domain learning capabilities initialized")

        logger.info(f"Neural orchestrator initialized with reasoning depth: {self.reasoning_depth}")

    async def _execute_react_loop(self, state: Dict[str, Any], session_id: str) -> AsyncGenerator[StreamData, None]:
        """Enhanced ReAct loop with neural web integration"""

        if self.enable_neural_reasoning:
            # Pre-process with neural reasoning
            neural_insights = await self._get_neural_insights(state["goal"])

            # Add cross-domain insights if enabled
            if self.cross_domain_learning and neural_insights.get("active_concepts"):
                cross_domain_insights = await self._get_cross_domain_insights(
                    state["goal"],
                    neural_insights["active_concepts"]
                )
                if cross_domain_insights:
                    neural_insights["cross_domain"] = cross_domain_insights
                    yield StreamData(
                        type="thinking",
                        content=f"Cross-domain analysis: {len(cross_domain_insights.get('analogies', []))} cross-domain analogies found",
                        source=self.agent_name,
                        metadata={"insights": cross_domain_insights, "type": "cross_domain_insight"}
                    )

            # Update context with neural insights
            if neural_insights:
                state["neural_context"] = neural_insights
                yield StreamData(
                    type="thinking",
                    content=f"Neural web analysis: {len(neural_insights.get('active_concepts', []))} relevant concepts activated",
                    source=self.agent_name,
                    metadata={"insights": neural_insights, "type": "neural_insight"}
                )

        # Execute enhanced ReAct loop
        async for stream_data in super()._execute_react_loop(state, session_id):
            yield stream_data

    async def _get_neural_insights(self, goal: str) -> Dict[str, Any]:
        """Get insights from the neural web for enhanced reasoning"""
        try:
            # Find relevant concepts
            relevant_concepts = await self.neural_web._find_relevant_concepts(goal)

            if not relevant_concepts:
                return {}

            # Activate relevant knowledge
            activated_concepts = []
            for concept_id in relevant_concepts:
                activated = await self.neural_web.activate_concept(concept_id, 0.8)
                activated_concepts.extend(activated)

            # Get reasoning results for different patterns
            insights = {}
            reasoning_patterns = ["chain", "modus_ponens", "analogy"]

            for pattern in reasoning_patterns:
                try:
                    result = await self.neural_web.reason(goal, pattern)
                    if result.get("results"):
                        insights[pattern] = result
                except Exception as e:
                    logger.warning(f"Error in {pattern} reasoning: {e}")

            # Get network statistics
            insights["network_stats"] = self.neural_web.get_statistics()

            # Get currently active concepts
            active_concepts = [
                {
                    "id": concept_id,
                    "content": concept.content[:100] + "..." if len(concept.content) > 100 else concept.content,
                    "activation": concept.activation_level,
                    "type": concept.concept_type
                }
                for concept_id, concept in self.neural_web.concepts.items()
                if concept.activation_level > self.neural_web.activation_threshold
            ]

            insights["active_concepts"] = sorted(
                active_concepts,
                key=lambda x: x["activation"],
                reverse=True
            )[:10]

            return insights

        except Exception as e:
            logger.error(f"Error getting neural insights: {e}")
            return {}

    async def _enhance_reasoning_with_neural_web(self, thought: str, goal: str) -> str:
        """Enhance agent reasoning with neural web insights"""
        try:
            # Get relevant concepts for current thought
            thought_concepts = await self.neural_web._find_relevant_concepts(thought)
            if not thought_concepts:
                return thought

            # Get reasoning chains
            reasoning_result = await self.neural_web.reason(thought, "chain")

            enhanced_thought = thought

            if reasoning_result.get("results"):
                # Add reasoning insights to thought
                insights = []
                for result in reasoning_result["results"][:3]:  # Top 3 insights
                    if result.get("reasoning"):
                        insights.append(result["reasoning"])

                if insights:
                    enhanced_thought += f"\n\nNeural reasoning insights:\n"
                    for i, insight in enumerate(insights, 1):
                        enhanced_thought += f"{i}. {insight}\n"

                return enhanced_thought

            return enhanced_thought

        except Exception as e:
            logger.error(f"Error enhancing reasoning: {e}")
            return thought

    async def _create_neural_memory(self, content: str, memory_type: str,
                                  concept_type: str = "memory") -> str:
        """Create a memory segment and corresponding neural web concept"""
        try:
            # Create traditional memory segment
            segment = MemorySegment(
                type=memory_type,
                source=self.agent_name,
                content=MemorySegmentContent(text=content),
                metadata={"neural_enhanced": True}
            )

            # Check if memory manager exists before adding segment
            if self.memory_manager:
                segment_id = await self.memory_manager.add_segment(segment)
            else:
                # Generate a fallback ID if no memory manager
                import uuid
                segment_id = f"memory_{uuid.uuid4().hex[:8]}"
                logger.warning(f"No memory manager available, using generated ID: {segment_id}")

            # Add to neural web if it's a neural memory backend (string check to avoid type issues)
            try:
                if (self.memory_manager and
                    hasattr(self.memory_manager, 'backend') and
                    self.memory_manager.backend is not None and
                    str(type(self.memory_manager.backend).__name__) == 'NeuralMemoryBackend'):
                    # Access neural_web using getattr to avoid type checking
                    neural_web = getattr(self.memory_manager.backend, 'neural_web', None)
                    if neural_web:
                        await neural_web.add_concept(
                            concept_id=segment_id,
                            content=content,
                            concept_type=concept_type,
                            metadata={
                                "memory_type": memory_type,
                                "agent": self.agent_name,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
            except Exception as neural_error:
                logger.warning(f"Could not add to neural web: {neural_error}")

            return segment_id

        except Exception as e:
            logger.error(f"Error creating neural memory: {e}")
            # Generate a valid concept ID instead of returning empty string
            import uuid
            fallback_id = f"error_concept_{uuid.uuid4().hex[:8]}"
            logger.info(f"Generated fallback concept ID: {fallback_id}")
            return fallback_id

    async def _synthesize_knowledge(self, concepts: List[str]) -> Dict[str, Any]:
        """Synthesize knowledge from multiple activated concepts"""
        try:
            if not concepts:
                return {}

            synthesis = {
                "patterns": [],
                "connections": [],
                "insights": []
            }

            # Find patterns between concepts
            for i, concept_id in enumerate(concepts):
                for j, other_id in enumerate(concepts[i+1:], i+1):
                    # Find paths between concepts
                    paths = await self.neural_web.find_path(concept_id, other_id, max_length=3)

                    if paths:
                        connection = {
                            "from": self.neural_web.concepts[concept_id].content[:50],
                            "to": self.neural_web.concepts[other_id].content[:50],
                            "path_length": len(paths[0]) if paths else 0,
                            "strength": self.neural_web._calculate_path_score(paths[0]) if paths else 0
                        }
                        synthesis["connections"].append(connection)

            # Sort connections by strength
            synthesis["connections"].sort(key=lambda x: x["strength"], reverse=True)
            synthesis["connections"] = synthesis["connections"][:5]  # Top 5

            # Generate insights
            if synthesis["connections"]:
                synthesis["insights"].append(
                    f"Found {len(synthesis['connections'])} strong knowledge connections"
                )

                strongest = synthesis["connections"][0]
                synthesis["insights"].append(
                    f"Strongest connection: {strongest['from']} → {strongest['to']}"
                )

            return synthesis

        except Exception as e:
            logger.error(f"Error synthesizing knowledge: {e}")
            return {}

    async def _plan_with_neural_reasoning(self, goal: str) -> List[Dict[str, Any]]:
        """Create an enhanced plan using neural web reasoning"""
        try:
            # Get traditional plan first
            traditional_plan = await self._create_basic_plan(goal)

            if not self.enable_neural_reasoning:
                return traditional_plan

            # Enhance with neural insights
            neural_insights = await self._get_neural_insights(goal)

            enhanced_plan = []

            for step in traditional_plan:
                enhanced_step = step.copy()

                # Add neural context if available
                if neural_insights.get("active_concepts"):
                    relevant_concepts = [
                        concept for concept in neural_insights["active_concepts"]
                        if any(word in step["description"].lower()
                              for word in concept["content"].lower().split()[:5])
                    ]

                    if relevant_concepts:
                        enhanced_step["neural_context"] = relevant_concepts[:3]
                        enhanced_step["confidence"] = min(1.0, step.get("confidence", 0.5) + 0.2)

                enhanced_plan.append(enhanced_step)

            # Add neural-specific steps if insights suggest them
            if neural_insights.get("chain", {}).get("results"):
                for result in neural_insights["chain"]["results"][:2]:
                    if result.get("confidence", 0) > 0.5:
                        enhanced_plan.append({
                            "description": f"Consider: {result.get('reasoning', '')}",
                            "type": "neural_insight",
                            "confidence": result.get("confidence", 0.5),
                            "neural_source": True
                        })

            return enhanced_plan

        except Exception as e:
            logger.error(f"Error in neural planning: {e}")
            return await self._create_basic_plan(goal)

    async def _create_basic_plan(self, goal: str) -> List[Dict[str, Any]]:
        """Create a basic plan without neural enhancement"""
        # This would be implemented based on the goal
        # For now, return a simple plan structure
        return [
            {
                "description": f"Analyze the goal: {goal}",
                "type": "analysis",
                "confidence": 0.8
            },
            {
                "description": "Gather relevant information",
                "type": "information_gathering",
                "confidence": 0.7
            },
            {
                "description": "Execute the plan",
                "type": "execution",
                "confidence": 0.6
            }
        ]

    async def get_neural_statistics(self) -> Dict[str, Any]:
        """Get current neural web statistics"""
        if not self.neural_web:
            return {}

        return {
            "neural_web_stats": self.neural_web.get_statistics(),
            "reasoning_enabled": self.enable_neural_reasoning,
            "reasoning_depth": self.reasoning_depth,
            "active_concepts": len([
                c for c in self.neural_web.concepts.values()
                if c.activation_level > self.neural_web.activation_threshold
            ])
        }

    async def _get_cross_domain_insights(self, goal: str, active_concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get cross-domain insights for enhanced reasoning across knowledge domains"""
        if not self.cross_domain_learning:
            return {}

        try:
            insights = {
                "analogies": [],
                "domain_connections": [],
                "knowledge_transfers": []
            }

            # Get domains represented in active concepts
            concept_domains = {}
            for concept in active_concepts:
                concept_id = concept["id"]
                domain = await self.cross_domain_learning.classify_concept_domain(concept_id)
                if domain != "unknown":
                    concept_domains[concept_id] = domain

            # Find unique domains
            domains = list(set(concept_domains.values()))

            # For each domain pair, find cross-domain analogies
            for domain in domains:
                concepts_in_domain = [cid for cid, d in concept_domains.items() if d == domain]
                other_domains = [d for d in domains if d != domain]

                for target_domain in other_domains:
                    for concept_id in concepts_in_domain[:3]:  # Limit to 3 concepts per domain
                        # Find analogies across domains
                        analogies = await self.cross_domain_learning.find_cross_domain_analogies(
                            concept_id, target_domain
                        )

                        if analogies:
                            # Get source concept details
                            source_concept = self.neural_web.get_node(concept_id)
                            source_text = source_concept.concept if source_concept else concept_id

                            # Get analogy concept details
                            analogy_concepts = []
                            for analogy_id in analogies:
                                analogy_node = self.neural_web.get_node(analogy_id)
                                if analogy_node:
                                    analogy_concepts.append({
                                        "id": analogy_id,
                                        "concept": analogy_node.concept,
                                        "domain": target_domain
                                    })

                            if analogy_concepts:
                                insights["analogies"].append({
                                    "source_concept": {
                                        "id": concept_id,
                                        "concept": source_text,
                                        "domain": domain
                                    },
                                    "analogy_concepts": analogy_concepts
                                })

            # Get domain relationship graph if we have multiple domains
            if len(domains) > 1:
                # Simplified representation for streamdata
                domain_pairs = []
                relationships = await self.cross_domain_learning.analyze_domain_relationships()
                for (domain1, domain2), strength in relationships.items():
                    domain_pairs.append({
                        "domain1": domain1,
                        "domain2": domain2,
                        "strength": strength
                    })
                insights["domain_connections"] = domain_pairs

            # Transfer knowledge between domains based on goal
            if len(domains) > 1:
                # Find primary domain for the goal
                goal_domain = await self._classify_goal_domain(goal)

                # Transfer knowledge to the goal domain
                for domain in domains:
                    if domain != goal_domain:
                        source_concepts = [cid for cid, d in concept_domains.items() if d == domain][:2]
                        if source_concepts:
                            transfers = await self.cross_domain_learning.transfer_knowledge(
                                domain, goal_domain, source_concepts
                            )

                            if transfers:
                                insights["knowledge_transfers"].append({
                                    "source_domain": domain,
                                    "target_domain": goal_domain,
                                    "transfers": transfers
                                })

            return insights

        except Exception as e:
            logger.error(f"Error getting cross-domain insights: {e}")
            return {}

    async def _classify_goal_domain(self, goal: str) -> str:
        """Classify the domain of the goal text"""
        if not self.cross_domain_learning:
            return "general"

        try:
            # Create a temporary concept to classify
            node = self.neural_web.add_node_by_text(goal, "goal")

            # Classify domain
            domain = await self.cross_domain_learning.domain_classifier.classify_domain(goal)

            return domain
        except Exception as e:
            logger.error(f"Error classifying goal domain: {e}")
            return "general"

    async def cross_domain_reasoning(self, query: str) -> Dict[str, Any]:
        """
        Apply cross-domain reasoning to answer a query.

        Args:
            query: The query to reason about

        Returns:
            Dictionary with reasoning results
        """
        if not self.cross_domain_learning:
            return {"error": "Cross-domain learning not enabled"}

        try:
            # Extract concepts from query
            concepts = await self.neural_web._find_relevant_concepts(query)

            if not concepts:
                return {"error": "No relevant concepts found for the query"}

            # Classify domains for each concept
            concept_domains = {}
            for concept_id in concepts:
                domain = await self.cross_domain_learning.classify_concept_domain(concept_id)
                concept_domains[concept_id] = domain

            # Activate relevant concepts
            for concept_id in concepts:
                await self.neural_web.activate_concept(concept_id, 0.8)

                # Propagate activation across domains
                if self.cross_domain_learning:
                    cross_domain_activations = await self.cross_domain_learning.propagate_cross_domain_activation(
                        concept_id, 0.8
                    )

                    # Add activated concepts to relevant concepts
                    for cid, activation_level in cross_domain_activations.items():
                        if cid not in concepts and activation_level > self.neural_web.activation_threshold:
                            concepts.append(cid)

            # Find cross-domain analogies
            analogies = {}
            domains = list(set(concept_domains.values()))

            for concept_id, domain in concept_domains.items():
                target_domains = [d for d in domains if d != domain]
                for target_domain in target_domains:
                    analogy_concepts = await self.cross_domain_learning.find_cross_domain_analogies(
                        concept_id, target_domain
                    )
                    if analogy_concepts:
                        if concept_id not in analogies:
                            analogies[concept_id] = []
                        analogies[concept_id].extend(analogy_concepts)

            # Get information about active concepts for response
            active_concepts_info = []
            all_concepts = set(concepts)
            for cid in analogies:
                all_concepts.add(cid)
                all_concepts.update(analogies[cid])

            for concept_id in all_concepts:
                concept = self.neural_web.get_node(concept_id)
                if concept:
                    domain = await self.cross_domain_learning.classify_concept_domain(concept_id)
                    active_concepts_info.append({
                        "id": concept_id,
                        "concept": concept.concept,
                        "domain": domain
                    })

            # Return results
            return {
                "query": query,
                "concepts": active_concepts_info,
                "analogies": [
                    {
                        "source_id": src_id,
                        "source_concept": next((c["concept"] for c in active_concepts_info if c["id"] == src_id), ""),
                        "source_domain": next((c["domain"] for c in active_concepts_info if c["id"] == src_id), ""),
                        "analogy_ids": analogy_ids,
                        "analogy_concepts": [
                            next((c["concept"] for c in active_concepts_info if c["id"] == a_id), "")
                            for a_id in analogy_ids
                        ],
                        "analogy_domains": [
                            next((c["domain"] for c in active_concepts_info if c["id"] == a_id), "")
                            for a_id in analogy_ids
                        ]
                    }
                    for src_id, analogy_ids in analogies.items()
                ]
            }

        except Exception as e:
            logger.error(f"Error in cross-domain reasoning: {e}")
            return {"error": str(e)}
