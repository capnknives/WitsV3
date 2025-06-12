"""
Cross-domain learning implementation for WitsV3 Neural Web system.

This module provides functionality for cross-domain knowledge transfer and
concept propagation between different knowledge domains in the Neural Web.
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import networkx as nx

from .neural_web_core import NeuralWeb, ConceptNode
from .knowledge_graph import KnowledgeGraph
from .config import WitsV3Config
from .llm_interface import LLMInterface, LLMMessage, LLMResponse


class DomainClassifier:
    """
    Classifies concepts into knowledge domains based on semantic similarity
    and content analysis.
    """

    def __init__(self, config: WitsV3Config):
        self.config = config
        self.llm_interface = LLMInterface(config)
        self.domain_embeddings = {}
        self.known_domains = [
            "science", "art", "mathematics", "history",
            "technology", "philosophy", "business", "literature",
            "psychology", "economics", "politics", "medicine"
        ]
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize domain embeddings if not already done."""
        if not self.domain_embeddings:
            self.logger.info("Initializing domain embeddings")
            for domain in self.known_domains:
                description = await self.get_domain_description(domain)
                # Store as placeholder until we get actual embeddings
                self.domain_embeddings[domain] = description

    async def classify_domain(self, concept_text: str) -> str:
        """
        Classifies text into a knowledge domain.

        Args:
            concept_text: The text to classify

        Returns:
            The classified domain name
        """
        await self.initialize()

        # Use LLM to classify domain
        prompt = f"""
        Classify the following concept into one of these knowledge domains:
        {', '.join(self.known_domains)}

        Concept: {concept_text}

        Return only the domain name, nothing else.
        """

        messages = [LLMMessage(role="user", content=prompt)]
        response = await self.llm_interface.generate_response(messages=messages)

        # Clean and validate the response
        domain = response.content.strip().lower()
        if domain not in self.known_domains:
            self.logger.warning(f"Domain '{domain}' not in known domains, defaulting to 'general'")
            return "general"

        return domain

    async def get_domain_description(self, domain: str) -> str:
        """
        Gets a canonical description of a knowledge domain.

        Args:
            domain: The domain name

        Returns:
            A description of the domain
        """
        # Use LLM to generate domain description
        prompt = f"""
        Give a concise description of the '{domain}' knowledge domain.
        Focus on the core concepts, methodologies, and distinguishing features.
        Keep it under 100 words.
        """

        messages = [LLMMessage(role="user", content=prompt)]
        response = await self.llm_interface.generate_response(messages=messages)

        return response.content.strip()

    async def get_domain_similarity(self, domain1: str, domain2: str) -> float:
        """
        Calculate similarity between two domains.

        Args:
            domain1: First domain name
            domain2: Second domain name

        Returns:
            Similarity score between 0 and 1
        """
        # Simple implementation using LLM for assessment
        prompt = f"""
        On a scale from 0 to 1, how similar are the knowledge domains '{domain1}' and '{domain2}'?
        Consider shared methodologies, overlapping concepts, and historical connections.
        Return only the numeric score, nothing else.
        """

        messages = [LLMMessage(role="user", content=prompt)]
        response = await self.llm_interface.generate_response(messages=messages)

        try:
            similarity = float(response.content.strip())
            # Ensure valid range
            similarity = max(0.0, min(1.0, similarity))
            return similarity
        except ValueError:
            self.logger.warning(f"Could not parse similarity from '{response.content}', defaulting to 0.5")
            return 0.5


class CrossDomainLearning:
    """
    Implements cross-domain learning capabilities for the Neural Web system.
    Enables knowledge transfer between different domains through concept mapping
    and similarity detection.
    """

    def __init__(self, config: WitsV3Config, neural_web: NeuralWeb,
                 knowledge_graph: Optional[KnowledgeGraph] = None):
        """
        Initialize the cross-domain learning system.

        Args:
            config: The WitsV3 configuration
            neural_web: The Neural Web instance to use
            knowledge_graph: Optional Knowledge Graph instance
        """
        self.config = config
        self.neural_web = neural_web
        self.knowledge_graph = knowledge_graph
        self.domain_classifier = DomainClassifier(config)
        self.domain_mappings = {}
        self.cross_domain_activations = {}
        self.llm_interface = LLMInterface(config)
        self.logger = logging.getLogger(__name__)

    async def classify_concept_domain(self, concept_id: str) -> str:
        """
        Classifies a concept into a specific knowledge domain.

        Args:
            concept_id: The ID of the concept to classify

        Returns:
            The domain classification
        """
        concept = self.neural_web.get_node(concept_id)
        if not concept:
            self.logger.warning(f"Concept {concept_id} not found in neural web")
            return "unknown"

        # Check if domain is already in metadata
        if "domain" in concept.metadata:
            return concept.metadata["domain"]

        # Classify domain using the domain classifier
        domain = await self.domain_classifier.classify_domain(concept.concept)

        # Update concept metadata with domain
        concept.metadata["domain"] = domain

        return domain

    async def find_cross_domain_analogies(self,
                                          source_concept_id: str,
                                          target_domain: str) -> List[str]:
        """
        Finds analogies for a concept in a different domain.

        Args:
            source_concept_id: The source concept ID
            target_domain: The target domain to find analogies in

        Returns:
            List of concept IDs that are analogous in the target domain
        """
        source_concept = self.neural_web.get_node(concept_id=source_concept_id)
        if not source_concept:
            self.logger.warning(f"Source concept {source_concept_id} not found")
            return []

        # Get all concepts in the target domain
        target_domain_concepts = []
        for node_id, node in self.neural_web.nodes.items():
            if node.metadata.get("domain") == target_domain:
                target_domain_concepts.append(node)

        if not target_domain_concepts:
            self.logger.info(f"No concepts found in domain {target_domain}")
            return []

        # Use LLM to find analogies
        prompt = f"""
        Find analogies between this concept:

        Concept: {source_concept.concept}
        Description: {source_concept.metadata.get('description', 'No description')}
        Domain: {source_concept.metadata.get('domain', 'unknown')}

        And these concepts from the {target_domain} domain:

        {chr(10).join([f"{i+1}. {c.concept}" for i, c in enumerate(target_domain_concepts[:5])])}

        Return the numbers of concepts that are analogous, separated by commas.
        """

        messages = [LLMMessage(role="user", content=prompt)]
        response = await self.llm_interface.generate_response(messages=messages)

        # Parse response to get analogous concept indices
        try:
            indices = [int(idx.strip()) - 1 for idx in response.content.split(',')]
            valid_indices = [i for i in indices if 0 <= i < len(target_domain_concepts)]
            return [target_domain_concepts[i].id for i in valid_indices]
        except ValueError:
            self.logger.warning(f"Could not parse analogy indices from '{response.content}'")
            return []

    async def transfer_knowledge(self,
                                 source_domain: str,
                                 target_domain: str,
                                 concept_ids: List[str]) -> Dict[str, str]:
        """
        Transfers knowledge from one domain to another.

        Args:
            source_domain: Source knowledge domain
            target_domain: Target knowledge domain
            concept_ids: List of concept IDs to transfer

        Returns:
            Dictionary mapping source concept IDs to new target domain concept IDs
        """
        if not concept_ids:
            return {}

        result_mapping = {}

        for concept_id in concept_ids:
            source_concept = self.neural_web.get_node(concept_id)
            if not source_concept:
                continue

            # Use LLM to translate the concept to the target domain
            prompt = f"""
            Translate this concept from {source_domain} to {target_domain} domain:

            Source concept: {source_concept.concept}
            Source domain: {source_domain}
            Target domain: {target_domain}

            Provide a concept name for the analogous concept in the target domain.
            """

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_interface.generate_response(messages=messages)

            # Create new concept in target domain
            new_concept_text = response.content.strip()
            new_concept_id = str(uuid.uuid4())

            # Create and add the new concept node
            new_concept = ConceptNode(
                id=new_concept_id,
                concept=new_concept_text,
                metadata={
                    "domain": target_domain,
                    "source_concept_id": concept_id,
                    "source_domain": source_domain,
                    "created_at": datetime.now().isoformat()
                }
            )

            self.neural_web.add_node(new_concept)

            # Create bidirectional connection between concepts
            self.neural_web.connect_nodes(
                source_id=concept_id,
                target_id=new_concept_id,
                connection_type="domain_analogy",
                strength=0.8,
                metadata={
                    "source_domain": source_domain,
                    "target_domain": target_domain
                }
            )

            result_mapping[concept_id] = new_concept_id

        return result_mapping

    async def propagate_cross_domain_activation(self,
                                               concept_id: str,
                                               activation_level: float) -> Dict[str, float]:
        """
        Propagates activation of a concept across domains.

        Args:
            concept_id: The concept ID to activate
            activation_level: The initial activation level

        Returns:
            Dictionary of concept IDs to activation levels
        """
        concept = self.neural_web.get_node(concept_id)
        if not concept:
            return {}

        # Get domain of activated concept
        source_domain = concept.metadata.get("domain")
        if not source_domain:
            source_domain = await self.classify_concept_domain(concept_id)

        # Get all connections of the concept
        connected_concepts = {}
        for conn_id, conn in concept.connections.items():
            connected_concepts[conn_id] = conn

        # Find cross-domain connections
        cross_domain_activations = {}
        for conn_id, conn in connected_concepts.items():
            target_concept = self.neural_web.get_node(conn_id)
            if not target_concept:
                continue

            target_domain = target_concept.metadata.get("domain")
            if not target_domain:
                target_domain = await self.classify_concept_domain(conn_id)

            # If this is a cross-domain connection
            if target_domain != source_domain:
                # Calculate propagated activation based on connection strength
                propagated_activation = activation_level * conn["strength"]

                # Get domain similarity to further adjust activation
                domain_similarity = await self.domain_classifier.get_domain_similarity(
                    source_domain, target_domain
                )

                # Final activation adjusted by domain similarity
                final_activation = propagated_activation * domain_similarity

                cross_domain_activations[conn_id] = final_activation

                # If above threshold, continue propagation to second-order connections
                threshold = self.config.memory_manager.neural_web_settings.connection_strength_threshold
                if final_activation > threshold:
                    # Recursively propagate with reduced activation
                    second_order = await self.propagate_cross_domain_activation(
                        conn_id, final_activation * 0.5
                    )

                    # Merge second-order activations
                    for sec_id, sec_activation in second_order.items():
                        if sec_id not in cross_domain_activations:
                            cross_domain_activations[sec_id] = sec_activation
                        else:
                            # Take higher of existing or new activation
                            cross_domain_activations[sec_id] = max(
                                cross_domain_activations[sec_id], sec_activation
                            )

        return cross_domain_activations

    async def get_domain_concepts(self, domain: str, limit: int = 10) -> List[str]:
        """
        Get concepts from a specific domain.

        Args:
            domain: The domain to get concepts from
            limit: Maximum number of concepts to return

        Returns:
            List of concept IDs in the domain
        """
        domain_concepts = []

        for node_id, node in self.neural_web.nodes.items():
            if node.metadata.get("domain") == domain:
                domain_concepts.append(node_id)

            if len(domain_concepts) >= limit:
                break

        return domain_concepts

    async def analyze_domain_relationships(self) -> Dict[Tuple[str, str], float]:
        """
        Analyze relationships between different knowledge domains.

        Returns:
            Dictionary mapping domain pairs to their relationship strength
        """
        # Get all domains present in the neural web
        domains = set()
        for node in self.neural_web.nodes.values():
            if "domain" in node.metadata:
                domains.add(node.metadata["domain"])

        # Calculate domain relationships
        domain_relationships = {}

        for domain1 in domains:
            for domain2 in domains:
                if domain1 >= domain2:  # Skip duplicate pairs and self-pairs
                    continue

                # Calculate domain similarity
                similarity = await self.domain_classifier.get_domain_similarity(domain1, domain2)
                domain_relationships[(domain1, domain2)] = similarity

        return domain_relationships

    async def create_cross_domain_knowledge_graph(self) -> nx.Graph:
        """
        Create a graph representation of cross-domain relationships.

        Returns:
            NetworkX graph of domain relationships
        """
        graph = nx.Graph()

        # Get domain relationships
        domain_relationships = await self.analyze_domain_relationships()

        # Add nodes for each domain
        domains = set()
        for (domain1, domain2), _ in domain_relationships.items():
            domains.add(domain1)
            domains.add(domain2)

        for domain in domains:
            graph.add_node(domain, type="domain")

        # Add edges for domain relationships
        for (domain1, domain2), strength in domain_relationships.items():
            graph.add_edge(domain1, domain2, weight=strength, type="domain_relationship")

        # Add cross-domain concept relationships
        for node_id, node in self.neural_web.nodes.items():
            if "domain" not in node.metadata:
                continue

            source_domain = node.metadata["domain"]

            for conn_id, conn in node.connections.items():
                target_node = self.neural_web.get_node(conn_id)
                if not target_node or "domain" not in target_node.metadata:
                    continue

                target_domain = target_node.metadata["domain"]

                if source_domain != target_domain:
                    # Add weight to the domain relationship edge
                    if graph.has_edge(source_domain, target_domain):
                        current_weight = graph.edges[source_domain, target_domain]["weight"]
                        graph.edges[source_domain, target_domain]["weight"] = current_weight + conn["strength"] * 0.1
                    else:
                        graph.add_edge(source_domain, target_domain, weight=conn["strength"] * 0.1, type="domain_relationship")

        return graph


# Example usage in tests
async def test_cross_domain_learning():
    """Test function for the CrossDomainLearning class."""
    config = WitsV3Config()
    neural_web = NeuralWeb()

    # Create test concepts
    node1 = ConceptNode(id="1", concept="Gravity", metadata={"domain": "physics"})
    node2 = ConceptNode(id="2", concept="Influence", metadata={"domain": "sociology"})

    neural_web.add_node(node1)
    neural_web.add_node(node2)

    # Create cross-domain learning instance
    cdl = CrossDomainLearning(config, neural_web)

    # Test domain classification
    domain = await cdl.classify_concept_domain("1")
    print(f"Domain: {domain}")

    # Test cross-domain analogy
    analogies = await cdl.find_cross_domain_analogies("1", "sociology")
    print(f"Analogies: {analogies}")

    # Test knowledge transfer
    mapping = await cdl.transfer_knowledge("physics", "sociology", ["1"])
    print(f"Mapping: {mapping}")

    # Test activation propagation
    activations = await cdl.propagate_cross_domain_activation("1", 1.0)
    print(f"Activations: {activations}")


if __name__ == "__main__":
    # Run the test function
    asyncio.run(test_cross_domain_learning())
