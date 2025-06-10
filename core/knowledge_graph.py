"""
Knowledge Graph implementation for WitsV3.

This module provides a graph-based knowledge representation system that integrates
with the Neural Web and Working Memory components for advanced reasoning capabilities.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
import networkx as nx
import numpy as np

from .config import WitsV3Config
from .llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    name: str
    entity_type: str  # person, place, thing, concept, event, etc.
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    properties: Dict[str, Any] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now())
    last_updated: datetime = field(default_factory=lambda: datetime.now())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Relation:
    """Represents a relation between entities in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: str  # isA, hasPart, causes, contradicts, etc.
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now())
    last_updated: datetime = field(default_factory=lambda: datetime.now())
    metadata: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraph:
    """
    Knowledge Graph implementation that provides a structured representation
    of entities and their relationships.
    """

    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        """Initialize the knowledge graph.

        Args:
            config: WitsV3 configuration
            llm_interface: LLM interface for embeddings and reasoning
        """
        self.config = config
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(__name__)

        # The graph structure
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}

        # Persistence settings - use a default path
        self.knowledge_file_path = Path("data/knowledge_graph.json")
        self.knowledge_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing knowledge graph if available
        asyncio.create_task(self._load_graph())

    async def _load_graph(self):
        """Load the knowledge graph from disk."""
        try:
            if not self.knowledge_file_path.exists():
                self.logger.info(f"No knowledge graph file found at {self.knowledge_file_path}")
                return

            with open(self.knowledge_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load entities
            for entity_data in data.get("entities", []):
                entity = Entity(**entity_data)
                self.entities[entity.id] = entity
                self.graph.add_node(entity.id, entity=entity)

            # Load relations
            for relation_data in data.get("relations", []):
                relation = Relation(**relation_data)
                self.relations[relation.id] = relation

                # Add edges to the graph
                if relation.source_id in self.entities and relation.target_id in self.entities:
                    self.graph.add_edge(
                        relation.source_id,
                        relation.target_id,
                        relation=relation
                    )

                    # Add reverse edge if bidirectional
                    if relation.bidirectional:
                        self.graph.add_edge(
                            relation.target_id,
                            relation.source_id,
                            relation=relation
                        )
                else:
                    self.logger.warning(f"Skipping relation {relation.id}: missing entities")

            self.logger.info(f"Loaded knowledge graph with {len(self.entities)} entities and {len(self.relations)} relations")

        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {e}")

    async def _save_graph(self):
        """Save the knowledge graph to disk."""
        try:
            data = {
                "entities": [asdict(entity) for entity in self.entities.values()],
                "relations": [asdict(relation) for relation in self.relations.values()]
            }

            with open(self.knowledge_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=self._json_serializer)

            self.logger.info(f"Saved knowledge graph with {len(self.entities)} entities and {len(self.relations)} relations")

        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {e}")

    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    async def add_entity(self,
                         name: str,
                         entity_type: str,
                         properties: Optional[Dict[str, Any]] = None,
                         observations: Optional[List[str]] = None,
                         confidence: float = 1.0,
                         metadata: Optional[Dict[str, Any]] = None) -> Entity:
        """Add a new entity to the knowledge graph.

        Args:
            name: Name of the entity
            entity_type: Type of entity (person, place, concept, etc.)
            properties: Optional properties dictionary
            observations: Optional list of observations about the entity
            confidence: Confidence score (0.0 to 1.0)
            metadata: Optional metadata dictionary

        Returns:
            The created Entity
        """
        # Create the entity
        entity = Entity(
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            observations=observations or [],
            confidence=confidence,
            metadata=metadata or {}
        )

        # Generate embedding if possible
        if observations and len(observations) > 0:
            text_to_embed = " ".join(observations)
            try:
                entity.embedding = await self.llm_interface.get_embedding(
                    text_to_embed,
                    model=self.config.ollama_settings.embedding_model
                )
            except Exception as e:
                self.logger.warning(f"Failed to generate embedding for entity {name}: {e}")

        # Add to collections
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, entity=entity)

        # Save changes
        await self._save_graph()

        self.logger.info(f"Added entity: {name} ({entity_type})")
        return entity

    async def add_relation(self,
                          source_id: str,
                          target_id: str,
                          relation_type: str,
                          properties: Optional[Dict[str, Any]] = None,
                          weight: float = 1.0,
                          confidence: float = 1.0,
                          bidirectional: bool = False,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[Relation]:
        """Add a relation between two entities.

        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            relation_type: Type of relation
            properties: Optional properties dictionary
            weight: Weight/strength of the relation (0.0 to 1.0)
            confidence: Confidence score (0.0 to 1.0)
            bidirectional: Whether the relation is bidirectional
            metadata: Optional metadata dictionary

        Returns:
            The created Relation or None if entities don't exist
        """
        # Check that both entities exist
        if source_id not in self.entities or target_id not in self.entities:
            self.logger.warning(f"Cannot create relation: one or both entities don't exist")
            return None

        # Create the relation
        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            weight=weight,
            confidence=confidence,
            bidirectional=bidirectional,
            metadata=metadata or {}
        )

        # Add to collections
        self.relations[relation.id] = relation
        self.graph.add_edge(source_id, target_id, relation=relation)

        # Add reverse edge if bidirectional
        if bidirectional:
            self.graph.add_edge(target_id, source_id, relation=relation)

        # Save changes
        await self._save_graph()

        self.logger.info(f"Added relation: {relation_type} from {source_id} to {target_id}")
        return relation

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID.

        Args:
            entity_id: ID of the entity to retrieve

        Returns:
            Entity if found, None otherwise
        """
        return self.entities.get(entity_id)

    async def find_entities_by_name(self, name: str, exact_match: bool = False) -> List[Entity]:
        """Find entities by name.

        Args:
            name: Name to search for
            exact_match: Whether to require exact name match

        Returns:
            List of matching entities
        """
        if exact_match:
            return [e for e in self.entities.values() if e.name == name]
        else:
            return [e for e in self.entities.values() if name.lower() in e.name.lower()]

    async def find_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Find entities by type.

        Args:
            entity_type: Entity type to filter by

        Returns:
            List of matching entities
        """
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    async def search_entities(self,
                             query: str,
                             entity_types: Optional[List[str]] = None,
                             min_confidence: float = 0.0,
                             limit: int = 10) -> List[Tuple[Entity, float]]:
        """Search for entities using semantic similarity.

        Args:
            query: Text query to search for
            entity_types: Optional list of entity types to filter by
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results

        Returns:
            List of (entity, score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = await self.llm_interface.get_embedding(
                query,
                model=self.config.ollama_settings.embedding_model
            )

            # Filter entities
            candidate_entities = self.entities.values()
            if entity_types:
                candidate_entities = [e for e in candidate_entities if e.entity_type in entity_types]

            candidate_entities = [e for e in candidate_entities if e.confidence >= min_confidence]

            # Score entities with embeddings
            scored_entities = []
            for entity in candidate_entities:
                # Skip entities without embeddings
                if not entity.embedding:
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, entity.embedding)
                scored_entities.append((entity, similarity))

            # Sort by similarity score
            scored_entities.sort(key=lambda x: x[1], reverse=True)

            return scored_entities[:limit]

        except Exception as e:
            self.logger.error(f"Error searching entities: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))

    async def find_relations(self,
                            source_id: Optional[str] = None,
                            target_id: Optional[str] = None,
                            relation_type: Optional[str] = None) -> List[Relation]:
        """Find relations matching the given criteria.

        Args:
            source_id: Optional source entity ID
            target_id: Optional target entity ID
            relation_type: Optional relation type

        Returns:
            List of matching relations
        """
        results = []

        for relation in self.relations.values():
            matches = True

            if source_id is not None and relation.source_id != source_id:
                matches = False

            if target_id is not None and relation.target_id != target_id:
                matches = False

            if relation_type is not None and relation.relation_type != relation_type:
                matches = False

            if matches:
                results.append(relation)

        return results

    async def get_connected_entities(self,
                                   entity_id: str,
                                   relation_types: Optional[List[str]] = None,
                                   max_depth: int = 1,
                                   direction: str = 'both') -> Dict[str, List[Tuple[Entity, Relation]]]:
        """Get entities connected to the given entity.

        Args:
            entity_id: ID of the entity
            relation_types: Optional list of relation types to filter by
            max_depth: Maximum traversal depth
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            Dictionary mapping depth to list of (entity, connecting_relation) tuples
        """
        if entity_id not in self.entities:
            return {}

        results: Dict[str, List[Tuple[Entity, Relation]]] = {}
        visited = set([entity_id])

        # Initialize queue with (entity_id, depth) tuples
        queue = [(entity_id, 1)]

        while queue:
            current_id, depth = queue.pop(0)

            if depth > max_depth:
                continue

            # Initialize results for this depth if needed
            depth_key = f"depth_{depth}"
            if depth_key not in results:
                results[depth_key] = []

            # Process outgoing relations
            if direction in ['outgoing', 'both']:
                for _, neighbor_id, edge_data in self.graph.out_edges(current_id, data=True):
                    relation = edge_data.get('relation')

                    # Skip if relation type doesn't match filter
                    if relation_types and relation.relation_type not in relation_types:
                        continue

                    # Add to results if not visited
                    if neighbor_id not in visited:
                        results[depth_key].append((self.entities[neighbor_id], relation))
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, depth + 1))

            # Process incoming relations
            if direction in ['incoming', 'both']:
                for neighbor_id, _, edge_data in self.graph.in_edges(current_id, data=True):
                    relation = edge_data.get('relation')

                    # Skip if relation type doesn't match filter
                    if relation_types and relation.relation_type not in relation_types:
                        continue

                    # Add to results if not visited
                    if neighbor_id not in visited:
                        results[depth_key].append((self.entities[neighbor_id], relation))
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, depth + 1))

        return results

    async def find_paths(self,
                        source_id: str,
                        target_id: str,
                        max_length: int = 5,
                        relation_types: Optional[List[str]] = None) -> List[List[Tuple[Entity, Optional[Relation]]]]:
        """Find paths between two entities.

        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            max_length: Maximum path length
            relation_types: Optional list of relation types to filter by

        Returns:
            List of paths, where each path is a list of (entity, relation) tuples
        """
        if source_id not in self.entities or target_id not in self.entities:
            return []

        # Create a filtered graph if relation types are specified
        graph = self.graph
        if relation_types:
            filtered_graph = nx.DiGraph()

            # Add all nodes
            for node in self.graph.nodes():
                filtered_graph.add_node(node, **self.graph.nodes[node])

            # Add only edges with matching relation types
            for source, target, data in self.graph.edges(data=True):
                relation = data.get('relation')
                if relation and relation.relation_type in relation_types:
                    filtered_graph.add_edge(source, target, **data)

            graph = filtered_graph

        try:
            # Find all simple paths up to max_length
            path_nodes = list(nx.all_simple_paths(
                graph, source_id, target_id, cutoff=max_length
            ))

            # Convert paths of node IDs to paths of (entity, relation) tuples
            paths = []
            for node_path in path_nodes:
                entity_relation_path = []

                # Add first entity with no incoming relation
                entity_relation_path.append((self.entities[node_path[0]], None))

                # Add remaining entities with their incoming relations
                for i in range(1, len(node_path)):
                    prev_id = node_path[i-1]
                    curr_id = node_path[i]

                    # Get the relation between previous and current entity
                    relation = None
                    for _, _, edge_data in graph.out_edges(prev_id, data=True):
                        if edge_data.get('relation').target_id == curr_id:
                            relation = edge_data.get('relation')
                            break

                    entity_relation_path.append((self.entities[curr_id], relation))

                paths.append(entity_relation_path)

            # Sort paths by quality (shorter paths first, then by average relation weight)
            paths.sort(key=lambda p: (len(p), -sum(r.weight if r else 0 for _, r in p[1:]) / (len(p) - 1)))

            return paths

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    async def update_entity(self,
                           entity_id: str,
                           properties: Optional[Dict[str, Any]] = None,
                           observations: Optional[List[str]] = None) -> Optional[Entity]:
        """Update an entity's properties and/or observations.

        Args:
            entity_id: ID of the entity to update
            properties: Optional properties to update
            observations: Optional observations to add

        Returns:
            Updated Entity if found, None otherwise
        """
        entity = self.entities.get(entity_id)
        if not entity:
            return None

        # Update properties
        if properties:
            entity.properties.update(properties)

        # Add observations
        if observations:
            for obs in observations:
                if obs not in entity.observations:
                    entity.observations.append(obs)

            # Update embedding
            if entity.observations:
                text_to_embed = " ".join(entity.observations)
                try:
                    entity.embedding = await self.llm_interface.get_embedding(
                        text_to_embed,
                        model=self.config.ollama_settings.embedding_model
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to update embedding for entity {entity.name}: {e}")

        # Update timestamp
        entity.last_updated = datetime.now()

        # Save changes
        await self._save_graph()

        self.logger.info(f"Updated entity: {entity.name}")
        return entity

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relations.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if deleted, False if not found
        """
        if entity_id not in self.entities:
            return False

        # Remove all relations involving this entity
        relation_ids_to_remove = []
        for relation_id, relation in self.relations.items():
            if relation.source_id == entity_id or relation.target_id == entity_id:
                relation_ids_to_remove.append(relation_id)

        for relation_id in relation_ids_to_remove:
            del self.relations[relation_id]

        # Remove from graph
        self.graph.remove_node(entity_id)

        # Remove entity
        entity_name = self.entities[entity_id].name
        del self.entities[entity_id]

        # Save changes
        await self._save_graph()

        self.logger.info(f"Deleted entity: {entity_name}")
        return True

    async def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation.

        Args:
            relation_id: ID of the relation to delete

        Returns:
            True if deleted, False if not found
        """
        if relation_id not in self.relations:
            return False

        relation = self.relations[relation_id]

        # Remove from graph
        self.graph.remove_edge(relation.source_id, relation.target_id)

        # Remove reverse edge if bidirectional
        if relation.bidirectional and self.graph.has_edge(relation.target_id, relation.source_id):
            self.graph.remove_edge(relation.target_id, relation.source_id)

        # Remove relation
        del self.relations[relation_id]

        # Save changes
        await self._save_graph()

        self.logger.info(f"Deleted relation: {relation.relation_type} from {relation.source_id} to {relation.target_id}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics.

        Returns:
            Dictionary of statistics
        """
        entity_types = {}
        for entity in self.entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1

        relation_types = {}
        for relation in self.relations.values():
            relation_types[relation.relation_type] = relation_types.get(relation.relation_type, 0) + 1

        return {
            'entity_count': len(self.entities),
            'relation_count': len(self.relations),
            'entity_types': entity_types,
            'relation_types': relation_types,
            'average_observations_per_entity': sum(len(e.observations) for e in self.entities.values()) / max(1, len(self.entities)),
            'graph_density': nx.density(self.graph),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'average_path_length': nx.average_shortest_path_length(self.graph) if nx.is_weakly_connected(self.graph) else None
        }
