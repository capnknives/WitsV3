"""
Working Memory implementation for WitsV3.

This module provides a short-term memory system that integrates with the
Knowledge Graph and Neural Web components to maintain active context
for reasoning and decision-making.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any, Union, AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
import heapq

from .config import WitsV3Config
from .llm_interface import BaseLLMInterface
from .knowledge_graph import KnowledgeGraph, Entity, Relation
from .neural_web_core import NeuralWeb, ConceptNode, Connection
from .schemas import StreamData

logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """Represents an item in working memory."""
    content: str
    item_type: str  # 'entity', 'fact', 'goal', 'task', etc.
    source: str  # where this item came from: 'user', 'agent', 'knowledge_graph', 'neural_web', etc.
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    importance: float = 1.0  # higher values = more important
    activation: float = 1.0  # current activation level
    created_at: datetime = field(default_factory=lambda: datetime.now())
    last_accessed: datetime = field(default_factory=lambda: datetime.now())
    access_count: int = 0
    ttl: Optional[timedelta] = None  # time-to-live
    related_items: List[str] = field(default_factory=list)  # IDs of related memory items
    entity_id: Optional[str] = None  # ID of related knowledge graph entity if applicable
    concept_id: Optional[str] = None  # ID of related neural web concept if applicable
    metadata: Dict[str, Any] = field(default_factory=dict)

    def activate(self, amount: float = 0.2):
        """Increase activation level and update access metrics."""
        self.activation = min(1.0, self.activation + amount)
        self.last_accessed = datetime.now()
        self.access_count += 1

    def decay(self, amount: float = 0.1):
        """Decrease activation level."""
        self.activation = max(0.0, self.activation - amount)

    @property
    def is_expired(self) -> bool:
        """Check if this item has expired based on TTL."""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at) > self.ttl

    @property
    def relevance_score(self) -> float:
        """Calculate overall relevance score based on activation, importance, and recency."""
        # Time factor - items accessed more recently get higher scores
        time_since_access = (datetime.now() - self.last_accessed).total_seconds()
        recency_factor = max(0.1, 1.0 - (time_since_access / (60 * 60 * 24)))  # 24 hour decay

        # Calculate combined score
        return self.activation * self.importance * recency_factor

class WorkingMemory:
    """
    Working Memory system that maintains the active context for an agent.

    Working memory is responsible for:
    1. Keeping track of current goals, tasks, and relevant facts
    2. Maintaining conversation context
    3. Connecting short-term memory to long-term knowledge (Knowledge Graph & Neural Web)
    4. Providing fast retrieval of immediately relevant information
    """

    def __init__(self,
                 config: WitsV3Config,
                 llm_interface: BaseLLMInterface,
                 knowledge_graph: Optional[KnowledgeGraph] = None,
                 neural_web: Optional[NeuralWeb] = None):
        """Initialize working memory.

        Args:
            config: WitsV3 configuration
            llm_interface: LLM interface for embeddings and reasoning
            knowledge_graph: Optional knowledge graph integration
            neural_web: Optional neural web integration
        """
        self.config = config
        self.llm_interface = llm_interface
        self.knowledge_graph = knowledge_graph
        self.neural_web = neural_web
        self.logger = logging.getLogger(__name__)

        # Working memory storage
        self.items: Dict[str, MemoryItem] = {}

        # Working memory capacity limits - using default values
        self.max_items = 50
        self.activation_threshold = 0.2
        self.decay_rate = 0.1
        self.default_ttl = timedelta(seconds=3600)
        self.decay_interval_seconds = 60

        # Try to load from config if available
        try:
            if hasattr(config, 'working_memory'):
                self.max_items = getattr(config.working_memory, 'max_items', self.max_items)
                self.activation_threshold = getattr(config.working_memory, 'activation_threshold', self.activation_threshold)
                self.decay_rate = getattr(config.working_memory, 'decay_rate', self.decay_rate)
                self.default_ttl = timedelta(seconds=getattr(config.working_memory, 'default_ttl_seconds', 3600))
                self.decay_interval_seconds = getattr(config.working_memory, 'decay_interval_seconds', self.decay_interval_seconds)
        except Exception as e:
            self.logger.warning(f"Error loading working memory configuration: {e}. Using defaults.")

        # Create decay task
        self._decay_task = None
        self._start_decay_task()

    def _start_decay_task(self):
        """Start the periodic decay task."""
        async def decay_loop():
            while True:
                await self.decay_items()
                await asyncio.sleep(self.decay_interval_seconds)

        self._decay_task = asyncio.create_task(decay_loop())

    async def add_item(self,
                      content: str,
                      item_type: str,
                      source: str,
                      importance: float = 1.0,
                      ttl: Optional[timedelta] = None,
                      entity_id: Optional[str] = None,
                      concept_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> MemoryItem:
        """Add a new item to working memory.

        Args:
            content: The content of the memory item
            item_type: Type of memory item
            source: Source of the memory item
            importance: Importance score (0.0 to 1.0)
            ttl: Optional time-to-live duration
            entity_id: Optional ID of related knowledge graph entity
            concept_id: Optional ID of related neural web concept
            metadata: Optional metadata

        Returns:
            The created MemoryItem
        """
        # Create the memory item
        item = MemoryItem(
            content=content,
            item_type=item_type,
            source=source,
            importance=importance,
            ttl=ttl or self.default_ttl,
            entity_id=entity_id,
            concept_id=concept_id,
            metadata=metadata or {}
        )

        # Add to storage
        self.items[item.id] = item

        # If we're over capacity, remove least relevant items
        if len(self.items) > self.max_items:
            await self._prune_items()

        # Activate related concepts in neural web if available
        if self.neural_web and concept_id:
            await self.neural_web.activate_concept(concept_id)

        self.logger.debug(f"Added item to working memory: {content[:50]}...")
        return item

    async def get_item(self, item_id: str) -> Optional[MemoryItem]:
        """Get a memory item by ID and activate it.

        Args:
            item_id: ID of the memory item

        Returns:
            MemoryItem if found, None otherwise
        """
        item = self.items.get(item_id)
        if item:
            item.activate()
        return item

    async def remove_item(self, item_id: str) -> bool:
        """Remove a memory item.

        Args:
            item_id: ID of the memory item

        Returns:
            True if removed, False if not found
        """
        if item_id not in self.items:
            return False

        del self.items[item_id]
        return True

    async def relate_items(self, item_id1: str, item_id2: str):
        """Create a relationship between two memory items.

        Args:
            item_id1: ID of the first memory item
            item_id2: ID of the second memory item
        """
        if item_id1 not in self.items or item_id2 not in self.items:
            return

        item1 = self.items[item_id1]
        item2 = self.items[item_id2]

        # Add bidirectional relationship
        if item_id2 not in item1.related_items:
            item1.related_items.append(item_id2)

        if item_id1 not in item2.related_items:
            item2.related_items.append(item_id1)

    async def get_active_items(self,
                              min_activation: float = 0.0,
                              item_types: Optional[List[str]] = None,
                              limit: int = 10) -> List[MemoryItem]:
        """Get active memory items filtered by activation level and type.

        Args:
            min_activation: Minimum activation level
            item_types: Optional list of item types to filter by
            limit: Maximum number of items to return

        Returns:
            List of memory items
        """
        filtered_items = []

        for item in self.items.values():
            # Skip expired items
            if item.is_expired:
                continue

            # Check activation threshold
            if item.activation < min_activation:
                continue

            # Check item type
            if item_types and item.item_type not in item_types:
                continue

            filtered_items.append(item)

        # Sort by relevance score
        filtered_items.sort(key=lambda x: x.relevance_score, reverse=True)

        return filtered_items[:limit]

    async def search_items(self,
                          query: str,
                          item_types: Optional[List[str]] = None,
                          min_relevance: float = 0.0,
                          limit: int = 10) -> List[Tuple[MemoryItem, float]]:
        """Search memory items based on content similarity.

        Args:
            query: Search query
            item_types: Optional list of item types to filter by
            min_relevance: Minimum relevance score threshold
            limit: Maximum number of items to return

        Returns:
            List of (memory_item, relevance_score) tuples
        """
        if not query:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())
        results = []

        try:
            # Generate query embedding if LLM interface is available
            query_embedding = await self.llm_interface.get_embedding(
                query,
                model=self.config.ollama_settings.embedding_model
            )
        except Exception as e:
            self.logger.warning(f"Failed to generate embedding for query: {e}")
            query_embedding = None

        # Calculate scores for all items
        for item in self.items.values():
            # Filter by item type if specified
            if item_types and item.item_type not in item_types:
                continue

            # Calculate semantic similarity if embeddings available
            semantic_score = 0.0
            if query_embedding is not None:
                try:
                    item_embedding = await self.llm_interface.get_embedding(
                        item.content,
                        model=self.config.ollama_settings.embedding_model
                    )
                    semantic_score = self._cosine_similarity(query_embedding, item_embedding)
                except Exception as e:
                    self.logger.debug(f"Failed to calculate embedding similarity: {e}")

            # Calculate keyword matching score
            item_lower = item.content.lower()
            keyword_score = 0.0

            # Direct keyword match score
            for word in query_words:
                if word in item_lower:
                    keyword_score += 0.2

            # Exact phrase match gives higher score
            if query_lower in item_lower:
                keyword_score += 0.6

            # Combined score - weigh keyword matches higher than semantic similarity
            combined_score = (keyword_score * 0.7) + (semantic_score * 0.3)

            # Factor in item relevance
            final_score = combined_score * item.relevance_score

            if final_score >= min_relevance:
                results.append((item, final_score))

        # Sort by score (descending) and take top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        if len(vec1) != len(vec2):
            raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))

    async def decay_items(self):
        """Apply decay to all memory items and remove expired/inactive items."""
        # Apply decay
        for item in self.items.values():
            item.decay(self.decay_rate)

        # Remove expired items
        expired_ids = [item_id for item_id, item in self.items.items()
                      if item.is_expired or item.activation <= 0.01]

        for item_id in expired_ids:
            del self.items[item_id]

        if expired_ids:
            self.logger.debug(f"Removed {len(expired_ids)} expired/inactive items from working memory")

    async def _prune_items(self):
        """Remove least relevant items when over capacity."""
        # Calculate relevance scores for all items
        scored_items = [(item.relevance_score, item.id) for item in self.items.values()]

        # Sort by relevance score (ascending)
        scored_items.sort()

        # Calculate how many items to remove
        num_to_remove = len(self.items) - self.max_items

        # Remove least relevant items
        for _, item_id in scored_items[:num_to_remove]:
            del self.items[item_id]

        self.logger.debug(f"Pruned {num_to_remove} items from working memory")

    async def clear(self):
        """Clear all items from working memory."""
        self.items.clear()
        self.logger.info("Cleared working memory")

    async def get_context_summary(self, max_items: int = 10) -> str:
        """Generate a summary of the current working memory context.

        Args:
            max_items: Maximum number of items to include

        Returns:
            Text summary of current context
        """
        active_items = await self.get_active_items(limit=max_items)

        if not active_items:
            return "No active context in working memory."

        # Group by item type
        items_by_type = {}
        for item in active_items:
            if item.item_type not in items_by_type:
                items_by_type[item.item_type] = []
            items_by_type[item.item_type].append(item)

        # Generate summary
        summary_parts = ["Current context:"]

        for item_type, items in items_by_type.items():
            summary_parts.append(f"\n## {item_type.capitalize()}")
            for item in items:
                summary_parts.append(f"- {item.content}")

        return "\n".join(summary_parts)

    async def stream_context_thinking(self) -> AsyncIterator[StreamData]:
        """Stream the current working memory context as thinking process.

        Yields:
            StreamData objects containing the thinking process
        """
        active_items = await self.get_active_items(limit=20)

        if not active_items:
            yield StreamData(
                type="thinking",
                content="No active context in working memory.",
                source="working_memory"
            )
            return

        # Group by item type
        items_by_type = {}
        for item in active_items:
            if item.item_type not in items_by_type:
                items_by_type[item.item_type] = []
            items_by_type[item.item_type].append(item)

        # Stream summary
        yield StreamData(
            type="thinking",
            content="Current working memory context:",
            source="working_memory"
        )

        for item_type, items in items_by_type.items():
            yield StreamData(
                type="thinking",
                content=f"\n## {item_type.capitalize()}",
                source="working_memory"
            )

            for item in items:
                yield StreamData(
                    type="thinking",
                    content=f"- {item.content} (relevance: {item.relevance_score:.2f})",
                    source="working_memory",
                    metadata={"item_id": item.id}
                )

                # Add connections if item relates to knowledge graph or neural web
                if item.entity_id and self.knowledge_graph:
                    entity = await self.knowledge_graph.get_entity(item.entity_id)
                    if entity:
                        yield StreamData(
                            type="thinking",
                            content=f"  Related to entity: {entity.name} ({entity.entity_type})",
                            source="working_memory",
                            metadata={"entity_id": entity.id}
                        )

                if item.concept_id and self.neural_web:
                    concept = self.neural_web.concepts.get(item.concept_id)
                    if concept:
                        yield StreamData(
                            type="thinking",
                            content=f"  Related to concept: {concept.content} ({concept.concept_type})",
                            source="working_memory",
                            metadata={"concept_id": concept.id}
                        )

    async def connect_to_knowledge_graph(self, item_id: str, entity_id: str) -> bool:
        """Connect a working memory item to a knowledge graph entity.

        Args:
            item_id: ID of the memory item
            entity_id: ID of the knowledge graph entity

        Returns:
            True if connected, False otherwise
        """
        if item_id not in self.items or not self.knowledge_graph:
            return False

        entity = await self.knowledge_graph.get_entity(entity_id)
        if not entity:
            return False

        # Update the memory item
        item = self.items[item_id]
        item.entity_id = entity_id

        self.logger.debug(f"Connected memory item {item_id} to entity {entity_id}")
        return True

    async def connect_to_neural_web(self, item_id: str, concept_id: str) -> bool:
        """Connect a working memory item to a neural web concept.

        Args:
            item_id: ID of the memory item
            concept_id: ID of the neural web concept

        Returns:
            True if connected, False otherwise
        """
        if item_id not in self.items or not self.neural_web:
            return False

        concept = self.neural_web.concepts.get(concept_id)
        if not concept:
            return False

        # Update the memory item
        item = self.items[item_id]
        item.concept_id = concept_id

        # Activate the concept
        concept.activate(0.5)

        self.logger.debug(f"Connected memory item {item_id} to concept {concept_id}")
        return True

    async def get_statistics(self) -> Dict[str, Any]:
        """Get working memory statistics.

        Returns:
            Dictionary of statistics
        """
        # Group items by type
        items_by_type = {}
        for item in self.items.values():
            if item.item_type not in items_by_type:
                items_by_type[item.item_type] = 0
            items_by_type[item.item_type] += 1

        # Calculate average activation
        avg_activation = sum(item.activation for item in self.items.values()) / max(1, len(self.items))

        # Count connections
        kg_connections = sum(1 for item in self.items.values() if item.entity_id)
        nw_connections = sum(1 for item in self.items.values() if item.concept_id)

        return {
            'total_items': len(self.items),
            'items_by_type': items_by_type,
            'average_activation': avg_activation,
            'knowledge_graph_connections': kg_connections,
            'neural_web_connections': nw_connections,
            'capacity_used': len(self.items) / self.max_items
        }
