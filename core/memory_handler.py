"""
Memory Handler for the WitsV3 Synthetic Brain.

This module provides a unified interface for managing various memory systems:
- Working Memory: Short-term active memory for current processing
- Episodic Memory: Event-based memories with temporal information
- Semantic Memory: Conceptual knowledge and facts
- Procedural Memory: Action sequences and procedures

The memory handler integrates with existing memory management systems in WitsV3
and provides new capabilities for the synthetic brain architecture.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel

from core.memory_manager import MemoryManager
from core.working_memory import WorkingMemory
from core.knowledge_graph import KnowledgeGraph
from core.memory_export import export_memory
from core.memory_summarization import summarize_memory_segment

logger = logging.getLogger("WitsV3.MemoryHandler")


# Define the models at module level for proper imports
class MemorySegment(BaseModel):
    """Model representing a segment of memory."""
    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: float
    importance: float = 0.5
    memory_type: str  # "episodic", "semantic", "procedural"
    associations: List[str] = []
    embedding_id: Optional[str] = None
    last_accessed: float = 0.0


class MemoryContext(BaseModel):
    """Model representing the current memory context."""
    working_memory: Dict[str, Any] = {}
    active_concepts: Set[str] = set()
    recent_memories: List[str] = []
    context_id: str = ""
    creation_time: float = 0.0
    last_updated: float = 0.0


class MemoryHandler:
    """
    Unified handler for all memory systems in the WitsV3 synthetic brain.

    This handler integrates with existing memory systems and provides new capabilities
    required for the synthetic brain architecture.
    """

    def __init__(self, config_path: str = "config/wits_core.yaml"):
        """
        Initialize the memory handler with configuration.

        Args:
            config_path: Path to the wits_core.yaml configuration file
        """
        self.logger = logging.getLogger("WitsV3.MemoryHandler")
        self.config = self._load_config(config_path)
        self.memory_manager = MemoryManager()  # Existing memory manager
        self.working_memory = WorkingMemory()  # Existing working memory
        self.knowledge_graph = KnowledgeGraph()  # Existing knowledge graph

        # Current memory context
        self.context = MemoryContext(
            context_id=str(uuid.uuid4()),
            creation_time=time.time(),
            last_updated=time.time()
        )

        # Directories for memory serialization
        self.episodic_path = Path(self.config.get("memory_systems", {}).get(
            "episodic", {}).get("serialization_path", "./logs/episodes"))
        os.makedirs(self.episodic_path, exist_ok=True)

        self.logger.info("Memory handler initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load memory configuration from wits_core.yaml"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            return {}

    async def remember(self, content: str, memory_type: str = "episodic",
                       metadata: Optional[Dict[str, Any]] = None,
                       importance: float = 0.5) -> str:
        """
        Store information in the appropriate memory system.

        Args:
            content: The content to remember
            memory_type: Type of memory ("episodic", "semantic", "procedural")
            metadata: Additional metadata about the memory
            importance: Importance score (0.0 to 1.0)

        Returns:
            Memory ID of the stored memory
        """
        if metadata is None:
            metadata = {}

        timestamp = time.time()
        memory_id = str(uuid.uuid4())

        # Create memory segment
        segment = MemorySegment(
            id=memory_id,
            content=content,
            metadata=metadata,
            timestamp=timestamp,
            importance=importance,
            memory_type=memory_type,
            last_accessed=timestamp
        )

        # Update working memory context
        self.context.last_updated = timestamp
        if len(self.context.recent_memories) >= 20:
            self.context.recent_memories.pop(0)  # Remove oldest
        self.context.recent_memories.append(memory_id)

        # Store in appropriate memory system
        if memory_type == "episodic":
            await self._store_episodic_memory(segment)
        elif memory_type == "semantic":
            await self._store_semantic_memory(segment)
        elif memory_type == "procedural":
            await self._store_procedural_memory(segment)
        else:
            self.logger.warning(f"Unknown memory type: {memory_type}")

        return memory_id

    async def recall(self, query: str, memory_type: Optional[str] = None,
                    limit: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Recall memories based on the query.

        Args:
            query: Search query
            memory_type: Optional filter by memory type
            limit: Maximum number of results to return
            threshold: Similarity threshold

        Returns:
            List of matching memories
        """
        # Update access time for tracking
        access_time = time.time()

        # Base on memory type, query appropriate system
        if memory_type == "episodic" or memory_type is None:
            episodic_results = await self._recall_episodic(query, limit, threshold)
            results = episodic_results

        if memory_type == "semantic" or memory_type is None:
            semantic_results = await self._recall_semantic(query, limit, threshold)
            if memory_type is None:
                # Combine results if not filtering by type
                results = episodic_results + semantic_results
                # Sort by relevance
                results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
                results = results[:limit]
            else:
                results = semantic_results

        # Update memory segments' access time
        for result in results:
            memory_id = result.get("id")
            if memory_id:
                await self._update_access_time(memory_id, access_time)

        return results

    async def _store_episodic_memory(self, segment: MemorySegment) -> None:
        """Store an episodic memory"""
        # Use existing memory manager to store in vector database
        key = f"episodic:{segment.id}"
        await self.memory_manager.store(key, segment.content, segment.metadata)

        # Serialize to disk for persistence
        date_str = datetime.fromtimestamp(segment.timestamp).strftime("%Y-%m-%d")
        episode_dir = self.episodic_path / date_str
        os.makedirs(episode_dir, exist_ok=True)

        with open(episode_dir / f"{segment.id}.json", "w") as f:
            json.dump(segment.dict(), f, indent=2)

    async def _store_semantic_memory(self, segment: MemorySegment) -> None:
        """Store a semantic memory"""
        # Store in the knowledge graph
        key = f"semantic:{segment.id}"
        await self.memory_manager.store(key, segment.content, segment.metadata)

        # Extract concepts and add to knowledge graph
        concepts = segment.metadata.get("concepts", [])
        if concepts:
            for concept in concepts:
                self.knowledge_graph.add_concept(concept, segment.content)

    async def _store_procedural_memory(self, segment: MemorySegment) -> None:
        """Store a procedural memory"""
        # To be implemented in future phase
        self.logger.info("Procedural memory storage not yet implemented")

    async def _recall_episodic(self, query: str, limit: int, threshold: float) -> List[Dict[str, Any]]:
        """Recall episodic memories"""
        results = await self.memory_manager.search(query, limit=limit)
        filtered_results = []

        for result in results:
            key = result.get("key", "")
            if key.startswith("episodic:"):
                # Check if relevance meets threshold
                if result.get("relevance", 0) >= threshold:
                    filtered_results.append(result)

        return filtered_results

    async def _recall_semantic(self, query: str, limit: int, threshold: float) -> List[Dict[str, Any]]:
        """Recall semantic memories"""
        results = await self.memory_manager.search(query, limit=limit)
        filtered_results = []

        for result in results:
            key = result.get("key", "")
            if key.startswith("semantic:"):
                # Check if relevance meets threshold
                if result.get("relevance", 0) >= threshold:
                    filtered_results.append(result)

        # Also check knowledge graph for relevant concepts
        concepts = self.knowledge_graph.search_concepts(query, limit=limit)
        for concept in concepts:
            filtered_results.append({
                "id": f"concept:{concept.id}",
                "content": concept.description,
                "metadata": {"type": "concept"},
                "relevance": concept.relevance
            })

        # Sort by relevance and limit results
        filtered_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return filtered_results[:limit]

    async def _update_access_time(self, memory_id: str, timestamp: float) -> None:
        """Update access time for a memory segment"""
        # This function will update access time metadata
        # Implementation will depend on the specifics of the memory backend
        pass

    async def get_current_context(self) -> Dict[str, Any]:
        """
        Get the current memory context for reasoning and decisions.

        Returns:
            Dictionary with context information
        """
        # Update context with current working memory state
        self.context.working_memory = self.working_memory.get_snapshot()
        self.context.active_concepts = set(self.knowledge_graph.get_active_concepts())

        # Sort recent memories by importance and recency
        recent_memory_data = []
        for memory_id in self.context.recent_memories:
            # In a real implementation, we would fetch this data from storage
            # For now, we'll return placeholder data
            recent_memory_data.append({
                "id": memory_id,
                "summary": f"Memory {memory_id[:8]}..."
            })

        return {
            "context_id": self.context.context_id,
            "working_memory": self.context.working_memory,
            "active_concepts": list(self.context.active_concepts),
            "recent_memories": recent_memory_data,
            "creation_time": self.context.creation_time,
            "last_updated": self.context.last_updated
        }

    async def consolidate_memories(self) -> None:
        """
        Consolidate memories periodically for long-term storage.
        This function should be called by the background agent.
        """
        # Get episodic memories from the last 24 hours
        yesterday = datetime.now().strftime("%Y-%m-%d")
        yesterday_dir = self.episodic_path / yesterday

        if not yesterday_dir.exists():
            self.logger.info(f"No episodes to consolidate for {yesterday}")
            return

        # Collect all episodes from the day
        episodes = []
        for episode_file in yesterday_dir.glob("*.json"):
            try:
                with open(episode_file, "r") as f:
                    episode = json.load(f)
                    episodes.append(episode)
            except Exception as e:
                self.logger.error(f"Error loading episode {episode_file}: {e}")

        if not episodes:
            return

        # Sort by timestamp
        episodes.sort(key=lambda x: x.get("timestamp", 0))

        # Create a summary using the memory summarization module
        summary_text = f"Daily memories for {yesterday}:\n\n"
        for episode in episodes:
            summary_text += f"- {episode.get('content', 'No content')})\n"

        # Generate a summary using the summarization module
        summary = await summarize_memory_segment(summary_text)

        # Store the consolidated memory
        metadata = {
            "date": yesterday,
            "num_episodes": len(episodes),
            "type": "daily_consolidation"
        }

        await self.remember(
            content=summary,
            memory_type="semantic",
            metadata=metadata,
            importance=0.8
        )

        self.logger.info(f"Consolidated {len(episodes)} memories from {yesterday}")

    async def export_all_memories(self, export_path: str) -> None:
        """
        Export all memories to the specified path.

        Args:
            export_path: Directory to export memories to
        """
        await export_memory(export_path)
        self.logger.info(f"Exported all memories to {export_path}")


# Test function
async def test_memory_handler():
    """Simple test for memory handler functionality"""
    handler = MemoryHandler()

    # Store some test memories
    memory_id = await handler.remember(
        "User asked about climate change solutions",
        memory_type="episodic",
        metadata={"source": "user", "topic": "climate"}
    )

    await handler.remember(
        "Solar panels convert sunlight directly into electricity",
        memory_type="semantic",
        metadata={"topic": "renewable energy", "concepts": ["solar", "electricity"]}
    )

    # Recall memories
    results = await handler.recall("climate solutions")

    return f"Stored and recalled memories successfully. Results: {len(results)}"


if __name__ == "__main__":
    asyncio.run(test_memory_handler())
