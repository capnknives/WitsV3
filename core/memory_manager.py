"""
Memory management system for WitsV3
"""

import json
import os
import asyncio
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, AsyncGenerator
from abc import ABC, abstractmethod

from .config import WitsV3Config, load_config
from pydantic import BaseModel, Field
import uuid
import numpy as np

from .config import WitsV3Config, MemoryManagerSettings
from .llm_interface import BaseLLMInterface # To get embeddings

# --- Pydantic Models for Memory Segments ---
class MemorySegmentContent(BaseModel):
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    # Add other content types as needed, e.g., image_url, code_block

class MemorySegment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    type: str # e.g., USER_INPUT, AGENT_THOUGHT, TOOL_CALL, TOOL_RESPONSE, SYSTEM_MESSAGE
    source: str # e.g., user, agent_name, tool_name
    content: MemorySegmentContent
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # For search results, not stored directly in the segment itself usually
    relevance_score: Optional[float] = None

# --- Base Memory Backend ---
class BaseMemoryBackend(ABC):
    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        self.config = config
        self.settings: MemoryManagerSettings = config.memory_manager
        self.llm_interface = llm_interface
        self.segments: List[MemorySegment] = []
        self.is_initialized: bool = False
        self.last_prune_time: float = time.monotonic()

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def add_segment(self, segment: MemorySegment) -> str:
        pass

    @abstractmethod
    async def get_segment(self, segment_id: str) -> Optional[MemorySegment]:
        pass

    @abstractmethod
    async def search_segments(
        self,
        query_text: str,
        limit: int = 5,
        min_relevance: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        pass

    @abstractmethod
    async def get_recent_segments(
        self,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        pass

    async def _generate_embedding_if_needed(self, segment: MemorySegment) -> None:
        if segment.embedding is None:
            text_to_embed = segment.content.text or segment.content.tool_output
            if text_to_embed:
                try:                    segment.embedding = await self.llm_interface.get_embedding(
                        text_to_embed,
                        model=self.config.ollama_settings.embedding_model
                    )
                except Exception as e:
                    print(f"Error generating embedding for segment {segment.id}: {e}")
                    # Decide if we want to store the segment without embedding or fail

    async def _prune_if_needed(self):
        """Check if memory pruning is needed based on time interval or size threshold."""
        if not self.settings.enable_auto_pruning:
            return

        current_time = time.monotonic()
        should_prune_by_time = (current_time - self.last_prune_time) > self.settings.pruning_interval_seconds
        should_prune_by_size = await self._should_prune_by_size()

        if should_prune_by_time or should_prune_by_size:
            await self.prune_memory()
            self.last_prune_time = current_time

    async def _should_prune_by_size(self) -> bool:
        """Check if memory size exceeds the configured threshold."""
        try:
            current_size_mb = await self._get_memory_size_mb()
            max_size_mb = self.settings.max_memory_size_mb
            threshold = self.settings.pruning_threshold

            return current_size_mb >= (max_size_mb * threshold)
        except Exception as e:
            print(f"Error checking memory size: {e}")
            return False

    async def _get_memory_size_mb(self) -> float:
        """Get the current memory size in megabytes. Override in subclasses for specific backends."""
        # Default implementation - estimate based on number of segments and average size
        if not self.segments:
            return 0.0

        # Rough estimate: average of 50KB per segment (based on our observation)
        estimated_size_mb = len(self.segments) * 0.05  # 50KB per segment in MB
        return estimated_size_mb

    @abstractmethod
    async def prune_memory(self):
        # Implement pruning logic (e.g., by count, by age, by importance)
        pass

# --- Basic JSON File Backend ---
class BasicMemoryBackend(BaseMemoryBackend):
    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        super().__init__(config, llm_interface)
        self.memory_file = self.settings.memory_file_path
        self._lock = asyncio.Lock() # For file operations
        self.logger = logging.getLogger("WitsV3.MemoryBackend")

    async def initialize(self):
        async with self._lock:
            if os.path.exists(self.memory_file):
                try:
                    with open(self.memory_file, 'r') as f:
                        segments_data = json.load(f)
                    self.segments = [MemorySegment(**data) for data in segments_data]
                    print(f"Loaded {len(self.segments)} segments from {self.memory_file}")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Error loading memory file {self.memory_file}: {e}. Starting with empty memory.")
                    self.segments = []
                except Exception as e:
                    print(f"Unexpected error loading memory file {self.memory_file}: {e}. Starting with empty memory.")
                    self.segments = []
            else:
                self.segments = []
                print(f"Memory file {self.memory_file} not found. Starting with empty memory.")
        self.is_initialized = True

    async def _save_to_disk(self):
        async with self._lock:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

                # Prepare data for JSON serialization, converting datetime objects to ISO format
                save_data = []
                for segment in self.segments:
                    segment_dict = segment.model_dump(exclude_none=True)
                    # Convert datetime objects to ISO format strings for JSON serialization
                    if 'timestamp' in segment_dict and hasattr(segment_dict['timestamp'], 'isoformat'):
                        segment_dict['timestamp'] = segment_dict['timestamp'].isoformat()
                    save_data.append(segment_dict)

                with open(self.memory_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
            except Exception as e:
                print(f"Error saving memory to {self.memory_file}: {e}")

    async def add_segment(self, segment: MemorySegment) -> str:
        if not self.is_initialized:
            await self.initialize()
        await self._generate_embedding_if_needed(segment)
        self.segments.append(segment)
        await self._save_to_disk()
        # TODO: Temporarily disabled auto-pruning to fix syntax errors
        # await self._prune_if_needed()
        return segment.id

    async def get_segment(self, segment_id: str) -> Optional[MemorySegment]:
        if not self.is_initialized: await self.initialize()
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        return None

    async def search_segments(
        self,
        query_text: str,
        limit: int = 5,
        min_relevance: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None # Basic filtering for this backend
    ) -> List[MemorySegment]:
        if not self.is_initialized: await self.initialize()
        if not query_text:
            return []

        query_embedding = await self.llm_interface.get_embedding(
            query_text,
            model=self.config.ollama_settings.embedding_model
        )
        if not query_embedding:
            return []

        query_np = np.array(query_embedding)
        scored_segments = []

        for segment in self.segments:
            if segment.embedding:
                # Basic filtering
                if filter_dict:
                    match = True
                    for key, value in filter_dict.items():
                        if getattr(segment, key, None) != value and segment.metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue

                segment_np = np.array(segment.embedding)

                # Check for dimension mismatch
                if query_np.shape[0] != segment_np.shape[0]:
                    self.logger.warning(f"Embedding dimension mismatch: query {query_np.shape[0]} != segment {segment_np.shape[0]}. Skipping this segment.")
                    continue

                # Cosine similarity
                similarity = np.dot(query_np, segment_np) / (np.linalg.norm(query_np) * np.linalg.norm(segment_np))
                if similarity >= min_relevance:
                    # Create a copy to add relevance score without modifying original
                    segment_copy = segment.model_copy(deep=True)
                    segment_copy.relevance_score = float(similarity)
                    scored_segments.append(segment_copy)

        # Sort by relevance score, descending
        scored_segments.sort(key=lambda s: s.relevance_score or 0.0, reverse=True)
        return scored_segments[:limit]

    async def get_recent_segments(
        self,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        if not self.is_initialized: await self.initialize()
        # Sort by timestamp, most recent first
        # This is inefficient for large lists, but okay for a basic backend
        sorted_segments = sorted(self.segments, key=lambda s: s.timestamp, reverse=True)

        results = []
        for segment in sorted_segments:
            if len(results) >= limit:
                break
            if filter_dict:
                match = True
                for key, value in filter_dict.items():
                    if getattr(segment, key, None) != value and segment.metadata.get(key) != value:
                        match = False
                        break

                if match:
                    results.append(segment)
            else:
                results.append(segment)
        return results

    async def _get_memory_size_mb(self) -> float:
        """Get the current memory size in megabytes for BasicMemoryBackend."""
        if os.path.exists(self.memory_file):
            size_bytes = os.path.getsize(self.memory_file)
            return size_bytes / (1024 * 1024)
        return 0.0

    async def prune_memory(self):
        """Enhanced pruning with both count and size-based strategies."""
        initial_count = len(self.segments)

        # Check count-based pruning first
        if initial_count > self.settings.max_memory_segments:
            num_to_prune = initial_count - self.settings.max_memory_segments
            print(f"Pruning {num_to_prune} segments due to count limit ({initial_count} > {self.settings.max_memory_segments})")

            # Enhanced pruning strategy
            strategy = getattr(self.settings, 'pruning_strategy', 'hybrid')

            if strategy == "oldest_first":
                # Sort by timestamp, keeping newest
                self.segments = sorted(
                    self.segments,
                    key=lambda s: s.timestamp or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True
                )[:self.settings.max_memory_segments]
            elif strategy == "least_relevant":
                # Sort by importance, keeping most important
                self.segments = sorted(
                    self.segments,
                    key=lambda s: s.importance or 0.0,
                    reverse=True
                )[:self.settings.max_memory_segments]
            else:  # hybrid strategy
                # Sort by combined score of importance and recency
                def hybrid_score(segment):
                    importance = segment.importance or 0.0
                    # Convert timestamp to a score (more recent = higher score)
                    if segment.timestamp:
                        ts = segment.timestamp
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        days_old = (datetime.now(timezone.utc) - ts).days
                        recency_score = max(0.0, 1.0 - (days_old / 365))  # Decay over a year
                    else:
                        recency_score = 0.0

                    return (importance * 0.7) + (recency_score * 0.3)

                self.segments = sorted(
                    self.segments,
                    key=hybrid_score,
                    reverse=True
                )[:self.settings.max_memory_segments]

            await self._save_to_disk()
            print(f"Pruned {num_to_prune} segments using {strategy} strategy.")

        # Check size-based pruning if still needed
        elif await self._should_prune_by_size():
            current_size = await self._get_memory_size_mb()
            target_size = self.settings.max_memory_size_mb * 0.7  # Prune to 70% of max size
            print(f"Memory size ({current_size:.2f} MB) exceeds threshold. Pruning to ~{target_size:.2f} MB...")

            # Remove segments until we reach target size
            original_count = len(self.segments)
            while len(self.segments) > 10:  # Keep at least 10 segments
                # Remove oldest/least important first (hybrid approach)
                if self.segments:
                    # Sort by combined score and remove lowest scoring
                    self.segments = sorted(
                        self.segments,
                        key=lambda s: (
                            (s.importance or 0.0) * 0.7 +
                            (0.3 if s.timestamp and s.timestamp > datetime.now(timezone.utc) - timedelta(days=7) else 0.0)
                        ),
                        reverse=True
                    )[:-1]  # Remove the last (lowest scoring) segment

                # Check if we've reached target size
                await self._save_to_disk()
                current_size = await self._get_memory_size_mb()
                if current_size <= target_size:
                    break

            pruned_count = original_count - len(self.segments)
            print(f"Size-based pruning complete. Removed {pruned_count} segments. New size: {current_size:.2f} MB")
        else:
            print("No pruning needed.")
            return

# --- Memory Manager (Facade) ---
class MemoryManager:
    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        self.config = config
        self.llm_interface = llm_interface
        self.backend: BaseMemoryBackend

        if config.memory_manager.backend == "basic":
            self.backend = BasicMemoryBackend(config, llm_interface)
        elif config.memory_manager.backend == "faiss_cpu":
            from .faiss_memory_backend import FaissCPUMemoryBackend
            self.backend = FaissCPUMemoryBackend(config, llm_interface)
        elif config.memory_manager.backend == "faiss_gpu":
            from .faiss_memory_backend import FaissGPUMemoryBackend
            self.backend = FaissGPUMemoryBackend(config, llm_interface)
        elif config.memory_manager.backend == "neural":
            from .neural_memory_backend import NeuralMemoryBackend
            self.backend = NeuralMemoryBackend(config, llm_interface)
        elif config.memory_manager.backend == "supabase":
            from .supabase_backend import SupabaseMemoryBackend
            self.backend = SupabaseMemoryBackend(config, llm_interface)
        elif config.memory_manager.backend == "supabase_neural":
            from .supabase_backend import SupabaseMemoryBackend
            self.backend = SupabaseMemoryBackend(config, llm_interface)
        else:
            raise ValueError(f"Unsupported memory backend: {config.memory_manager.backend}")

    async def initialize(self):
        await self.backend.initialize()

    async def add_memory(
        self,
        type: str,
        source: str,
        content_text: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_output: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        segment_content = MemorySegmentContent(
            text=content_text,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_output=tool_output
        )
        segment = MemorySegment(
            type=type,
            source=source,
            content=segment_content,
            importance=importance,
            metadata=metadata or {}
        )
        return await self.backend.add_segment(segment)

    async def get_memory(self, segment_id: str) -> Optional[MemorySegment]:
        return await self.backend.get_segment(segment_id)

    async def search_memory(
        self,
        query_text: str,
        limit: int = 5,
        min_relevance: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        return await self.backend.search_segments(query_text, limit, min_relevance, filter_dict)

    async def get_recent_memory(
        self,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        return await self.backend.get_recent_segments(limit, filter_dict)

    # Additional convenience method used by agents
    async def add_segment(self, segment: MemorySegment) -> str:
        """Direct method to add a memory segment"""
        return await self.backend.add_segment(segment)

# Example usage (for testing this file directly)
if __name__ == "__main__":
    from .config import WitsV3Config
    import httpx
    from .llm_interface import get_llm_interface

    async def main_memory_test():
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(current_script_dir)
        config_file_path = os.path.join(project_root_dir, "config.yaml")

        print(f"Loading config from: {config_file_path}")
        app_config = load_config(config_file_path)
        app_config.memory_manager.memory_file_path = os.path.join(project_root_dir, "data", "test_memory.json")
        # Ensure a clean test memory file for each run if it exists
        if os.path.exists(app_config.memory_manager.memory_file_path):
            os.remove(app_config.memory_manager.memory_file_path)

        print("Initializing LLM interface...")
        llm_interface = get_llm_interface(app_config)

        print("Initializing Memory Manager with BasicBackend...")
        memory_manager = MemoryManager(app_config, llm_interface)
        await memory_manager.initialize()
        print(f"Memory initialized. Using file: {app_config.memory_manager.memory_file_path}")

        try:
            # Add some memories
            print("\n--- Adding Memories ---")
            id1 = await memory_manager.add_memory(
                type="USER_INPUT", source="user", content_text="Hello Wits, what is the weather like?"
            )
            print(f"Added segment with ID: {id1}")
            await asyncio.sleep(0.1) # Ensure timestamps are different
            id2 = await memory_manager.add_memory(
                type="AGENT_THOUGHT", source="WeatherAgent", content_text="User asked for weather. I should call a weather tool."
            )
            print(f"Added segment with ID: {id2}")
            await asyncio.sleep(0.1)
            id3 = await memory_manager.add_memory(
                type="TOOL_CALL", source="WeatherAgent", tool_name="get_weather", tool_args={"location": "London"}
            )
            print(f"Added segment with ID: {id3}")
            await asyncio.sleep(0.1)
            id4 = await memory_manager.add_memory(
                type="TOOL_RESPONSE", source="get_weather", tool_output="The weather in London is sunny, 22°C."
            )
            print(f"Added segment with ID: {id4}")

            # Retrieve a segment
            print("\n--- Retrieving a Segment ---")
            retrieved_segment = await memory_manager.get_memory(id2)
            if retrieved_segment:
                print(f"Retrieved segment: {retrieved_segment.type} - {retrieved_segment.content.text[:30]}...")
            else:
                print(f"Segment {id2} not found.")

            # Get recent memories
            print("\n--- Recent Memories (limit 2) ---")
            recent_memories = await memory_manager.get_recent_memory(limit=2)
            for mem in recent_memories:
                print(f"- {mem.timestamp.isoformat()} - {mem.type} - {mem.source} - {(mem.content.text or mem.content.tool_output or mem.content.tool_name)[:50]}...")

            # Search memories
            print("\n--- Searching Memories (query: 'London weather') ---")
            search_results = await memory_manager.search_memory(query_text="London weather", limit=3)
            if search_results:
                for res in search_results:
                    print(f"- Score: {res.relevance_score:.4f} - Type: {res.type} - Text: {(res.content.text or res.content.tool_output)[:50]}...")
            else:
                print("No search results found.")

            print("\n--- Searching Memories (query: 'User greeting') ---")
            search_results_greeting = await memory_manager.search_memory(query_text="User greeting", limit=1)
            if search_results_greeting:
                for res in search_results_greeting:
                    print(f"- Score: {res.relevance_score:.4f} - Type: {res.type} - Text: {(res.content.text or res.content.tool_output)[:50]}...")
            else:
                print("No search results found for greeting.")

            # Test pruning (add more segments than max_memory_segments if config is low enough)
            # For this test, we'd need to set max_memory_segments very low in config or add many segments.
            # Example: app_config.memory_manager.max_memory_segments = 2
            # await memory_manager.initialize() # Re-init with new config if changed in code
            # for i in range(5):
            #     await memory_manager.add_memory(type="TEST_PRUNE", source="test", content_text=f"Test segment {i}")
            # print(f"Total segments after adding for prune test: {len(memory_manager.backend.segments)}")

        except httpx.RequestError as e:
            print(f"\nConnection Error: Could not connect to Ollama at {app_config.ollama_settings.url}.")
            print("Please ensure Ollama is running and accessible for embedding generation.")
        except Exception as e:
            print(f"\nAn unexpected error occurred during memory manager tests: {e}")
        finally:
            # Clean up the test memory file
            if os.path.exists(app_config.memory_manager.memory_file_path):
                # os.remove(app_config.memory_manager.memory_file_path)
                print(f"Test memory file at {app_config.memory_manager.memory_file_path} was not removed for inspection.")
            pass

    asyncio.run(main_memory_test())
