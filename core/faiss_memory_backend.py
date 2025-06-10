"""
FAISS CPU memory backend for WitsV3.
Provides vector similarity search capabilities for memory segments.
"""

import os
import logging
import time
import json
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path

import faiss
from pydantic import parse_obj_as

from .memory_manager import BaseMemoryBackend, MemorySegment
from .config import WitsV3Config
from .llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)

class FaissCPUMemoryBackend(BaseMemoryBackend):
    """
    FAISS CPU-based memory backend for efficient vector similarity search.

    This backend stores memory segments on disk and in a FAISS index for
    fast similarity searching based on embeddings.
    """

    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        """Initialize FAISS CPU memory backend.

        Args:
            config: WitsV3 configuration
            llm_interface: LLM interface for embeddings
        """
        super().__init__(config, llm_interface)
        self.logger = logging.getLogger(__name__)
        self.index = None
        self.index_path = Path(self.settings.faiss_index_path)
        self.memory_file_path = Path(self.settings.memory_file_path)
        self.vector_dim = self.settings.vector_dim

    async def initialize(self):
        """Initialize FAISS index and load memory segments."""
        try:
            # Create directories if they don't exist
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Load memory segments from disk
            await self._load_from_disk()

            # Create or load FAISS index
            if self.index_path.exists():
                self.logger.info(f"Loading FAISS index from {self.index_path}")
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    # Verify index dimension matches configuration
                    if self.index.d != self.vector_dim:
                        self.logger.warning(f"FAISS index dimension mismatch: index {self.index.d} != config {self.vector_dim}")
                        self.logger.warning("Recreating index with correct dimensions")
                        self._create_new_index()
                except Exception as e:
                    self.logger.error(f"Error loading FAISS index: {e}")
                    self.logger.warning("Creating new FAISS index")
                    self._create_new_index()
            else:
                self.logger.info("Creating new FAISS index")
                self._create_new_index()

            # Add existing embeddings to index
            await self._populate_index()

            self.is_initialized = True
            self.logger.info("FAISS CPU memory backend initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS CPU memory backend: {e}")
            raise

    def _create_new_index(self):
        """Create a new FAISS index with the configured dimension."""
        # L2 distance is standard for cosine similarity after normalization
        self.index = faiss.IndexFlatL2(self.vector_dim)

    async def _load_from_disk(self):
        """Load memory segments from disk."""
        try:
            if self.memory_file_path.exists():
                with open(self.memory_file_path, 'r') as f:
                    segments_data = json.load(f)
                    self.segments = parse_obj_as(List[MemorySegment], segments_data)
                    self.logger.info(f"Loaded {len(self.segments)} memory segments from {self.memory_file_path}")
            else:
                self.segments = []
                self.logger.info(f"No memory file found at {self.memory_file_path}, starting with empty memory")
        except Exception as e:
            self.logger.error(f"Error loading memory segments from disk: {e}")
            self.segments = []

    async def _save_to_disk(self):
        """Save memory segments to disk."""
        try:
            with open(self.memory_file_path, 'w') as f:
                json.dump([segment.model_dump() for segment in self.segments], f, indent=2)

            # Save FAISS index if it exists
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))

            self.logger.debug(f"Saved {len(self.segments)} memory segments to {self.memory_file_path}")
        except Exception as e:
            self.logger.error(f"Error saving memory segments to disk: {e}")
            raise

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    async def _populate_index(self):
        """Populate FAISS index with existing segment embeddings."""
        if not self.segments:
            return

        # Collect all valid embeddings
        valid_embeddings = []

        for segment in self.segments:
            if segment.embedding and len(segment.embedding) == self.vector_dim:
                # Convert to numpy and normalize
                embedding = np.array(segment.embedding, dtype=np.float32)
                # Normalize using our helper method
                embedding = self._normalize_vector(embedding)
                valid_embeddings.append(embedding)

        if valid_embeddings:
            # Concatenate all embeddings and add to index
            all_embeddings = np.vstack(valid_embeddings)
            self.index.add(all_embeddings)
            self.logger.info(f"Added {len(valid_embeddings)} embeddings to FAISS index")

    async def add_segment(self, segment: MemorySegment) -> str:
        """Add a memory segment and its embedding to the index.

        Args:
            segment: Memory segment to add

        Returns:
            ID of the added segment
        """
        if not self.is_initialized:
            await self.initialize()

        # Generate embedding if needed
        await self._generate_embedding_if_needed(segment)

        # Add segment to collection
        self.segments.append(segment)

        # Add embedding to FAISS index if available
        if segment.embedding and len(segment.embedding) == self.vector_dim:
            try:
                # Convert to numpy and normalize
                embedding = np.array([segment.embedding], dtype=np.float32)
                # Normalize using our helper method (operating on each row)
                for i in range(embedding.shape[0]):
                    embedding[i] = self._normalize_vector(embedding[i])
                # Add to index
                self.index.add(embedding)
            except Exception as e:
                self.logger.error(f"Error adding embedding to FAISS index: {e}")

        # Save to disk periodically or based on a threshold
        # For now, save on every add for simplicity
        await self._save_to_disk()

        # Prune memory if needed
        current_time = time.monotonic()
        if current_time - self.last_prune_time > self.settings.pruning_interval_seconds:
            await self.prune_memory()
            self.last_prune_time = current_time

        return segment.id

    async def get_segment(self, segment_id: str) -> Optional[MemorySegment]:
        """Get a specific memory segment by ID.

        Args:
            segment_id: ID of segment to retrieve

        Returns:
            MemorySegment if found, None otherwise
        """
        if not self.is_initialized:
            await self.initialize()

        for segment in self.segments:
            if segment.id == segment_id:
                return segment

        return None

    async def search_segments(
        self,
        query_text: str,
        limit: int = 5,
        min_relevance: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        """Search memory segments by vector similarity.

        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            min_relevance: Minimum relevance score threshold
            filter_dict: Optional metadata filters

        Returns:
            List of matching MemorySegments
        """
        if not self.is_initialized:
            await self.initialize()

        if not query_text or not self.segments:
            return []

        try:
            # Generate query embedding
            query_embedding = await self.llm_interface.get_embedding(
                query_text,
                model=self.config.ollama_settings.embedding_model
            )

            if not query_embedding or len(query_embedding) != self.vector_dim:
                self.logger.warning(f"Invalid query embedding dimension: {len(query_embedding) if query_embedding else 0}")
                return []

            # Prepare query vector
            query_vector = np.array([query_embedding], dtype=np.float32)
            # Normalize the query vector (operating on each row)
            for i in range(query_vector.shape[0]):
                query_vector[i] = self._normalize_vector(query_vector[i])

            # Perform FAISS search
            # Get more results than needed to account for filtering
            search_limit = min(len(self.segments), max(limit * 4, 20))

            # Search the index (standard FAISS API)
            distances, indices = self.index.search(query_vector, search_limit)

            # Process results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.segments):
                    continue

                segment = self.segments[idx]

                # Apply filters if provided
                if filter_dict:
                    match = True
                    for key, value in filter_dict.items():
                        segment_value = segment.metadata.get(key)
                        if segment_value != value:
                            match = False
                            break
                    if not match:
                        continue

                # Convert distance to similarity score (FAISS returns L2 distance)
                # Lower distance means higher similarity
                # Convert to a 0-1 score where 1 is most similar
                similarity = max(0.0, 1.0 - distances[0][i] / 2.0)

                if similarity >= min_relevance:
                    # Create a copy with relevance score
                    segment_copy = segment.model_copy(deep=True)
                    segment_copy.relevance_score = float(similarity)
                    results.append(segment_copy)

                    if len(results) >= limit:
                        break

            return results

        except Exception as e:
            self.logger.error(f"Error searching segments: {e}")
            return []

    async def get_recent_segments(
        self,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        """Get most recent memory segments.

        Args:
            limit: Maximum number of segments to return
            filter_dict: Optional metadata filters

        Returns:
            List of recent MemorySegments
        """
        if not self.is_initialized:
            await self.initialize()

        # Sort by timestamp (most recent first)
        sorted_segments = sorted(
            self.segments,
            key=lambda s: s.timestamp,
            reverse=True
        )

        # Apply filters if provided
        if filter_dict:
            sorted_segments = [
                s for s in sorted_segments
                if all(s.metadata.get(k) == v for k, v in filter_dict.items())
            ]

        return sorted_segments[:limit]

    async def prune_memory(self):
        """Prune memory segments to prevent unbounded growth."""
        if len(self.segments) <= self.settings.max_memory_segments:
            return

        self.logger.info(f"Pruning memory: {len(self.segments)} segments exceed limit of {self.settings.max_memory_segments}")

        # Sort by importance and timestamp (keep important and recent)
        sorted_segments = sorted(
            self.segments,
            key=lambda s: (s.importance, s.timestamp.timestamp()),
            reverse=True
        )

        # Keep only the top max_memory_segments
        pruned_segments = sorted_segments[:self.settings.max_memory_segments]
        num_pruned = len(self.segments) - len(pruned_segments)
        self.segments = pruned_segments

        # Rebuild FAISS index from scratch with remaining segments
        self._create_new_index()
        await self._populate_index()

        # Save changes to disk
        await self._save_to_disk()

        self.logger.info(f"Pruned {num_pruned} memory segments")

class FaissGPUMemoryBackend(FaissCPUMemoryBackend):
    """
    FAISS GPU-based memory backend extending the CPU implementation.

    This backend leverages GPU acceleration for faster vector similarity search.
    """

    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        """Initialize FAISS GPU memory backend.

        Args:
            config: WitsV3 configuration
            llm_interface: LLM interface for embeddings
        """
        super().__init__(config, llm_interface)
        self.gpu_resources = None

    def _create_new_index(self):
        """Create a new FAISS index with GPU support."""
        # Create CPU index first
        cpu_index = faiss.IndexFlatL2(self.vector_dim)

        try:
            # Check for available GPUs
            self.gpu_resources = faiss.StandardGpuResources()

            # Move index to GPU
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
            self.logger.info("Created FAISS GPU index successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create GPU index, falling back to CPU: {e}")
            self.index = cpu_index
