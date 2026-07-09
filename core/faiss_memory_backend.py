"""
FAISS CPU memory backend for WitsV3.
Provides vector similarity search capabilities for memory segments.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from pydantic import TypeAdapter

from .config import WitsV3Config
from .llm_interface import BaseLLMInterface
from .memory_manager import BaseMemoryBackend, MemorySegment

logger = logging.getLogger(__name__)


def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    """Rename temp file into place (Windows-safe replace)."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        final_path.unlink()
    tmp_path.replace(final_path)


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
        self._io_lock = asyncio.Lock()
        self._json_load_failed = False
        self._persist_pending = False

    async def initialize(self):
        """Initialize FAISS index and load memory segments."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_file_path.parent.mkdir(parents=True, exist_ok=True)

            await self._load_from_disk()

            if self._json_load_failed:
                self.logger.warning(
                    "Memory JSON load failed — discarding stale FAISS index and rebuilding"
                )
                self._quarantine_stale_index()
                self._create_new_index()
            elif self.index_path.exists():
                self.logger.info(f"Loading FAISS index from {self.index_path}")
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    if self.index.d != self.vector_dim:
                        self.logger.warning(
                            f"FAISS index dimension mismatch: index {self.index.d} "
                            f"!= config {self.vector_dim}"
                        )
                        self._create_new_index()
                except Exception as e:
                    self.logger.error(f"Error loading FAISS index: {e}")
                    self._create_new_index()
            else:
                self.logger.info("Creating new FAISS index")
                self._create_new_index()

            await self._sync_index_to_segments()

            self.is_initialized = True
            self.logger.info("FAISS CPU memory backend initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS CPU memory backend: {e}")
            raise

    def _quarantine_stale_index(self) -> None:
        if not self.index_path.exists():
            return
        stale = self.index_path.with_suffix(".bin.stale")
        try:
            if stale.exists():
                stale.unlink()
            self.index_path.replace(stale)
            self.logger.info("Quarantined stale FAISS index to %s", stale)
        except OSError as exc:
            self.logger.warning("Could not quarantine stale index: %s", exc)
            try:
                self.index_path.unlink()
            except OSError:
                pass

    def _create_new_index(self):
        """Create a new FAISS index with the configured dimension."""
        self.index = faiss.IndexFlatL2(self.vector_dim)

    async def _load_from_disk(self):
        """Load memory segments from disk."""
        self._json_load_failed = False
        try:
            if self.memory_file_path.exists():
                async with self._io_lock:
                    with open(self.memory_file_path, encoding="utf-8") as f:
                        segments_data = json.load(f)
                if isinstance(segments_data, dict):
                    segments_data = segments_data.get("segments", [])
                self.segments = TypeAdapter(list[MemorySegment]).validate_python(segments_data)
                self.logger.info(
                    f"Loaded {len(self.segments)} memory segments from {self.memory_file_path}"
                )
            else:
                self.segments = []
                self.logger.info(
                    f"No memory file found at {self.memory_file_path}, starting with empty memory"
                )
        except Exception as e:
            self.logger.error(f"Error loading memory segments from disk: {e}")
            self._json_load_failed = True
            self.segments = []

    async def _save_to_disk(self):
        """Save memory segments and FAISS index atomically."""
        async with self._io_lock:
            try:
                segments_data = [
                    segment.model_dump(mode="json") for segment in self.segments
                ]
                json_tmp = self.memory_file_path.with_suffix(".json.tmp")
                json_tmp.write_text(
                    json.dumps(segments_data, indent=2), encoding="utf-8"
                )
                _atomic_replace(json_tmp, self.memory_file_path)

                if self.index is not None:
                    index_tmp = self.index_path.with_suffix(".bin.tmp")
                    faiss.write_index(self.index, str(index_tmp))
                    _atomic_replace(index_tmp, self.index_path)

                self._persist_pending = False
                self.logger.debug(
                    f"Saved {len(self.segments)} memory segments to {self.memory_file_path}"
                )
            except Exception as e:
                self.logger.error(f"Error saving memory segments to disk: {e}")
                raise

    async def flush_persist(self) -> None:
        """Write pending segment adds to disk."""
        if self._persist_pending:
            await self._save_to_disk()

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def _embedded_segment_count(self) -> int:
        return sum(
            1
            for segment in self.segments
            if segment.embedding and len(segment.embedding) == self.vector_dim
        )

    async def _sync_index_to_segments(self) -> None:
        """Ensure FAISS index matches in-memory segments (rebuild if mismatched)."""
        if self.index is None:
            self._create_new_index()
        expected = self._embedded_segment_count()
        if self.index.ntotal != expected:
            if self.index.ntotal > 0:
                self.logger.warning(
                    "FAISS index has %s vectors but %s embedded segments — rebuilding",
                    self.index.ntotal,
                    expected,
                )
            self._create_new_index()
            await self._populate_index()

    async def delete_segments(self, filter_dict) -> int:
        """Delete matching segments and rebuild the FAISS index."""
        removed = await super().delete_segments(filter_dict)
        if removed:
            self._create_new_index()
            await self._populate_index()
            await self._save_to_disk()
        return removed

    async def _populate_index(self):
        """Populate FAISS index with existing segment embeddings."""
        if not self.segments or self.index is None:
            return

        valid_embeddings = []
        for segment in self.segments:
            if segment.embedding and len(segment.embedding) == self.vector_dim:
                embedding = np.array(segment.embedding, dtype=np.float32)
                embedding = self._normalize_vector(embedding)
                valid_embeddings.append(embedding)

        if valid_embeddings:
            all_embeddings = np.vstack(valid_embeddings)
            self.index.add(all_embeddings)
            self.logger.info(f"Added {len(valid_embeddings)} embeddings to FAISS index")

    async def add_segment(self, segment: MemorySegment, *, persist: bool = True) -> str:
        """Add a memory segment and its embedding to the index."""
        if not self.is_initialized:
            await self.initialize()

        await self._generate_embedding_if_needed(segment)
        self.segments.append(segment)

        if segment.embedding and len(segment.embedding) == self.vector_dim:
            try:
                embedding = np.array([segment.embedding], dtype=np.float32)
                for i in range(embedding.shape[0]):
                    embedding[i] = self._normalize_vector(embedding[i])
                self.index.add(embedding)
            except Exception as e:
                self.logger.error(f"Error adding embedding to FAISS index: {e}")

        if persist:
            await self._save_to_disk()
        else:
            self._persist_pending = True

        current_time = time.monotonic()
        if current_time - self.last_prune_time > self.settings.pruning_interval_seconds:
            await self.prune_memory()
            self.last_prune_time = current_time

        return segment.id

    async def get_segment(self, segment_id: str) -> MemorySegment | None:
        """Get a specific memory segment by ID."""
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
        filter_dict: dict[str, Any] | None = None,
    ) -> list[MemorySegment]:
        """Search memory segments by vector similarity."""
        if not self.is_initialized:
            await self.initialize()

        if not query_text or not self.segments:
            return []

        try:
            query_embedding = await self.llm_interface.get_embedding(
                query_text, model=self.config.ollama_settings.embedding_model
            )

            if not query_embedding or len(query_embedding) != self.vector_dim:
                self.logger.warning(
                    f"Invalid query embedding dimension: "
                    f"{len(query_embedding) if query_embedding else 0}"
                )
                return []

            query_vector = np.array([query_embedding], dtype=np.float32)
            for i in range(query_vector.shape[0]):
                query_vector[i] = self._normalize_vector(query_vector[i])

            search_limit = min(len(self.segments), max(limit * 4, 20))
            distances, indices = self.index.search(query_vector, search_limit)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.segments):
                    continue

                segment = self.segments[idx]

                if filter_dict:
                    match = True
                    for key, value in filter_dict.items():
                        segment_value = segment.metadata.get(key)
                        if segment_value != value:
                            match = False
                            break
                    if not match:
                        continue

                similarity = max(0.0, 1.0 - distances[0][i] / 2.0)

                if similarity >= min_relevance:
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
        self, limit: int = 10, filter_dict: dict[str, Any] | None = None
    ) -> list[MemorySegment]:
        """Get most recent memory segments."""
        if not self.is_initialized:
            await self.initialize()

        sorted_segments = sorted(self.segments, key=lambda s: s.timestamp, reverse=True)

        if filter_dict:
            sorted_segments = [
                s
                for s in sorted_segments
                if all(s.metadata.get(k) == v for k, v in filter_dict.items())
            ]

        return sorted_segments[:limit]

    async def prune_memory(self):
        """Prune memory segments to prevent unbounded growth."""
        if len(self.segments) <= self.settings.max_memory_segments:
            return

        self.logger.info(
            f"Pruning memory: {len(self.segments)} segments exceed limit of "
            f"{self.settings.max_memory_segments}"
        )

        sorted_segments = sorted(
            self.segments, key=lambda s: (s.importance, s.timestamp.timestamp()), reverse=True
        )

        pruned_segments = sorted_segments[: self.settings.max_memory_segments]
        num_pruned = len(self.segments) - len(pruned_segments)
        self.segments = pruned_segments

        self._create_new_index()
        await self._populate_index()
        await self._save_to_disk()

        self.logger.info(f"Pruned {num_pruned} memory segments")


class FaissGPUMemoryBackend(FaissCPUMemoryBackend):
    """
    FAISS GPU-based memory backend extending the CPU implementation.

    This backend leverages GPU acceleration for faster vector similarity search.
    """

    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        """Initialize FAISS GPU memory backend."""
        super().__init__(config, llm_interface)
        self.gpu_resources = None

    def _create_new_index(self):
        """Create a new FAISS index with GPU support."""
        cpu_index = faiss.IndexFlatL2(self.vector_dim)

        try:
            self.gpu_resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
            self.logger.info("Created FAISS GPU index successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create GPU index, falling back to CPU: {e}")
            self.index = cpu_index
