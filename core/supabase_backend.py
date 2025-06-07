"""
Supabase-powered memory backend for WitsV3
Integrates with existing vector search and neural web capabilities
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, cast
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np

from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from .memory_manager import BaseMemoryBackend, MemorySegment
from .config import WitsV3Config
from .llm_interface import BaseLLMInterface
from .neural_web_core import NeuralWeb

logger = logging.getLogger(__name__)

class SupabaseMemoryBackend(BaseMemoryBackend):
    """
    Supabase-powered memory backend that combines traditional memory storage
    with vector search and neural web capabilities
    """
    
    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface, neural_web: Optional[NeuralWeb] = None):
        """Initialize Supabase backend.
        
        Args:
            config: WitsV3 configuration
            llm_interface: LLM interface for embeddings
            neural_web: Optional neural web for enhanced search
        """
        super().__init__(config, llm_interface)
        self.neural_web = neural_web
        self.logger = logging.getLogger(__name__)
        
        # Initialize Supabase client with async support
        self.supabase: Client = create_client(
            config.supabase.url,
            config.supabase.key
        )
        self.vector_dim = config.memory_manager.vector_dim
        
    async def initialize(self) -> None:
        """Initialize Supabase tables and indexes."""
        try:
            # Create memory_segments table if not exists (no-op if exists)
            await asyncio.to_thread(
                lambda: self.supabase.table("memory_segments").insert({
                    "id": "uuid",
                    "type": "text",
                    "source": "text",
                    "content": "jsonb",
                    "importance": "float",
                    "embedding": "vector",
                    "metadata": "jsonb",
                    "created_at": "timestamp with time zone",
                    "updated_at": "timestamp with time zone"
                }).execute()
            )
            
            # Create vector index (no-op if exists)
            await asyncio.to_thread(
                lambda: self.supabase.rpc(
                    "create_vector_index",
                    {
                        "table_name": "memory_segments",
                        "column_name": "embedding",
                        "dim": self.vector_dim
                    }
                ).execute()
            )
            
            self.logger.info("Supabase backend initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Supabase backend: {e}")
            raise
    
    async def add_segment(self, segment: MemorySegment) -> str:
        """Add a memory segment to Supabase.
        
        Args:
            segment: Memory segment to add
            
        Returns:
            ID of the added segment
        """
        try:
            await self._generate_embedding_if_needed(segment)
            data = {
                "id": segment.id,
                "type": segment.type,
                "source": segment.source,
                "content": segment.content.model_dump(),
                "importance": segment.importance,
                "embedding": segment.embedding.tolist() if hasattr(segment.embedding, 'tolist') else segment.embedding,
                "metadata": segment.metadata,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            self.logger.info(f"[DEBUG] Calling Supabase add_segment with data: {data}")
            await asyncio.to_thread(
                lambda: self.supabase.table("memory_segments").insert(data).execute()
            )
            if self.neural_web:
                await self.neural_web.add_concept(
                    concept_id=segment.id,
                    content=segment.content.text or "",
                    concept_type=segment.type,
                    metadata=segment.metadata
                )
            self.logger.info(f"Added memory segment to Supabase: {segment.id}")
            return segment.id
        except Exception as e:
            self.logger.error(f"Failed to add memory segment to Supabase: {e}")
            raise
    
    async def get_segment(self, segment_id: str) -> Optional[MemorySegment]:
        """Get a memory segment from Supabase.
        
        Args:
            segment_id: ID of segment to retrieve
            
        Returns:
            MemorySegment if found, None otherwise
        """
        try:
            result = await asyncio.to_thread(
                lambda: self.supabase.table("memory_segments").select("*").eq("id", segment_id).execute()
            )
            
            if not result.data:
                return None
                
            data = result.data[0]
            return MemorySegment(
                id=data["id"],
                type=data["type"],
                source=data["source"],
                content=data["content"],
                importance=data["importance"],
                embedding=cast(List[float], data["embedding"]),
                metadata=data["metadata"]
            )
        except Exception as e:
            self.logger.error(f"Failed to get memory segment from Supabase: {e}")
            return None
    
    async def search_segments(
        self,
        query_text: str,
        limit: int = 5,
        min_relevance: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        """Search memory segments using vector similarity.
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            min_relevance: Minimum relevance score threshold
            filter_dict: Optional metadata filters
            
        Returns:
            List of matching MemorySegments
        """
        try:
            query_embedding = await self.llm_interface.get_embedding(
                query_text,
                model=self.config.ollama_settings.embedding_model
            )
            def do_query():
                # Call rpc directly on the client
                query = self.supabase.rpc(
                    "match_segments",
                    {
                        "query_embedding": query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                        "match_threshold": min_relevance,
                        "match_count": limit
                    }
                )
                if filter_dict:
                    for key, value in filter_dict.items():
                        query = query.eq(f"metadata->{key}", value)
                return query.execute()
            result = await asyncio.to_thread(do_query)
            segments = []
            for data in result.data:
                segment = MemorySegment(
                    id=data["id"],
                    type=data["type"],
                    source=data["source"],
                    content=data["content"],
                    importance=data["importance"],
                    embedding=cast(List[float], data["embedding"]),
                    metadata=data["metadata"]
                )
                segments.append(segment)
            if self.neural_web:
                for segment in segments:
                    await self.neural_web.add_concept(
                        concept_id=segment.id,
                        content=segment.content.text or "",
                        concept_type=segment.type,
                        metadata=segment.metadata
                    )
                reasoning = await self.neural_web.reason(query_text, "chain")
                if reasoning and "related_concepts" in reasoning:
                    for concept_id in reasoning["related_concepts"]:
                        if concept_id not in [s.id for s in segments]:
                            concept = await self.get_segment(concept_id)
                            if concept:
                                segments.append(concept)
            return segments
        except Exception as e:
            self.logger.error(f"Failed to search memory segments in Supabase: {e}")
            return []
    
    async def get_recent_segments(
        self,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        """Get recent memory segments.
        
        Args:
            limit: Maximum number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of recent MemorySegments
        """
        try:
            # Build query
            def do_query():
                query = self.supabase.table("memory_segments")\
                    .select("*")\
                    .order("created_at", desc=True)\
                    .limit(limit)
                    
                # Add filters if provided
                if filter_dict:
                    for key, value in filter_dict.items():
                        query = query.eq(f"metadata->{key}", value)
                    
                return query.execute()
            
            # Execute query
            result = await asyncio.to_thread(do_query)
            
            # Convert results to MemorySegments
            segments = []
            for data in result.data:
                segment = MemorySegment(
                    id=data["id"],
                    type=data["type"],
                    source=data["source"],
                    content=data["content"],
                    importance=data["importance"],
                    embedding=cast(List[float], data["embedding"]),
                    metadata=data["metadata"]
                )
                segments.append(segment)
                
            return segments
        except Exception as e:
            self.logger.error(f"Failed to get recent segments from Supabase: {e}")
            return []
    
    async def prune_memory(self) -> None:
        """Prune old memory segments."""
        try:
            # Get segments ordered by creation date
            def do_query():
                return self.supabase.table("memory_segments")\
                    .select("id")\
                    .order("created_at")\
                    .execute()
            
            # Calculate how many to prune
            result = await asyncio.to_thread(do_query)
            max_segments = self.config.memory_manager.max_memory_segments
            to_prune = max(0, len(result.data) - max_segments)
            
            if to_prune > 0:
                # Delete oldest segments
                for data in result.data[:to_prune]:
                    await asyncio.to_thread(
                        lambda: self.supabase.table("memory_segments")\
                            .delete()\
                            .eq("id", data["id"])\
                            .execute()
                    )
                        
                self.logger.info(f"Pruned {to_prune} old memory segments")
        except Exception as e:
            self.logger.error(f"Failed to prune memory segments: {e}")
            raise
    
    # Neural web methods (copied from NeuralMemoryBackend)
    async def _create_neural_concept(self, concept_id: str, segment: MemorySegment):
        """Create a neural concept from a memory segment"""
        if not self.neural_web:
            return
            
        try:
            await self.neural_web.create_concept(
                concept_id=concept_id,
                content=segment.content.text,
                metadata=segment.metadata
            )
        except Exception as e:
            logger.error(f"Failed to create neural concept: {e}")
    
    async def _auto_connect_concepts(self, new_concept_id: str, segment: MemorySegment):
        """Automatically connect new concept to existing ones"""
        if not self.neural_web:
            return
            
        try:
            connections_made = 0
            for concept_id, concept_node in self.neural_web.concepts.items():
                if concept_id == new_concept_id:
                    continue
                    
                if connections_made >= self.max_concept_connections:
                    break
                    
                try:
                    content1 = segment.content.text if segment.content else ""
                    content2 = getattr(concept_node, 'content', '') or ""
                    
                    if content1 and content2:
                        content1_words = set(content1.lower().split())
                        content2_words = set(content2.lower().split())
                        
                        # Calculate Jaccard similarity
                        intersection = len(content1_words & content2_words)
                        union = len(content1_words | content2_words)
                        
                        if union > 0:
                            similarity = intersection / union
                            if similarity >= self.connection_strength_threshold:
                                await self.neural_web.connect_concepts(
                                    source_id=new_concept_id,
                                    target_id=concept_id,
                                    relationship_type="similar",
                                    strength=similarity
                                )
                                connections_made += 1
                except Exception as e:
                    logger.debug(f"Error connecting concepts: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error in auto-connect: {e}")
    
    async def _neural_enhanced_search(self, query: str, max_results: int) -> List[MemorySegment]:
        """Use neural web reasoning to enhance search results"""
        if not self.neural_web:
            return []
            
        try:
            # Find relevant concepts using neural web
            relevant_concepts = await self.neural_web._find_relevant_concepts(query)
            
            if not relevant_concepts:
                return []
            
            # Perform reasoning to find additional relevant concepts
            reasoning_result = await self.neural_web.reason(query, "chain")
            
            # Extract additional concept IDs from reasoning results
            additional_concepts = set()
            if reasoning_result and reasoning_result.get("results"):
                for result in reasoning_result["results"]:
                    if "path" in result:
                        for concept_content in result["path"]:
                            for concept_id, concept in self.neural_web.concepts.items():
                                if concept.content == concept_content:
                                    additional_concepts.add(concept_id)
            
            # Combine all relevant concepts
            all_concepts = set(relevant_concepts) | additional_concepts
            
            # Get segments for all concepts
            neural_segments = []
            for concept_id in all_concepts:
                segment = await self.get_segment(concept_id)
                if segment:
                    neural_segments.append(segment)
            
            return neural_segments
            
        except Exception as e:
            logger.error(f"Error in neural enhanced search: {e}")
            return []
    
    async def _activate_search_concepts(self, query: str):
        """Activate concepts related to the search query"""
        if not self.neural_web:
            return
            
        try:
            relevant_concepts = await self.neural_web._find_relevant_concepts(query)
            
            if relevant_concepts:
                for concept_id in relevant_concepts:
                    if concept_id and concept_id in self.neural_web.concepts:
                        await self.neural_web.activate_concept(concept_id, 0.6)
        except Exception as e:
            logger.error(f"Error activating search concepts: {e}")
    
    def _combine_search_results(self, traditional: List[MemorySegment], 
                               neural: List[MemorySegment]) -> List[MemorySegment]:
        """Combine and rank search results from traditional and neural methods"""
        seen_ids = set()
        combined = []
        
        # Prioritize traditional results
        for segment in traditional:
            if segment.id not in seen_ids:
                combined.append(segment)
                seen_ids.add(segment.id)
        
        # Add neural results
        for segment in neural:
            if segment.id not in seen_ids:
                combined.append(segment)
                seen_ids.add(segment.id)
        
        return combined 