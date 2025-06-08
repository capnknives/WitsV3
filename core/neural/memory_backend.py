"""
Neural Memory Backend for WitsV3

This is the main neural memory backend implementation that combines all
the modular components for graph-based knowledge representation.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..memory_manager import BaseMemoryBackend, MemorySegment
from ..neural_web_core import NeuralWeb
from ..config import WitsV3Config
from ..llm_interface import BaseLLMInterface

from .concept_manager import ConceptManager
from .connection_manager import ConnectionManager
from .persistence_manager import PersistenceManager
from .similarity_utils import cosine_similarity
from .relationship_analyzer import RelationshipAnalyzer

logger = logging.getLogger(__name__)


class NeuralMemoryBackend(BaseMemoryBackend):
    """
    Neural web-powered memory backend with modular architecture.
    
    This implementation properly follows the BaseMemoryBackend interface:
    - add_segment() instead of add_memory()
    - get_segment() instead of get_memory() 
    - search_segments() instead of search_memory()
    - get_recent_segments() instead of get_recent_memory()
    """
    
    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        """
        Initialize neural memory backend with modular components.
        
        Args:
            config: System configuration
            llm_interface: LLM interface for embeddings
        """
        super().__init__(config, llm_interface)
        
        # Initialize neural web
        neural_settings = config.memory_manager.neural_web_settings
        self.neural_web = NeuralWeb(
            activation_threshold=neural_settings.activation_threshold,
            decay_rate=neural_settings.decay_rate
        )
        
        # Initialize managers
        self.concept_manager = ConceptManager(self.neural_web)
        self.connection_manager = ConnectionManager(
            self.neural_web,
            auto_connect=getattr(neural_settings, 'auto_connect', True),
            max_connections=getattr(neural_settings, 'max_concept_connections', 5),
            connection_threshold=getattr(neural_settings, 'connection_strength_threshold', 0.7)
        )
        
        # Persistence setup
        self.neural_web_path = Path(config.memory_manager.neural_web_path)
        self.persistence_manager = PersistenceManager(self.neural_web, self.neural_web_path)
        
        logger.info("NeuralMemoryBackend initialized with modular architecture")

    async def initialize(self):
        """Initialize the neural memory backend."""
        if not self.is_initialized:
            await self.persistence_manager.load_neural_web()
            self.is_initialized = True
            logger.info("Neural memory backend initialized successfully")

    # ========== REQUIRED INTERFACE METHODS ==========
    
    async def add_segment(self, segment: MemorySegment) -> str:
        """
        Add a memory segment to storage and create corresponding neural web concepts.
        
        Args:
            segment: Memory segment to add
            
        Returns:
            Segment ID
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Generate/validate ID
        if not segment.id:
            segment.id = self.concept_manager.generate_id()
            logger.debug(f"Generated new segment ID: {segment.id}")
        
        if not self.concept_manager.validate_segment_id(segment.id):
            old_id = segment.id
            segment.id = self.concept_manager.generate_id()
            logger.warning(f"Invalid segment ID '{old_id}' replaced with '{segment.id}'")
        
        # Ensure timezone-aware timestamp
        if segment.timestamp and segment.timestamp.tzinfo is None:
            segment.timestamp = segment.timestamp.replace(tzinfo=timezone.utc)
        elif not segment.timestamp:
            segment.timestamp = datetime.now(timezone.utc)
        
        # Generate embedding if needed
        if (self.llm_interface and 
            segment.content and 
            segment.content.text and 
            not segment.embedding):
            try:
                segment.embedding = await self.llm_interface.get_embedding(
                    segment.content.text,
                    model=self.config.ollama_settings.embedding_model
                )
                logger.debug(f"Generated embedding for segment {segment.id}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for segment {segment.id}: {e}")
        
        # Add to segments list
        self.segments.append(segment)
        logger.debug(f"Added segment to storage: {segment.id}")
        
        # Create neural concept
        success = await self.concept_manager.create_neural_concept(segment.id, segment)
        if not success:
            logger.warning(f"Failed to create neural concept for {segment.id}")
        
        # Auto-connect to other concepts
        connections = await self.connection_manager.auto_connect_concept(segment.id, segment)
        if connections > 0:
            logger.debug(f"Created {connections} connections for {segment.id}")
        
        # Save neural web state
        await self.persistence_manager.save_neural_web()
        
        logger.info(f"Successfully added neural memory segment: {segment.id}")
        return segment.id

    async def get_segment(self, segment_id: str) -> Optional[MemorySegment]:
        """
        Get a specific memory segment by ID.
        
        Args:
            segment_id: ID of segment to retrieve
            
        Returns:
            Memory segment or None
        """
        if not self.is_initialized:
            await self.initialize()
            
        if not self.concept_manager.validate_segment_id(segment_id):
            logger.warning(f"Invalid segment_id requested: {segment_id}")
            return None
            
        for segment in self.segments:
            if segment.id == segment_id:
                logger.debug(f"Found segment: {segment_id}")
                return segment
        
        logger.debug(f"Segment not found: {segment_id}")
        return None

    async def search_segments(
        self, 
        query_text: str, 
        limit: int = 5, 
        min_relevance: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        """
        Search memory segments by similarity to query text.
        
        Args:
            query_text: Text to search for
            limit: Maximum results to return
            min_relevance: Minimum similarity score
            filter_dict: Optional filters
            
        Returns:
            List of matching segments
        """
        if not self.is_initialized:
            await self.initialize()
            
        if not query_text or not query_text.strip():
            logger.debug("Empty query text provided to search_segments")
            return []
        
        try:
            # Get query embedding
            if not self.llm_interface:
                logger.warning("No LLM interface available for embedding generation")
                return []
                
            query_embedding = await self.llm_interface.get_embedding(
                query_text,
                model=self.config.ollama_settings.embedding_model
            )
            
            results = []
            for segment in self.segments:
                if not segment.embedding:
                    continue
                    
                # Calculate similarity
                try:
                    similarity = cosine_similarity(query_embedding, segment.embedding)
                    if similarity >= min_relevance:
                        results.append((segment, similarity))
                except Exception as e:
                    logger.debug(f"Error calculating similarity for segment {segment.id}: {e}")
                    continue
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            found_segments = [segment for segment, _ in results[:limit]]
            
            logger.debug(f"Found {len(found_segments)} segments matching query")
            return found_segments
            
        except Exception as e:
            logger.error(f"Error in segment search: {e}")
            return []

    async def get_recent_segments(
        self, 
        limit: int = 10, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[MemorySegment]:
        """
        Get the most recent memory segments.
        
        Args:
            limit: Maximum segments to return
            filter_dict: Optional filters
            
        Returns:
            List of recent segments
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Sort by timestamp (most recent first)
            sorted_segments = sorted(
                self.segments, 
                key=lambda s: s.timestamp or datetime.min.replace(tzinfo=timezone.utc), 
                reverse=True
            )
            
            recent_segments = sorted_segments[:limit]
            logger.debug(f"Retrieved {len(recent_segments)} recent segments")
            return recent_segments
            
        except Exception as e:
            logger.error(f"Error getting recent segments: {e}")
            return []

    async def prune_memory(self):
        """
        Prune old memory segments based on configuration.
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get pruning settings
            max_segments = getattr(self.config.memory_manager, 'max_memory_segments', 1000)
            
            if len(self.segments) <= max_segments:
                logger.debug(f"No pruning needed. Current: {len(self.segments)}, max: {max_segments}")
                return
            
            # Calculate how many to prune
            num_to_prune = len(self.segments) - max_segments
            logger.info(f"Pruning {num_to_prune} segments")
            
            # Sort by timestamp and importance
            sorted_segments = sorted(
                self.segments, 
                key=lambda s: (
                    s.timestamp or datetime.min.replace(tzinfo=timezone.utc),
                    s.importance or 0.0
                ), 
                reverse=True
            )
            
            # Keep the top segments, remove the rest
            segments_to_keep = sorted_segments[:max_segments]
            segments_to_remove = sorted_segments[max_segments:]
            
            # Update segments list
            self.segments = segments_to_keep
            
            # Remove corresponding neural concepts
            for segment in segments_to_remove:
                await self.concept_manager.remove_concept(segment.id)
            
            # Save updated neural web
            await self.persistence_manager.save_neural_web()
            
            logger.info(f"Pruning complete. Removed {num_to_prune} segments")
            
        except Exception as e:
            logger.error(f"Error during memory pruning: {e}")