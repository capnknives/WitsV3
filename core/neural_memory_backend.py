# core/neural_memory_backend.py
"""
Neural Web-powered memory backend for WitsV3
Integrates graph-based knowledge representation with traditional memory management

This implementation properly follows the BaseMemoryBackend interface:
- add_segment() instead of add_memory()
- get_segment() instead of get_memory() 
- search_segments() instead of search_memory()
- get_recent_segments() instead of get_recent_memory()

Key fixes for None ID errors:
1. Robust ID generation and validation
2. Proper error handling for None values
3. Enhanced logging for debugging
4. Safe neural web operations with validation
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pathlib import Path

from .memory_manager import BaseMemoryBackend, MemorySegment
from .neural_web_core import NeuralWeb, ConceptNode
from .config import WitsV3Config
from .llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)


class NeuralMemoryBackend(BaseMemoryBackend):
    """
    Neural web-powered memory backend that properly implements BaseMemoryBackend interface
    and provides robust ID validation to prevent None segment errors.
    """
    
    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        super().__init__(config, llm_interface)
        
        # Initialize neural web
        neural_settings = config.memory_manager.neural_web_settings
        self.neural_web = NeuralWeb(
            activation_threshold=neural_settings.activation_threshold,
            decay_rate=neural_settings.decay_rate
        )
        
        # Configuration from neural web settings
        self.auto_connect = getattr(neural_settings, 'auto_connect', True)
        self.max_concept_connections = getattr(neural_settings, 'max_concept_connections', 5)
        self.connection_strength_threshold = getattr(neural_settings, 'connection_strength_threshold', 0.7)
        
        # Persistence setup
        self.neural_web_path = Path(config.memory_manager.neural_web_path)
        self.neural_web_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("NeuralMemoryBackend initialized with proper interface methods")

    def _generate_id(self) -> str:
        """Generate a robust, unique ID for memory segments."""
        return f"seg_{uuid.uuid4().hex[:12]}"

    def _validate_segment_id(self, segment_id: str) -> bool:
        """Validate that a segment ID is proper and not None/empty."""
        if not segment_id:
            return False
        if segment_id in ["null", "None", "", "undefined"]:
            return False
        if not isinstance(segment_id, str):
            return False
        return True

    async def initialize(self):
        """Initialize the neural memory backend."""
        if not self.is_initialized:
            await self._load_neural_web()
            self.is_initialized = True
            logger.info("Neural memory backend initialized successfully")

    # ========== REQUIRED INTERFACE METHODS ==========
    
    async def add_segment(self, segment: MemorySegment) -> str:
        """
        Add a memory segment to storage and create corresponding neural web concepts.
        This is the CORRECT interface method (not add_memory).
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Enhanced ID generation and validation
        if not segment.id:
            segment.id = self._generate_id()
            logger.debug(f"Generated new segment ID: {segment.id}")
        
        # Validate the segment ID to prevent None errors
        if not self._validate_segment_id(segment.id):
            old_id = segment.id
            segment.id = self._generate_id()
            logger.warning(f"Invalid segment ID '{old_id}' replaced with '{segment.id}'")
        
        # Ensure timezone-aware timestamp
        if segment.timestamp and segment.timestamp.tzinfo is None:
            segment.timestamp = segment.timestamp.replace(tzinfo=timezone.utc)
        elif not segment.timestamp:
            segment.timestamp = datetime.now(timezone.utc)
        
        # Generate embedding if needed and possible
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
        
        # Create neural concept with robust validation
        try:
            await self._create_neural_concept(segment.id, segment)
        except Exception as e:
            logger.error(f"Failed to create neural concept for {segment.id}: {e}")
        
        # Auto-connect to other concepts if enabled
        if self.auto_connect:
            try:
                await self._auto_connect_concepts(segment.id, segment)
            except Exception as e:
                logger.error(f"Failed to auto-connect concept {segment.id}: {e}")
        
        # Save neural web state
        try:
            await self._save_neural_web()
        except Exception as e:
            logger.error(f"Failed to save neural web after adding {segment.id}: {e}")
        
        logger.info(f"Successfully added neural memory segment: {segment.id}")
        return segment.id

    async def get_segment(self, segment_id: str) -> Optional[MemorySegment]:
        """
        Get a specific memory segment by ID.
        This is the CORRECT interface method (not get_memory).
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Validate input to prevent None errors
        if not self._validate_segment_id(segment_id):
            logger.warning(f"Invalid segment_id requested: {segment_id}")
            return None
            
        # Search for the segment
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
        This is the CORRECT interface method (not search_memory).
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
                    similarity = self._cosine_similarity(query_embedding, segment.embedding)
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
        This is the CORRECT interface method (not get_recent_memory).
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
        This is a REQUIRED method from BaseMemoryBackend interface.
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get pruning settings
            max_segments = getattr(self.config.memory_manager, 'max_memory_segments', 1000)
            
            if len(self.segments) <= max_segments:
                logger.debug(f"No pruning needed. Current segments: {len(self.segments)}, max: {max_segments}")
                return
            
            # Calculate how many to prune
            num_to_prune = len(self.segments) - max_segments
            logger.info(f"Pruning {num_to_prune} segments (current: {len(self.segments)}, max: {max_segments})")
            
            # Sort by timestamp and importance (keep most recent and important)
            sorted_segments = sorted(
                self.segments, 
                key=lambda s: (
                    s.timestamp or datetime.min.replace(tzinfo=timezone.utc),
                    s.importance or 0.0
                ), 
                reverse=True  # Most recent and important first
            )
            
            # Keep the top segments, remove the rest
            segments_to_keep = sorted_segments[:max_segments]
            segments_to_remove = sorted_segments[max_segments:]
            
            # Update segments list
            self.segments = segments_to_keep
            
            # Remove corresponding neural concepts
            for segment in segments_to_remove:
                try:
                    if (hasattr(self.neural_web, 'concepts') and 
                        segment.id in self.neural_web.concepts):
                        # Remove connections first
                        connections_to_remove = []
                        for (source_id, target_id) in self.neural_web.connections:
                            if source_id == segment.id or target_id == segment.id:
                                connections_to_remove.append((source_id, target_id))
                        
                        for conn_key in connections_to_remove:
                            del self.neural_web.connections[conn_key]
                        
                        # Remove concept
                        del self.neural_web.concepts[segment.id]
                        logger.debug(f"Removed neural concept: {segment.id}")
                        
                except Exception as e:
                    logger.debug(f"Error removing neural concept {segment.id}: {e}")
            
            # Save updated neural web
            await self._save_neural_web()
            
            logger.info(f"Pruning complete. Removed {num_to_prune} segments, kept {len(self.segments)} segments")
            
        except Exception as e:
            logger.error(f"Error during memory pruning: {e}")

    # ========== NEURAL WEB METHODS ==========
    
    async def _create_neural_concept(self, segment_id: str, segment: MemorySegment):
        """
        Create or update a neural web concept from a memory segment.
        Enhanced with robust validation to prevent None ID errors.
        """
        # Enhanced validation - this is where the None errors were coming from
        if not self._validate_segment_id(segment_id):
            logger.error(f"VALIDATION FAILED - Invalid segment_id: '{segment_id}', cannot create neural concept")
            return
            
        if not segment:
            logger.error(f"VALIDATION FAILED - Invalid segment provided for segment_id: {segment_id}")
            return
            
        if not segment.content:
            logger.warning(f"Segment {segment_id} has no content, creating minimal concept")
        
        try:
            # Determine concept type
            concept_type = self._determine_concept_type(segment)
            
            # Create concept metadata with safe access
            metadata_for_concept = {
                "content_preview": (segment.content.text or "")[:100] if segment.content else "",
                "timestamp": segment.timestamp.isoformat() if segment.timestamp else datetime.now(timezone.utc).isoformat(),
                "source": getattr(segment, 'source', '') or "",
                "memory_type": getattr(segment, 'type', '') or "",
                "original_metadata": getattr(segment, 'metadata', {}) or {},
                "has_embedding": bool(segment.embedding),
                "segment_id_ref": segment.id
            }
            
            # Safe content extraction
            content_text = ""
            if segment.content and segment.content.text:
                content_text = segment.content.text
            elif segment.content:
                # Try other content fields if text is not available
                content_text = str(segment.content)[:200]
              # Add concept to neural web with validation (NO embedding parameter)
            if hasattr(self.neural_web, 'add_concept'):
                await self.neural_web.add_concept(
                    concept_id=segment_id,
                    content=content_text,
                    concept_type=concept_type,
                    metadata=metadata_for_concept
                )
                logger.debug(f"SUCCESS - Created neural concept for segment: {segment_id}")
            else:
                logger.warning(f"Neural web does not support add_concept method")
            
        except Exception as e:
            logger.error(f"EXCEPTION - Failed to create neural concept for segment {segment_id}: {e}")
            # Don't re-raise - this should not break the main flow

    async def _auto_connect_concepts(self, new_concept_id: str, segment: MemorySegment):
        """
        Auto-connect the new concept to related existing concepts.
        Enhanced with robust validation to prevent None ID errors.
        """
        # Enhanced validation - this is where the None errors were coming from
        if not self._validate_segment_id(new_concept_id):
            logger.error(f"VALIDATION FAILED - Invalid new_concept_id: '{new_concept_id}', cannot auto-connect")
            return
            
        if not segment:
            logger.error(f"VALIDATION FAILED - Invalid segment provided for concept_id: {new_concept_id}")
            return
            
        try:
            # Check if neural web has concepts and our concept exists
            if not hasattr(self.neural_web, 'concepts'):
                logger.debug("Neural web has no concepts attribute")
                return
                
            if new_concept_id not in self.neural_web.concepts:
                logger.warning(f"New concept {new_concept_id} not found in neural web")
                return
                
            new_concept = self.neural_web.concepts[new_concept_id]
            
            # Find similar concepts to connect to
            connections_made = 0
            for concept_id, concept_node in self.neural_web.concepts.items():
                if concept_id == new_concept_id:
                    continue
                    
                if connections_made >= self.max_concept_connections:
                    break
                  # Use content-based similarity instead of embeddings (ConceptNode has no embedding)
                try:
                    content1 = segment.content.text if segment.content else ""
                    content2 = getattr(concept_node, 'content', '') or ""
                    
                    if content1 and content2:  # Both must have content
                        # Simple content similarity (can be enhanced)
                        content1_words = set(content1.lower().split())
                        content2_words = set(content2.lower().split())
                        
                        if content1_words and content2_words:
                            overlap = len(content1_words.intersection(content2_words))
                            union = len(content1_words.union(content2_words))
                            similarity = overlap / union if union > 0 else 0.0
                            
                            if similarity >= self.connection_strength_threshold:
                                # Determine relationship type with proper string validation
                                relationship_type = self._determine_relationship_type(
                                    content1 or "", 
                                    content2 or ""
                                )
                                
                                # Create connection using correct method
                                if hasattr(self.neural_web, 'connect_concepts'):
                                    await self.neural_web.connect_concepts(
                                        source_id=new_concept_id, 
                                        target_id=concept_id, 
                                        relationship_type=relationship_type, 
                                        strength=similarity
                                    )
                                    connections_made += 1
                                    logger.debug(f"Connected {new_concept_id} to {concept_id} (similarity: {similarity:.3f})")
                            
                except Exception as e:
                    logger.debug(f"Error connecting to concept {concept_id}: {e}")
                    continue
                        
            logger.debug(f"SUCCESS - Auto-connected concept {new_concept_id} to {connections_made} other concepts")
            
        except Exception as e:
            logger.error(f"EXCEPTION - Failed to auto-connect concept {new_concept_id}: {e}")
            # Don't re-raise - this should not break the main flow

    def _determine_concept_type(self, segment: MemorySegment) -> str:
        """Determine the type of concept based on segment content."""
        try:
            if not segment or not segment.content:
                return "unknown"
                
            content_text = segment.content.text or ""
            content_lower = content_text.lower()
            
            # Simple heuristics for concept type determination
            if any(word in content_lower for word in ["question", "?", "how", "what", "why", "when", "where"]):
                return "question"
            elif any(word in content_lower for word in ["task", "todo", "action", "do", "complete", "finish"]):
                return "task"
            elif any(word in content_lower for word in ["fact", "is", "are", "definition", "means"]):
                return "fact"
            elif any(word in content_lower for word in ["thought", "think", "believe", "opinion"]):
                return "thought"
            elif any(word in content_lower for word in ["error", "problem", "issue", "bug"]):
                return "problem"
            else:
                return "information"
                
        except Exception as e:
            logger.debug(f"Error determining concept type: {e}")
            return "unknown"

    def _determine_relationship_type(self, content1: str, content2: str) -> str:
        """Determine the type of relationship between two pieces of content."""
        try:
            if not content1 or not content2:
                return "related"
                
            c1_lower = content1.lower()
            c2_lower = content2.lower()
            
            # Check for causal relationships
            causal_words = ["because", "caused", "due to", "result", "leads to", "triggers"]
            if any(word in c1_lower for word in causal_words) or \
               any(word in c2_lower for word in causal_words):
                return "causal"
            
            # Check for temporal relationships
            temporal_words = ["before", "after", "then", "next", "previously", "following"]
            if any(word in c1_lower for word in temporal_words) or \
               any(word in c2_lower for word in temporal_words):
                return "temporal"
            
            # Check for similarity
            similarity_words = ["similar", "like", "same", "resembles", "comparable"]
            if any(word in c1_lower for word in similarity_words) or \
               any(word in c2_lower for word in similarity_words):
                return "similarity"
            
            # Check for contrast
            contrast_words = ["different", "unlike", "however", "but", "opposite"]
            if any(word in c1_lower for word in contrast_words) or \
               any(word in c2_lower for word in contrast_words):
                return "contrast"
            
            return "semantic"
            
        except Exception as e:
            logger.debug(f"Error determining relationship type: {e}")
            return "related"

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors with robust error handling."""
        try:
            if not vec1 or not vec2:
                return 0.0
                
            if len(vec1) != len(vec2):
                logger.debug(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
                return 0.0
            
            # Calculate dot product and magnitudes
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(a * a for a in vec2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            similarity = dot_product / (magnitude1 * magnitude2)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.debug(f"Error calculating cosine similarity: {e}")
            return 0.0    # ========== PERSISTENCE METHODS ==========
    
    async def _save_neural_web(self):
        """Save the neural web to disk using custom serialization."""
        try:
            # Custom serialization since neural web doesn't have to_dict
            neural_data = {
                "concepts": {
                    concept_id: {
                        "id": concept.id,
                        "content": concept.content,
                        "concept_type": concept.concept_type,
                        "activation_level": concept.activation_level,
                        "base_strength": concept.base_strength,
                        "metadata": concept.metadata,
                        "created_at": concept.created_at.isoformat(),
                        "last_accessed": concept.last_accessed.isoformat(),
                        "access_count": concept.access_count
                    }
                    for concept_id, concept in self.neural_web.concepts.items()
                },
                "connections": {
                    f"{conn.source_id}->{conn.target_id}": {
                        "source_id": conn.source_id,
                        "target_id": conn.target_id,
                        "relationship_type": conn.relationship_type,
                        "strength": conn.strength,
                        "confidence": conn.confidence,
                        "created_at": conn.created_at.isoformat(),
                        "reinforcement_count": conn.reinforcement_count
                    }
                    for (source_id, target_id), conn in self.neural_web.connections.items()
                }
            }
            
            with open(self.neural_web_path, 'w', encoding='utf-8') as f:
                json.dump(neural_data, f, indent=2, ensure_ascii=False)
            logger.debug("Neural web saved to disk successfully")
            
        except Exception as e:
            logger.error(f"Failed to save neural web: {e}")

    async def _load_neural_web(self):
        """Load the neural web from disk using custom deserialization."""
        try:
            if self.neural_web_path.exists():
                with open(self.neural_web_path, 'r', encoding='utf-8') as f:
                    neural_data = json.load(f)
                
                # Custom deserialization - restore concepts first
                for concept_id, concept_data in neural_data.get("concepts", {}).items():
                    # Skip concepts with invalid IDs
                    stored_concept_id = concept_data.get("id")
                    if stored_concept_id is None or stored_concept_id == "null" or stored_concept_id == "":
                        import uuid
                        valid_concept_id = f"concept_{uuid.uuid4().hex[:8]}"
                        logger.warning(f"Found concept with invalid ID ({stored_concept_id}), assigning new ID: {valid_concept_id}")
                    else:
                        valid_concept_id = stored_concept_id
                    
                    await self.neural_web.add_concept(
                        concept_id=valid_concept_id,
                        content=concept_data["content"],
                        concept_type=concept_data["concept_type"],
                        metadata=concept_data["metadata"]
                    )
                    
                    # Restore concept state
                    concept = self.neural_web.concepts[valid_concept_id]
                    concept.activation_level = concept_data["activation_level"]
                    concept.base_strength = concept_data["base_strength"]
                    concept.created_at = datetime.fromisoformat(concept_data["created_at"])
                    concept.last_accessed = datetime.fromisoformat(concept_data["last_accessed"])
                    concept.access_count = concept_data["access_count"]
                
                # Restore connections
                for conn_key, conn_data in neural_data.get("connections", {}).items():
                    source_id = conn_data["source_id"]
                    target_id = conn_data["target_id"]
                    
                    # Skip connections with invalid IDs
                    if (source_id is None or source_id == "null" or source_id == "" or
                        target_id is None or target_id == "null" or target_id == ""):
                        logger.warning(f"Skipping connection with invalid IDs: {source_id} -> {target_id}")
                        continue
                    
                    # Only create connection if both concepts exist
                    if source_id in self.neural_web.concepts and target_id in self.neural_web.concepts:
                        await self.neural_web.connect_concepts(
                            source_id=source_id,
                            target_id=target_id,
                            relationship_type=conn_data["relationship_type"],
                            strength=conn_data["strength"],
                            confidence=conn_data["confidence"]
                        )
                        
                        # Restore connection state
                        connection = self.neural_web.connections[(source_id, target_id)]
                        connection.created_at = datetime.fromisoformat(conn_data["created_at"])
                        connection.reinforcement_count = conn_data["reinforcement_count"]
                    else:
                        logger.warning(f"Skipping connection {source_id} -> {target_id}: one or both concepts don't exist")
                
                logger.info(f"Loaded neural web with {len(self.neural_web.concepts)} concepts and {len(self.neural_web.connections)} connections")
            else:
                logger.debug("No existing neural web found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load neural web: {e}")
            logger.info("Continuing with fresh neural web")