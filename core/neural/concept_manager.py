"""
Neural concept management for memory backend
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from ..memory_manager import MemorySegment
from ..neural_web_core import NeuralWeb
from .relationship_analyzer import RelationshipAnalyzer

logger = logging.getLogger(__name__)


class ConceptManager:
    """Manages neural concept creation, validation, and manipulation"""
    
    def __init__(self, neural_web: NeuralWeb):
        """
        Initialize concept manager.
        
        Args:
            neural_web: Neural web instance to manage concepts for
        """
        self.neural_web = neural_web
        self.relationship_analyzer = RelationshipAnalyzer()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate_id(self) -> str:
        """Generate a robust, unique ID for concepts."""
        return f"seg_{uuid.uuid4().hex[:12]}"
    
    def validate_segment_id(self, segment_id: str) -> bool:
        """
        Validate that a segment ID is proper and not None/empty.
        
        Args:
            segment_id: ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not segment_id:
            return False
        if segment_id in ["null", "None", "", "undefined"]:
            return False
        if not isinstance(segment_id, str):
            return False
        return True
    
    async def create_neural_concept(self, segment_id: str, segment: MemorySegment) -> bool:
        """
        Create or update a neural web concept from a memory segment.
        
        Args:
            segment_id: ID for the concept
            segment: Memory segment to create concept from
            
        Returns:
            True if successful, False otherwise
        """
        # Enhanced validation
        if not self.validate_segment_id(segment_id):
            self.logger.error(f"Invalid segment_id: '{segment_id}', cannot create neural concept")
            return False
            
        if not segment:
            self.logger.error(f"Invalid segment provided for segment_id: {segment_id}")
            return False
            
        if not segment.content:
            self.logger.warning(f"Segment {segment_id} has no content, creating minimal concept")
        
        try:
            # Determine concept type
            concept_type = self.relationship_analyzer.determine_concept_type(segment)
            
            # Create concept metadata with safe access
            metadata = self._build_concept_metadata(segment)
            
            # Safe content extraction
            content_text = self._extract_content_text(segment)
            
            # Add concept to neural web
            if hasattr(self.neural_web, 'add_concept'):
                await self.neural_web.add_concept(
                    concept_id=segment_id,
                    content=content_text,
                    concept_type=concept_type,
                    metadata=metadata
                )
                self.logger.debug(f"Created neural concept for segment: {segment_id}")
                return True
            else:
                self.logger.warning(f"Neural web does not support add_concept method")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to create neural concept for segment {segment_id}: {e}")
            return False
    
    def _build_concept_metadata(self, segment: MemorySegment) -> Dict[str, Any]:
        """
        Build metadata for a concept from a memory segment.
        
        Args:
            segment: Memory segment
            
        Returns:
            Metadata dictionary
        """
        return {
            "content_preview": (segment.content.text or "")[:100] if segment.content else "",
            "timestamp": segment.timestamp.isoformat() if segment.timestamp else datetime.now(timezone.utc).isoformat(),
            "source": getattr(segment, 'source', '') or "",
            "memory_type": getattr(segment, 'type', '') or "",
            "original_metadata": getattr(segment, 'metadata', {}) or {},
            "has_embedding": bool(segment.embedding),
            "segment_id_ref": segment.id
        }
    
    def _extract_content_text(self, segment: MemorySegment) -> str:
        """
        Extract text content from a memory segment.
        
        Args:
            segment: Memory segment
            
        Returns:
            Extracted text content
        """
        if segment.content and segment.content.text:
            return segment.content.text
        elif segment.content:
            # Try other content fields if text is not available
            return str(segment.content)[:200]
        return ""
    
    async def remove_concept(self, concept_id: str) -> bool:
        """
        Remove a concept and its connections from the neural web.
        
        Args:
            concept_id: ID of concept to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not hasattr(self.neural_web, 'concepts'):
                return False
                
            if concept_id not in self.neural_web.concepts:
                return False
            
            # Remove connections first
            connections_to_remove = []
            
            if hasattr(self.neural_web, 'connections'):
                for (source_id, target_id) in self.neural_web.connections:
                    if source_id == concept_id or target_id == concept_id:
                        connections_to_remove.append((source_id, target_id))
                
                for conn_key in connections_to_remove:
                    del self.neural_web.connections[conn_key]
            
            # Remove concept
            del self.neural_web.concepts[concept_id]
            self.logger.debug(f"Removed neural concept: {concept_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing concept {concept_id}: {e}")
            return False
    
    def get_concept_info(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a concept.
        
        Args:
            concept_id: ID of concept
            
        Returns:
            Concept information dictionary or None
        """
        try:
            if not hasattr(self.neural_web, 'concepts'):
                return None
                
            if concept_id not in self.neural_web.concepts:
                return None
            
            concept = self.neural_web.concepts[concept_id]
            
            return {
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
            
        except Exception as e:
            self.logger.error(f"Error getting concept info for {concept_id}: {e}")
            return None