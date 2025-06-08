"""
Neural concept connection management
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from ..memory_manager import MemorySegment
from ..neural_web_core import NeuralWeb
from .relationship_analyzer import RelationshipAnalyzer

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages connections between neural concepts"""
    
    def __init__(
        self,
        neural_web: NeuralWeb,
        auto_connect: bool = True,
        max_connections: int = 5,
        connection_threshold: float = 0.7
    ):
        """
        Initialize connection manager.
        
        Args:
            neural_web: Neural web instance
            auto_connect: Whether to auto-connect concepts
            max_connections: Maximum connections per concept
            connection_threshold: Minimum strength for connections
        """
        self.neural_web = neural_web
        self.auto_connect = auto_connect
        self.max_connections = max_connections
        self.connection_threshold = connection_threshold
        self.relationship_analyzer = RelationshipAnalyzer()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def auto_connect_concept(
        self, 
        new_concept_id: str, 
        segment: MemorySegment
    ) -> int:
        """
        Auto-connect a new concept to related existing concepts.
        
        Args:
            new_concept_id: ID of the new concept
            segment: Memory segment for the concept
            
        Returns:
            Number of connections made
        """
        if not self.auto_connect:
            return 0
        
        # Validation
        if not self._validate_concept_id(new_concept_id):
            self.logger.error(f"Invalid new_concept_id: '{new_concept_id}', cannot auto-connect")
            return 0
            
        if not segment:
            self.logger.error(f"Invalid segment provided for concept_id: {new_concept_id}")
            return 0
            
        try:
            # Check if neural web has concepts
            if not hasattr(self.neural_web, 'concepts'):
                self.logger.debug("Neural web has no concepts attribute")
                return 0
                
            if new_concept_id not in self.neural_web.concepts:
                self.logger.warning(f"New concept {new_concept_id} not found in neural web")
                return 0
            
            # Get content for new concept
            new_content = self._extract_content(segment)
            if not new_content:
                self.logger.debug(f"No content for concept {new_concept_id}, skipping auto-connect")
                return 0
            
            # Find and create connections
            connections_made = await self._create_connections(new_concept_id, new_content)
            
            self.logger.debug(f"Auto-connected concept {new_concept_id} to {connections_made} other concepts")
            return connections_made
            
        except Exception as e:
            self.logger.error(f"Failed to auto-connect concept {new_concept_id}: {e}")
            return 0
    
    async def _create_connections(self, new_concept_id: str, new_content: str) -> int:
        """
        Create connections between concepts based on similarity.
        
        Args:
            new_concept_id: ID of new concept
            new_content: Content of new concept
            
        Returns:
            Number of connections made
        """
        connections_made = 0
        
        for concept_id, concept_node in self.neural_web.concepts.items():
            if concept_id == new_concept_id:
                continue
                
            if connections_made >= self.max_connections:
                break
            
            # Get existing concept content
            existing_content = getattr(concept_node, 'content', '')
            if not existing_content:
                continue
            
            # Calculate similarity
            similarity = self.relationship_analyzer.calculate_content_similarity(
                new_content, 
                existing_content
            )
            
            if similarity >= self.connection_threshold:
                # Determine relationship type
                relationship_type = self.relationship_analyzer.determine_relationship_type(
                    new_content,
                    existing_content
                )
                
                # Create connection
                success = await self._create_single_connection(
                    new_concept_id,
                    concept_id,
                    relationship_type,
                    similarity
                )
                
                if success:
                    connections_made += 1
                    self.logger.debug(
                        f"Connected {new_concept_id} to {concept_id} "
                        f"(similarity: {similarity:.3f}, type: {relationship_type})"
                    )
        
        return connections_made
    
    async def _create_single_connection(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        strength: float
    ) -> bool:
        """
        Create a single connection between concepts.
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relationship_type: Type of relationship
            strength: Connection strength
            
        Returns:
            True if successful
        """
        try:
            if hasattr(self.neural_web, 'connect_concepts'):
                await self.neural_web.connect_concepts(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=relationship_type,
                    strength=strength
                )
                return True
            else:
                self.logger.warning("Neural web does not support connect_concepts method")
                return False
                
        except Exception as e:
            self.logger.debug(f"Error creating connection {source_id} -> {target_id}: {e}")
            return False
    
    def _validate_concept_id(self, concept_id: str) -> bool:
        """Validate that a concept ID is proper."""
        if not concept_id:
            return False
        if concept_id in ["null", "None", "", "undefined"]:
            return False
        if not isinstance(concept_id, str):
            return False
        return True
    
    def _extract_content(self, segment: MemorySegment) -> str:
        """Extract content text from segment."""
        if segment.content and segment.content.text:
            return segment.content.text
        elif segment.content:
            return str(segment.content)
        return ""
    
    def get_concept_connections(self, concept_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all connections for a concept.
        
        Args:
            concept_id: Concept ID to get connections for
            
        Returns:
            Dictionary with 'incoming' and 'outgoing' connection lists
        """
        try:
            if not hasattr(self.neural_web, 'connections'):
                return {"incoming": [], "outgoing": []}
            
            incoming = []
            outgoing = []
            
            for (source_id, target_id), connection in self.neural_web.connections.items():
                if target_id == concept_id:
                    incoming.append({
                        "source_id": source_id,
                        "relationship_type": connection.relationship_type,
                        "strength": connection.strength,
                        "confidence": connection.confidence,
                        "created_at": connection.created_at.isoformat()
                    })
                elif source_id == concept_id:
                    outgoing.append({
                        "target_id": target_id,
                        "relationship_type": connection.relationship_type,
                        "strength": connection.strength,
                        "confidence": connection.confidence,
                        "created_at": connection.created_at.isoformat()
                    })
            
            return {"incoming": incoming, "outgoing": outgoing}
            
        except Exception as e:
            self.logger.error(f"Error getting connections for {concept_id}: {e}")
            return {"incoming": [], "outgoing": []}
    
    async def reinforce_connection(
        self, 
        source_id: str, 
        target_id: str,
        reinforcement_strength: float = 0.1
    ) -> bool:
        """
        Reinforce an existing connection.
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID  
            reinforcement_strength: Amount to increase connection strength
            
        Returns:
            True if successful
        """
        try:
            if not hasattr(self.neural_web, 'connections'):
                return False
            
            conn_key = (source_id, target_id)
            if conn_key not in self.neural_web.connections:
                return False
            
            connection = self.neural_web.connections[conn_key]
            connection.strength = min(1.0, connection.strength + reinforcement_strength)
            connection.reinforcement_count += 1
            
            self.logger.debug(
                f"Reinforced connection {source_id} -> {target_id} "
                f"(new strength: {connection.strength:.3f})"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error reinforcing connection: {e}")
            return False