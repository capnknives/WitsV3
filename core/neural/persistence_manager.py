"""
Neural web persistence management
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ..neural_web_core import NeuralWeb

logger = logging.getLogger(__name__)


class PersistenceManager:
    """Manages saving and loading neural web state to/from disk"""
    
    def __init__(self, neural_web: NeuralWeb, storage_path: Path):
        """
        Initialize persistence manager.
        
        Args:
            neural_web: Neural web instance to persist
            storage_path: Path to storage file
        """
        self.neural_web = neural_web
        self.storage_path = storage_path
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def save_neural_web(self) -> bool:
        """
        Save the neural web to disk using custom serialization.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            neural_data = self._serialize_neural_web()
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(neural_data, f, indent=2, ensure_ascii=False)
                
            self.logger.debug(f"Neural web saved to {self.storage_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save neural web: {e}")
            return False
    
    async def load_neural_web(self) -> bool:
        """
        Load the neural web from disk using custom deserialization.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.storage_path.exists():
                self.logger.debug("No existing neural web found, starting fresh")
                return True
            
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                neural_data = json.load(f)
            
            await self._deserialize_neural_web(neural_data)
            
            # Log statistics
            concept_count = len(self.neural_web.concepts) if hasattr(self.neural_web, 'concepts') else 0
            connection_count = len(self.neural_web.connections) if hasattr(self.neural_web, 'connections') else 0
            
            self.logger.info(
                f"Loaded neural web with {concept_count} concepts "
                f"and {connection_count} connections"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load neural web: {e}")
            self.logger.info("Continuing with fresh neural web")
            return False
    
    def _serialize_neural_web(self) -> Dict[str, Any]:
        """
        Serialize neural web to JSON-compatible format.
        
        Returns:
            Serialized neural web data
        """
        data = {
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "concepts": {},
            "connections": {}
        }
        
        # Serialize concepts
        if hasattr(self.neural_web, 'concepts'):
            for concept_id, concept in self.neural_web.concepts.items():
                data["concepts"][concept_id] = self._serialize_concept(concept)
        
        # Serialize connections
        if hasattr(self.neural_web, 'connections'):
            for (source_id, target_id), connection in self.neural_web.connections.items():
                conn_key = f"{source_id}->{target_id}"
                data["connections"][conn_key] = self._serialize_connection(connection)
        
        return data
    
    def _serialize_concept(self, concept) -> Dict[str, Any]:
        """Serialize a single concept."""
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
    
    def _serialize_connection(self, connection) -> Dict[str, Any]:
        """Serialize a single connection."""
        return {
            "source_id": connection.source_id,
            "target_id": connection.target_id,
            "relationship_type": connection.relationship_type,
            "strength": connection.strength,
            "confidence": connection.confidence,
            "created_at": connection.created_at.isoformat(),
            "reinforcement_count": connection.reinforcement_count
        }
    
    async def _deserialize_neural_web(self, data: Dict[str, Any]) -> None:
        """
        Deserialize neural web from loaded data.
        
        Args:
            data: Loaded neural web data
        """
        # Restore concepts first
        for concept_id, concept_data in data.get("concepts", {}).items():
            await self._restore_concept(concept_id, concept_data)
        
        # Restore connections
        for conn_key, conn_data in data.get("connections", {}).items():
            await self._restore_connection(conn_data)
    
    async def _restore_concept(self, concept_id: str, concept_data: Dict[str, Any]) -> None:
        """Restore a single concept."""
        try:
            # Validate concept ID
            stored_id = concept_data.get("id")
            if not self._is_valid_id(stored_id):
                valid_id = f"concept_{uuid.uuid4().hex[:8]}"
                self.logger.warning(
                    f"Invalid concept ID '{stored_id}', using new ID: {valid_id}"
                )
            else:
                valid_id = stored_id
            
            # Add concept to neural web
            if hasattr(self.neural_web, 'add_concept'):
                await self.neural_web.add_concept(
                    concept_id=valid_id,
                    content=concept_data.get("content", ""),
                    concept_type=concept_data.get("concept_type", "unknown"),
                    metadata=concept_data.get("metadata", {})
                )
                
                # Restore additional properties
                if valid_id in self.neural_web.concepts:
                    concept = self.neural_web.concepts[valid_id]
                    concept.activation_level = concept_data.get("activation_level", 0.0)
                    concept.base_strength = concept_data.get("base_strength", 1.0)
                    concept.created_at = datetime.fromisoformat(concept_data["created_at"])
                    concept.last_accessed = datetime.fromisoformat(concept_data["last_accessed"])
                    concept.access_count = concept_data.get("access_count", 0)
                    
        except Exception as e:
            self.logger.error(f"Error restoring concept {concept_id}: {e}")
    
    async def _restore_connection(self, conn_data: Dict[str, Any]) -> None:
        """Restore a single connection."""
        try:
            source_id = conn_data.get("source_id")
            target_id = conn_data.get("target_id")
            
            # Validate IDs
            if not self._is_valid_id(source_id) or not self._is_valid_id(target_id):
                self.logger.warning(
                    f"Skipping connection with invalid IDs: {source_id} -> {target_id}"
                )
                return
            
            # Check if both concepts exist
            if not hasattr(self.neural_web, 'concepts'):
                return
                
            if source_id not in self.neural_web.concepts or target_id not in self.neural_web.concepts:
                self.logger.warning(
                    f"Skipping connection {source_id} -> {target_id}: "
                    f"one or both concepts don't exist"
                )
                return
            
            # Create connection
            if hasattr(self.neural_web, 'connect_concepts'):
                await self.neural_web.connect_concepts(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=conn_data.get("relationship_type", "related"),
                    strength=conn_data.get("strength", 0.5),
                    confidence=conn_data.get("confidence", 1.0)
                )
                
                # Restore additional properties
                conn_key = (source_id, target_id)
                if hasattr(self.neural_web, 'connections') and conn_key in self.neural_web.connections:
                    connection = self.neural_web.connections[conn_key]
                    connection.created_at = datetime.fromisoformat(conn_data["created_at"])
                    connection.reinforcement_count = conn_data.get("reinforcement_count", 0)
                    
        except Exception as e:
            self.logger.error(f"Error restoring connection: {e}")
    
    def _is_valid_id(self, id_value: Any) -> bool:
        """Check if an ID is valid."""
        if not id_value:
            return False
        if not isinstance(id_value, str):
            return False
        if id_value in ["null", "None", "", "undefined"]:
            return False
        return True
    
    async def backup_neural_web(self, backup_suffix: str = "backup") -> bool:
        """
        Create a backup of the current neural web.
        
        Args:
            backup_suffix: Suffix to add to backup filename
            
        Returns:
            True if successful
        """
        try:
            if not self.storage_path.exists():
                return True  # Nothing to backup
            
            # Create backup filename
            backup_path = self.storage_path.with_suffix(f".{backup_suffix}.json")
            
            # Copy current file to backup
            import shutil
            shutil.copy2(self.storage_path, backup_path)
            
            self.logger.info(f"Created neural web backup at {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False