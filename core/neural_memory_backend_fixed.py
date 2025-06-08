# core/neural_memory_backend.py
"""
Neural Web-powered memory backend for WitsV3
Integrates graph-based knowledge representation with traditional memory management
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from .memory_manager import BaseMemoryBackend, MemorySegment
from .neural_web_core import NeuralWeb, ConceptNode
from .config import WitsV3Config
from .llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)


class NeuralMemoryBackend(BaseMemoryBackend):
    """
    Neural web-powered memory backend that combines traditional memory storage
    with graph-based knowledge representation and reasoning
    """
    
    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        super().__init__(config, llm_interface)
        
        # Initialize neural web
        neural_settings = config.memory_manager.neural_web_settings
        self.neural_web = NeuralWeb(
            activation_threshold=neural_settings.activation_threshold,
            decay_rate=neural_settings.decay_rate
        )
        
        # Configuration
        self.auto_connect = neural_settings.auto_connect
        self.reasoning_patterns = neural_settings.reasoning_patterns
        self.max_concept_connections = neural_settings.max_concept_connections
        self.connection_strength_threshold = neural_settings.connection_strength_threshold
        
        # Persistence
        self.neural_web_path = Path(config.memory_manager.neural_web_path)
        self.neural_web_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing neural web if available
        asyncio.create_task(self._load_neural_web())
        
        logger.info("Neural memory backend initialized")
    
    async def initialize(self):
        """Initialize the neural memory backend"""
        await super().initialize()
        logger.info("Neural memory backend initialized")
    
    async def add_segment(self, segment: MemorySegment) -> str:
        """Add a memory segment and create corresponding neural web concepts"""
        # Add to traditional storage first
        segment_id = await super().add_segment(segment)
        
        # Create neural web concept
        await self._create_neural_concept(segment_id, segment)
        
        # Auto-connect to related concepts if enabled
        if self.auto_connect:
            await self._auto_connect_concepts(segment_id, segment)
            
        # Persist neural web
        await self._save_neural_web()
        
        logger.debug(f"Added neural memory segment: {segment_id}")
        return segment_id
    
    async def get_segment(self, segment_id: str) -> Optional[MemorySegment]:
        """Get a specific memory segment by ID"""
        return await super().get_segment(segment_id)
    
    async def search_segments(self, query: str, max_results: int = 5, 
                            min_relevance: float = 0.0,
                            filter_dict: Optional[Dict[str, Any]] = None) -> List[MemorySegment]:
        """
        Enhanced search using neural web activation propagation
        """
        # Start with traditional search
        traditional_results = await super().search_segments(query, max_results, min_relevance, filter_dict)
        
        # Enhance with neural web reasoning
        neural_results = await self._neural_enhanced_search(query, max_results)
        
        # Combine and deduplicate results
        combined_results = self._combine_search_results(traditional_results, neural_results)
        
        # Activate relevant concepts for future queries
        await self._activate_search_concepts(query)
        
        return combined_results[:max_results]
    
    async def get_recent_segments(self, limit: int = 10, 
                                 filter_dict: Optional[Dict[str, Any]] = None) -> List[MemorySegment]:
        """Get recent memory segments with neural web enhancement"""
        # Use the base implementation for basic functionality
        recent_segments = await super().get_recent_segments(limit, filter_dict)
        
        # Enhance with neural activation for recently accessed concepts
        for segment in recent_segments:
            if segment.id in self.neural_web.concepts:
                # Activate recently accessed concepts to maintain their relevance
                await self.neural_web.activate_concept(segment.id, 0.3)
        
        return recent_segments

    async def _create_neural_concept(self, segment_id: str, segment: MemorySegment):
        """Create a neural web concept from a memory segment"""
        # Validate segment_id is not None
        if not segment_id or segment_id == "null":
            logger.error(f"Invalid segment_id: {segment_id}, cannot create neural concept")
            return
            
        concept_type = self._determine_concept_type(segment)
        await self.neural_web.add_concept(
            concept_id=segment_id,
            content=segment.content.text or "",
            concept_type=concept_type,
            metadata={
                "timestamp": segment.timestamp.isoformat(),
                "source": segment.source,
                "memory_type": segment.type,
                "original_metadata": segment.metadata
            }
        )
    
    def _determine_concept_type(self, segment: MemorySegment) -> str:
        """Determine the neural web concept type from memory segment"""
        content_lower = (segment.content.text or "").lower()
        
        # Pattern matching for concept types
        if any(word in content_lower for word in ["how to", "step", "procedure", "process"]):
            return "procedure"
        elif any(word in content_lower for word in ["goal", "objective", "want to", "need to"]):
            return "goal"
        elif any(word in content_lower for word in ["because", "causes", "results in", "leads to"]):
            return "pattern"
        elif segment.type in ["conversation", "user_input", "agent_response"]:
            return "memory"
        else:
            return "fact"
    
    async def _auto_connect_concepts(self, new_concept_id: str, segment: MemorySegment):
        """Automatically connect new concepts to related existing concepts"""
        # Validate new_concept_id is not None
        if not new_concept_id or new_concept_id == "null":
            logger.error(f"Invalid new_concept_id: {new_concept_id}, cannot auto-connect")
            return
            
        content = segment.content.text or ""
        content_words = set(content.lower().split())
        
        # Find related concepts based on content similarity
        related_concepts = []
        for concept_id, concept in self.neural_web.concepts.items():
            if concept_id == new_concept_id:
                continue
                
            concept_words = set(concept.content.lower().split())
            overlap = len(content_words.intersection(concept_words))
            
            if overlap >= 3:  # Minimum word overlap threshold
                similarity = overlap / len(content_words.union(concept_words))
                if similarity > 0.2:
                    related_concepts.append((concept_id, similarity))
        
        # Create connections to most similar concepts
        related_concepts.sort(key=lambda x: x[1], reverse=True)
        max_connections = min(self.max_concept_connections, len(related_concepts))
        
        for concept_id, similarity in related_concepts[:max_connections]:
            try:
                relationship_type = self._determine_relationship_type(
                    segment.content.text or "", 
                    self.neural_web.concepts[concept_id].content
                )
                
                await self.neural_web.connect_concepts(
                    source_id=new_concept_id,
                    target_id=concept_id,
                    relationship_type=relationship_type,
                    strength=similarity,
                    confidence=min(0.8, similarity * 1.5)
                )
            except Exception as e:
                logger.error(f"Error connecting concepts {new_concept_id} -> {concept_id}: {e}")
    
    def _determine_relationship_type(self, content1: str, content2: str) -> str:
        """Determine the relationship type between two pieces of content"""
        content1_lower = content1.lower()
        content2_lower = content2.lower()
        
        # Causal relationships
        if any(word in content1_lower for word in ["causes", "results in", "leads to"]):
            return "causes"
        elif any(word in content1_lower for word in ["enables", "allows", "helps"]):
            return "enables"
        elif any(word in content1_lower for word in ["contradicts", "opposes", "conflicts"]):
            return "contradicts"
        elif any(word in content1_lower for word in ["similar", "like", "resembles"]):
            return "similar"
        elif any(word in content1_lower for word in ["part of", "contains", "includes"]):
            return "part_of"
        else:
            return "related"
    
    async def _neural_enhanced_search(self, query: str, max_results: int) -> List[MemorySegment]:
        """Use neural web reasoning to enhance search results"""
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
                        # Extract concept IDs from reasoning paths
                        for concept_content in result["path"]:
                            for concept_id, concept in self.neural_web.concepts.items():
                                if concept.content == concept_content:
                                    additional_concepts.add(concept_id)
            
            # Combine all relevant concepts
            all_concepts = set(relevant_concepts) | additional_concepts
            
            # Convert to memory segments - search through our segment list
            neural_segments = []
            for concept_id in all_concepts:
                # Find the memory segment with this concept ID
                for segment in self.segments:
                    if segment.id == concept_id:
                        neural_segments.append(segment)
                        break
            
            return neural_segments
            
        except Exception as e:
            logger.error(f"Error in neural enhanced search: {e}")
            return []
    
    async def _activate_search_concepts(self, query: str):
        """Activate concepts related to the search query for future reasoning"""
        try:
            relevant_concepts = await self.neural_web._find_relevant_concepts(query)
            
            if relevant_concepts:  # Check if we got a valid list
                for concept_id in relevant_concepts:
                    if concept_id and concept_id in self.neural_web.concepts:  # Check concept_id is valid
                        await self.neural_web.activate_concept(concept_id, 0.6)
        except Exception as e:
            logger.error(f"Error activating search concepts: {e}")
    
    def _combine_search_results(self, traditional: List[MemorySegment], 
                               neural: List[MemorySegment]) -> List[MemorySegment]:
        """Combine and rank search results from traditional and neural methods"""
        # Create a combined list with deduplication
        seen_ids = set()
        combined = []
        
        # Prioritize traditional results (they match the query directly)
        for segment in traditional:
            if segment.id not in seen_ids:
                combined.append(segment)
                seen_ids.add(segment.id)
        
        # Add neural results that weren't already included
        for segment in neural:
            if segment.id not in seen_ids:
                combined.append(segment)
                seen_ids.add(segment.id)
        
        return combined
    
    async def get_neural_insights(self, query: str) -> Dict[str, Any]:
        """Get insights from the neural web for a given query"""
        insights = {}
        
        # Get reasoning results for different patterns
        for pattern in self.reasoning_patterns:
            try:
                result = await self.neural_web.reason(query, pattern)
                if result.get("results"):
                    insights[pattern] = result
            except Exception as e:
                logger.warning(f"Error in {pattern} reasoning: {e}")
        
        # Get network statistics
        insights["network_stats"] = self.neural_web.get_statistics()
        
        # Get activation state
        active_concepts = [
            {
                "id": concept_id,
                "content": concept.content[:100] + "..." if len(concept.content) > 100 else concept.content,
                "activation": concept.activation_level,
                "type": concept.concept_type
            }
            for concept_id, concept in self.neural_web.concepts.items()
            if concept.activation_level > self.neural_web.activation_threshold
        ]
        
        insights["active_concepts"] = sorted(
            active_concepts, 
            key=lambda x: x["activation"], 
            reverse=True
        )[:10]
        
        return insights
    
    async def _save_neural_web(self):
        """Persist the neural web to disk"""
        try:
            # Create serializable representation
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
                    for conn in self.neural_web.connections.values()
                },
                "metadata": {
                    "activation_threshold": self.neural_web.activation_threshold,
                    "decay_rate": self.neural_web.decay_rate,
                    "saved_at": datetime.now().isoformat()
                }
            }
            
            with open(self.neural_web_path, 'w') as f:
                json.dump(neural_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving neural web: {e}")
    
    async def _load_neural_web(self):
        """Load the neural web from disk"""
        try:
            if not self.neural_web_path.exists():
                logger.info("No existing neural web found, starting fresh")
                return
            
            with open(self.neural_web_path, 'r') as f:
                neural_data = json.load(f)
            
            # Restore concepts
            for concept_id, concept_data in neural_data.get("concepts", {}).items():
                # Skip concepts with null/None IDs or generate a new ID
                stored_concept_id = concept_data.get("id")
                if stored_concept_id is None or stored_concept_id == "null" or stored_concept_id == "":
                    import uuid
                    # Generate a new valid concept ID
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
                
                # Skip connections with null/None IDs
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
            
        except Exception as e:
            logger.error(f"Error loading neural web: {e}")
            logger.info("Starting with fresh neural web")
    
    async def maintain_neural_web(self):
        """Perform maintenance on the neural web"""
        # Apply natural decay
        await self.neural_web.decay_activation()
        
        # Prune weak connections
        await self.neural_web.prune_weak_connections(self.connection_strength_threshold)
        
        # Save state
        await self._save_neural_web()
        
        logger.debug("Neural web maintenance completed")
    
    async def prune_memory(self):
        """Prune old memory and maintain neural web"""
        await super().prune_memory()
        await self.maintain_neural_web()
