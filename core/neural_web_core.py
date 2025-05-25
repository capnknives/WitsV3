# core/neural_web.py
"""
Neural Web implementation for WitsV3
Creates a graph-based knowledge network with emergent behavior
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConceptNode:
    """Represents a concept in the neural web"""
    id: str
    content: str
    concept_type: str  # 'fact', 'procedure', 'pattern', 'goal', 'memory'
    activation_level: float = 0.0
    base_strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    def activate(self, strength: float = 1.0):
        """Activate this concept node"""
        self.activation_level = min(1.0, self.activation_level + strength)
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def decay(self, rate: float = 0.1):
        """Natural decay of activation"""
        self.activation_level = max(0.0, self.activation_level - rate)


@dataclass
class Connection:
    """Represents a connection between concepts"""
    source_id: str
    target_id: str
    relationship_type: str  # 'causes', 'enables', 'contradicts', 'similar', 'part_of'
    strength: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    reinforcement_count: int = 0
    
    def reinforce(self, amount: float = 0.1):
        """Strengthen the connection through use"""
        self.strength = min(2.0, self.strength + amount)
        self.reinforcement_count += 1
    
    def weaken(self, amount: float = 0.05):
        """Weaken unused connections"""
        self.strength = max(0.1, self.strength - amount)


class NeuralWeb:
    """
    Graph-based neural network for knowledge representation and reasoning
    """
    
    def __init__(self, activation_threshold: float = 0.3, decay_rate: float = 0.1):
        self.graph = nx.DiGraph()
        self.concepts: Dict[str, ConceptNode] = {}
        self.connections: Dict[Tuple[str, str], Connection] = {}
        self.activation_threshold = activation_threshold
        self.decay_rate = decay_rate
        
        # Reasoning patterns
        self.inference_patterns = {
            'modus_ponens': self._modus_ponens,
            'analogy': self._analogical_reasoning,
            'chain': self._chain_reasoning,
            'contradiction': self._contradiction_detection
        }
        
        logger.info("Neural web initialized")
    
    async def add_concept(self, concept_id: str, content: str, concept_type: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> ConceptNode:
        """Add a new concept to the neural web"""
        # Validate concept_id is not None or empty
        if concept_id is None or concept_id == "" or concept_id == "null":
            import uuid
            concept_id = f"concept_{uuid.uuid4().hex[:8]}"
            logger.warning(f"Invalid concept ID (None/empty/null) replaced with generated ID: {concept_id}")
        
        if concept_id in self.concepts:
            # Update existing concept
            node = self.concepts[concept_id]
            node.content = content
            node.metadata.update(metadata or {})
        else:
            # Create new concept
            node = ConceptNode(
                id=concept_id,
                content=content,
                concept_type=concept_type,
                metadata=metadata or {}
            )
            self.concepts[concept_id] = node
            self.graph.add_node(concept_id, node=node)
        
        logger.debug(f"Added concept: {concept_id}")
        return node
    
    async def connect_concepts(self, source_id: str, target_id: str, 
                              relationship_type: str, strength: float = 1.0,
                              confidence: float = 1.0) -> Connection:
        """Create a connection between two concepts"""
        # Validate that neither ID is null/None
        if (source_id is None or source_id == "null" or source_id == "" or
            target_id is None or target_id == "null" or target_id == ""):
            raise ValueError(f"Invalid concept IDs for connection: {source_id} -> {target_id}")
        
        if source_id not in self.concepts or target_id not in self.concepts:
            raise ValueError("Both concepts must exist before connecting")
        
        connection_key = (source_id, target_id)
        
        if connection_key in self.connections:
            # Reinforce existing connection
            connection = self.connections[connection_key]
            connection.reinforce()
        else:
            # Create new connection
            connection = Connection(
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type,
                strength=strength,
                confidence=confidence
            )
            self.connections[connection_key] = connection
            self.graph.add_edge(source_id, target_id, connection=connection)
        
        logger.debug(f"Connected {source_id} -> {target_id} ({relationship_type})")
        return connection
    
    async def activate_concept(self, concept_id: str, activation_strength: float = 1.0) -> List[str]:
        """Activate a concept and propagate activation through the network"""
        if concept_id not in self.concepts:
            return []
        
        # Activate the initial concept
        self.concepts[concept_id].activate(activation_strength)
        activated_concepts = [concept_id]
        
        # Propagate activation
        propagation_queue = [(concept_id, activation_strength)]
        visited = {concept_id}
        
        while propagation_queue:
            current_id, current_strength = propagation_queue.pop(0)
            
            # Find connected concepts
            for neighbor_id in self.graph.successors(current_id):
                if neighbor_id in visited:
                    continue
                
                connection = self.connections.get((current_id, neighbor_id))
                if not connection:
                    continue
                
                # Calculate propagated activation
                propagated_strength = current_strength * connection.strength * 0.8
                
                if propagated_strength > self.activation_threshold:
                    self.concepts[neighbor_id].activate(propagated_strength)
                    activated_concepts.append(neighbor_id)
                    propagation_queue.append((neighbor_id, propagated_strength))
                    visited.add(neighbor_id)
        
        logger.debug(f"Activation propagated to {len(activated_concepts)} concepts")
        return activated_concepts
    
    async def find_path(self, start_id: str, end_id: str, max_length: int = 5) -> List[List[str]]:
        """Find reasoning paths between two concepts"""
        if start_id not in self.concepts or end_id not in self.concepts:
            return []
        
        try:
            # Find all simple paths up to max_length
            paths = list(nx.all_simple_paths(
                self.graph, start_id, end_id, cutoff=max_length
            ))
            
            # Sort by path quality (considering connection strengths)
            scored_paths = []
            for path in paths:
                score = self._calculate_path_score(path)
                scored_paths.append((score, path))
            
            scored_paths.sort(reverse=True)
            return [path for score, path in scored_paths[:10]]  # Top 10 paths
            
        except nx.NetworkXNoPath:
            return []
    
    def _calculate_path_score(self, path: List[str]) -> float:
        """Calculate the quality score of a reasoning path"""
        if len(path) < 2:
            return 0.0
        
        total_score = 1.0
        for i in range(len(path) - 1):
            connection = self.connections.get((path[i], path[i + 1]))
            if connection:
                total_score *= connection.strength * connection.confidence
            else:
                total_score *= 0.1  # Penalty for missing connection
        
        # Penalize longer paths
        length_penalty = 0.9 ** (len(path) - 2)
        return total_score * length_penalty
    
    async def reason(self, query: str, reasoning_type: str = 'chain') -> Dict[str, Any]:
        """Perform reasoning using the neural web"""
        # Activate concepts related to the query
        relevant_concepts = await self._find_relevant_concepts(query)
        
        if not relevant_concepts:
            return {"error": "No relevant concepts found"}
        
        # Apply reasoning pattern
        if reasoning_type in self.inference_patterns:
            result = await self.inference_patterns[reasoning_type](relevant_concepts, query)
        else:
            result = await self._chain_reasoning(relevant_concepts, query)
        
        return result
    
    async def _find_relevant_concepts(self, query: str) -> List[str]:
        """Find concepts relevant to a query using activation propagation"""
        # Always return a list, never None
        if not query or not query.strip():
            return []
            
        # Simple text matching for now - could be enhanced with embeddings
        relevant = []
        query_lower = query.lower()
        
        for concept_id, concept in self.concepts.items():
            if query_lower in concept.content.lower():
                relevant.append(concept_id)
        
        if not relevant:
            return []
        
        # Activate and propagate
        all_activated = []
        for concept_id in relevant:
            try:
                activated = await self.activate_concept(concept_id, 0.8)
                if activated:  # Ensure activated is not None
                    all_activated.extend(activated)
            except Exception as e:
                logger.error(f"Error activating concept {concept_id}: {e}")
                continue
        
        # Return concepts above activation threshold
        result = [cid for cid in set(all_activated) 
                  if cid in self.concepts and self.concepts[cid].activation_level > self.activation_threshold]
        return result if result else []
    
    async def _modus_ponens(self, concepts: List[str], query: str) -> Dict[str, Any]:
        """Apply modus ponens reasoning (If A then B, A is true, therefore B)"""
        results = []
        
        for concept_id in concepts:
            # Find 'causes' or 'enables' relationships
            for neighbor_id in self.graph.successors(concept_id):
                connection = self.connections.get((concept_id, neighbor_id))
                if connection and connection.relationship_type in ['causes', 'enables']:
                    if self.concepts[concept_id].activation_level > self.activation_threshold:
                        results.append({
                            'premise': self.concepts[concept_id].content,
                            'conclusion': self.concepts[neighbor_id].content,
                            'confidence': connection.confidence,
                            'reasoning': f"Since {self.concepts[concept_id].content}, therefore {self.concepts[neighbor_id].content}"
                        })
        
        return {'type': 'modus_ponens', 'results': results}
    
    async def _analogical_reasoning(self, concepts: List[str], query: str) -> Dict[str, Any]:
        """Find analogies between concepts"""
        analogies = []
        
        # Find concepts with similar relationship patterns
        for concept_id in concepts:
            neighbors = list(self.graph.successors(concept_id))
            
            # Find other concepts with similar neighbor patterns
            for other_id, other_concept in self.concepts.items():
                if other_id == concept_id or other_id not in concepts:
                    continue
                
                other_neighbors = list(self.graph.successors(other_id))
                
                # Calculate structural similarity
                similarity = self._calculate_structural_similarity(neighbors, other_neighbors)
                
                if similarity > 0.3:
                    analogies.append({
                        'source': self.concepts[concept_id].content,
                        'target': self.concepts[other_id].content,
                        'similarity': similarity,
                        'reasoning': f"{self.concepts[concept_id].content} is analogous to {self.concepts[other_id].content}"
                    })
        
        return {'type': 'analogy', 'results': analogies}
    
    async def _chain_reasoning(self, concepts: List[str], query: str) -> Dict[str, Any]:
        """Chain reasoning through multiple steps"""
        chains = []
        
        # Find reasoning chains starting from highly activated concepts
        start_concepts = [cid for cid in concepts 
                         if self.concepts[cid].activation_level > 0.7]
        
        for start_id in start_concepts:
            for end_id in concepts:
                if start_id != end_id:
                    paths = await self.find_path(start_id, end_id, max_length=4)
                    
                    for path in paths[:3]:  # Top 3 paths
                        chain_reasoning = self._build_chain_explanation(path)
                        if chain_reasoning:
                            chains.append({
                                'path': [self.concepts[pid].content for pid in path],
                                'reasoning': chain_reasoning,
                                'confidence': self._calculate_path_score(path)
                            })
        
        chains.sort(key=lambda x: x['confidence'], reverse=True)
        return {'type': 'chain', 'results': chains[:5]}
    
    async def _contradiction_detection(self, concepts: List[str], query: str) -> Dict[str, Any]:
        """Detect contradictions in the knowledge network"""
        contradictions = []
        
        for concept_id in concepts:
            for neighbor_id in self.graph.successors(concept_id):
                connection = self.connections.get((concept_id, neighbor_id))
                if connection and connection.relationship_type == 'contradicts':
                    contradictions.append({
                        'concept1': self.concepts[concept_id].content,
                        'concept2': self.concepts[neighbor_id].content,
                        'reasoning': f"Contradiction detected: {self.concepts[concept_id].content} contradicts {self.concepts[neighbor_id].content}"
                    })
        
        return {'type': 'contradiction', 'results': contradictions}
    
    def _calculate_structural_similarity(self, neighbors1: List[str], neighbors2: List[str]) -> float:
        """Calculate structural similarity between concept neighborhoods"""
        if not neighbors1 or not neighbors2:
            return 0.0
        
        # Simple Jaccard similarity for now
        set1 = set(neighbors1)
        set2 = set(neighbors2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _build_chain_explanation(self, path: List[str]) -> str:
        """Build a natural language explanation of a reasoning chain"""
        if len(path) < 2:
            return ""
        
        explanation_parts = []
        
        for i in range(len(path) - 1):
            current = self.concepts[path[i]].content
            next_concept = self.concepts[path[i + 1]].content
            
            connection = self.connections.get((path[i], path[i + 1]))
            if connection:
                if connection.relationship_type == 'causes':
                    explanation_parts.append(f"{current} causes {next_concept}")
                elif connection.relationship_type == 'enables':
                    explanation_parts.append(f"{current} enables {next_concept}")
                else:
                    explanation_parts.append(f"{current} relates to {next_concept}")
        
        return " → ".join(explanation_parts)
    
    async def decay_activation(self):
        """Apply natural decay to all concept activations"""
        for concept in self.concepts.values():
            concept.decay(self.decay_rate)
    
    async def prune_weak_connections(self, threshold: float = 0.2):
        """Remove weak connections to keep the network manageable"""
        to_remove = []
        
        for key, connection in self.connections.items():
            if connection.strength < threshold:
                to_remove.append(key)
        
        for key in to_remove:
            del self.connections[key]
            self.graph.remove_edge(key[0], key[1])
        
        logger.info(f"Pruned {len(to_remove)} weak connections")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get neural web statistics"""
        return {
            'concepts': len(self.concepts),
            'connections': len(self.connections),
            'active_concepts': len([c for c in self.concepts.values() 
                                  if c.activation_level > self.activation_threshold]),
            'graph_density': nx.density(self.graph),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'average_path_length': nx.average_shortest_path_length(self.graph) 
                                 if nx.is_weakly_connected(self.graph) else None
        }


# Test function
async def test_neural_web():
    """Test the neural web functionality"""
    web = NeuralWeb()
    
    # Add some test concepts
    await web.add_concept("rain", "It is raining", "fact")
    await web.add_concept("wet_ground", "The ground is wet", "fact")
    await web.add_concept("slippery", "Wet ground is slippery", "fact")
    await web.add_concept("careful_walking", "Walk carefully on slippery ground", "procedure")
    
    # Create connections
    await web.connect_concepts("rain", "wet_ground", "causes", 0.9)
    await web.connect_concepts("wet_ground", "slippery", "causes", 0.8)
    await web.connect_concepts("slippery", "careful_walking", "enables", 0.7)
    
    # Test activation propagation
    activated = await web.activate_concept("rain", 1.0)
    print(f"Activated concepts: {activated}")
    
    # Test reasoning
    result = await web.reason("rain and walking", "chain")
    print(f"Reasoning result: {result}")
    
    # Test path finding
    paths = await web.find_path("rain", "careful_walking")
    print(f"Reasoning paths: {paths}")
    
    # Get statistics
    stats = web.get_statistics()
    print(f"Neural web stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_neural_web())