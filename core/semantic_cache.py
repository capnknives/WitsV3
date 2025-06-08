"""
SemanticCache for WitsV3 Adaptive LLM System.

This module implements a semantic cache that stores and retrieves patterns
based on semantic similarity for improved performance.
"""

import logging
import asyncio
import time
import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from .config import WitsV3Config
from .llm_interface import BaseLLMInterface
from .adaptive_llm_config import SemanticCacheSettings

class SemanticCache:
    """
    Stores and retrieves patterns based on semantic similarity.
    
    The SemanticCache stores patterns (input, output, metadata) and retrieves
    them based on semantic similarity for improved performance.
    """
    
    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        """
        Initialize the SemanticCache.
        
        Args:
            config: System configuration
            llm_interface: LLM interface for embeddings
        """
        self.config = config
        self.settings = SemanticCacheSettings()  # Use defaults
        self.llm_interface = llm_interface
        self.logger = logging.getLogger("WitsV3.SemanticCache")
        
        # Pattern storage
        self.embeddings = np.zeros((self.settings.cache_size, self.settings.embedding_dim), dtype=np.float16)
        self.metadata = []
        self.current_idx = 0
        self.is_full = False
        
        # User patterns
        self.user_patterns = {
            'common_domains': {},
            'complexity_history': [],
            'module_performance': {},
        }
        
        # Load cache if enabled
        if self.settings.enable_persistence:
            asyncio.create_task(self._load_cache())
        
        self.logger.info("SemanticCache initialized")
    
    async def add_pattern(
        self,
        input_text: str,
        response: str,
        computation_path: str,
        complexity: float,
        generation_time: float,
        output_quality: float = 1.0
    ) -> None:
        """
        Add a pattern to the cache.
        
        Args:
            input_text: The input text
            response: The generated response
            computation_path: The module used for generation
            complexity: The complexity score
            generation_time: The generation time in seconds
            output_quality: The quality score of the output (0.0 to 1.0)
        """
        try:
            # Get embedding for input
            input_embedding = await self.llm_interface.get_embedding(input_text)
            
            # Convert to numpy array
            input_embedding_np = np.array(input_embedding, dtype=np.float16)
            
            # Store embedding
            self.embeddings[self.current_idx] = input_embedding_np
            
            # Compute hash for deduplication
            input_hash = self._compute_hash(input_text)
            
            # Store metadata
            metadata = {
                'idx': self.current_idx,
                'timestamp': datetime.now().isoformat(),
                'input': input_text[:200] + ('...' if len(input_text) > 200 else ''),
                'response': response,
                'path': computation_path,
                'complexity': complexity,
                'generation_time': generation_time,
                'quality': output_quality,
                'hash': input_hash,
            }
            
            # Check if we need to replace an existing entry
            if self.is_full:
                self.metadata[self.current_idx] = metadata
            else:
                self.metadata.append(metadata)
            
            # Update index
            self.current_idx = (self.current_idx + 1) % self.settings.cache_size
            if self.current_idx == 0:
                self.is_full = True
                
            # Update user patterns
            if self.settings.track_user_patterns:
                self._update_user_patterns(metadata)
                
            # Save cache periodically
            if self.settings.enable_persistence and len(self.metadata) % self.settings.persistence_interval == 0:
                asyncio.create_task(self._save_cache())
                
            self.logger.debug(f"Pattern added to cache: {input_text[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Error adding pattern to cache: {e}")
    
    async def find_similar(
        self,
        input_text: str,
        threshold: Optional[float] = None,
        top_k: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Find similar patterns in the cache.
        
        Args:
            input_text: The input text
            threshold: The similarity threshold (0.0 to 1.0)
            top_k: The number of top matches to return
            
        Returns:
            Dictionary with similar pattern information or None if no match
        """
        if not self.metadata:
            return None
            
        try:
            # Get embedding for input
            input_embedding = await self.llm_interface.get_embedding(input_text)
            
            # Convert to numpy array
            input_embedding_np = np.array(input_embedding, dtype=np.float16)
            
            # Use effective threshold
            effective_threshold = threshold or self.settings.cache_similarity_threshold
            
            # Compute similarities
            num_patterns = len(self.metadata) if not self.is_full else self.settings.cache_size
            
            # Compute cosine similarity
            dot_products = np.dot(self.embeddings[:num_patterns], input_embedding_np)
            norms = np.linalg.norm(self.embeddings[:num_patterns], axis=1) * np.linalg.norm(input_embedding_np)
            similarities = dot_products / norms
            
            # Get top matches above threshold
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            for idx in top_indices:
                if similarities[idx] > effective_threshold:
                    # Check if pattern is expired
                    if self.settings.ttl_seconds > 0:
                        pattern_time = datetime.fromisoformat(self.metadata[idx]['timestamp'])
                        current_time = datetime.now()
                        age_seconds = (current_time - pattern_time).total_seconds()
                        
                        if age_seconds > self.settings.ttl_seconds:
                            self.logger.debug(f"Pattern expired: {age_seconds:.1f}s old")
                            continue
                    
                    result = {
                        'similarity': float(similarities[idx]),
                        'metadata': self.metadata[idx],
                        'response': self.metadata[idx]['response'],
                    }
                    
                    self.logger.debug(f"Found similar pattern with similarity: {similarities[idx]:.4f}")
                    return result
            
            self.logger.debug("No similar patterns found above threshold")
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {e}")
            return None
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash for a text string."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _update_user_patterns(self, metadata: Dict[str, Any]) -> None:
        """Update user patterns based on new metadata."""
        # Update domain statistics
        domain = metadata.get('domain', 'unknown')
        if domain in self.user_patterns['common_domains']:
            self.user_patterns['common_domains'][domain] += 1
        else:
            self.user_patterns['common_domains'][domain] = 1
            
        # Update complexity history
        self.user_patterns['complexity_history'].append(metadata.get('complexity', 0.0))
        if len(self.user_patterns['complexity_history']) > self.settings.max_user_patterns:
            self.user_patterns['complexity_history'] = self.user_patterns['complexity_history'][-self.settings.max_user_patterns:]
            
        # Update module performance
        module = metadata.get('path', 'unknown')
        if module not in self.user_patterns['module_performance']:
            self.user_patterns['module_performance'][module] = {
                'count': 0,
                'avg_time': 0.0,
                'avg_quality': 0.0,
            }
            
        perf = self.user_patterns['module_performance'][module]
        perf['count'] += 1
        perf['avg_time'] = (perf['avg_time'] * (perf['count'] - 1) + metadata.get('generation_time', 0.0)) / perf['count']
        perf['avg_quality'] = (perf['avg_quality'] * (perf['count'] - 1) + metadata.get('quality', 0.0)) / perf['count']
    
    async def _save_cache(self) -> None:
        """Save cache to disk."""
        if not self.settings.enable_persistence:
            return
            
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(self.settings.cache_file_path), exist_ok=True)
            
            # Prepare data for saving
            save_data = {
                'metadata': self.metadata,
                'current_idx': self.current_idx,
                'is_full': self.is_full,
                'user_patterns': self.user_patterns,
                'timestamp': datetime.now().isoformat(),
            }
            
            # Save metadata and user patterns
            with open(self.settings.cache_file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            # Save embeddings
            embeddings_path = self.settings.cache_file_path.replace('.json', '_embeddings.npy')
            np.save(embeddings_path, self.embeddings)
            
            self.logger.info(f"Cache saved to {self.settings.cache_file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")
    
    async def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.settings.enable_persistence:
            return
            
        try:
            # Check if cache file exists
            if not os.path.exists(self.settings.cache_file_path):
                self.logger.info("No cache file found, starting with empty cache")
                return
                
            # Load metadata and user patterns
            with open(self.settings.cache_file_path, 'r') as f:
                load_data = json.load(f)
                
            self.metadata = load_data.get('metadata', [])
            self.current_idx = load_data.get('current_idx', 0)
            self.is_full = load_data.get('is_full', False)
            self.user_patterns = load_data.get('user_patterns', {
                'common_domains': {},
                'complexity_history': [],
                'module_performance': {},
            })
            
            # Load embeddings
            embeddings_path = self.settings.cache_file_path.replace('.json', '_embeddings.npy')
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                
            self.logger.info(f"Cache loaded from {self.settings.cache_file_path} with {len(self.metadata)} patterns")
            
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            
            # Reset to empty cache
            self.embeddings = np.zeros((self.settings.cache_size, self.settings.embedding_dim), dtype=np.float16)
            self.metadata = []
            self.current_idx = 0
            self.is_full = False
    
    def get_user_patterns(self) -> Dict[str, Any]:
        """
        Get user patterns.
        
        Returns:
            Dictionary with user patterns
        """
        return self.user_patterns
    
    async def compress_cache(self) -> None:
        """Compress cache to save memory."""
        if not self.settings.enable_compression:
            return
            
        if len(self.metadata) < self.settings.compression_threshold:
            return
            
        try:
            self.logger.info("Compressing cache...")
            
            # Placeholder for actual compression logic
            # In a real implementation, this would use techniques like:
            # - Removing low-quality or old entries
            # - Clustering similar entries
            # - Dimensionality reduction on embeddings
            
            self.logger.info("Cache compression completed")
            
        except Exception as e:
            self.logger.error(f"Error compressing cache: {e}")
    
    async def clear_cache(self) -> None:
        """Clear the cache."""
        self.embeddings = np.zeros((self.settings.cache_size, self.settings.embedding_dim), dtype=np.float16)
        self.metadata = []
        self.current_idx = 0
        self.is_full = False
        
        # Reset user patterns
        self.user_patterns = {
            'common_domains': {},
            'complexity_history': [],
            'module_performance': {},
        }
        
        # Update cache file path to ensure it's set correctly
        self.settings.cache_file_path = os.path.join(self.settings.cache_dir, "semantic_cache.json")
        
        self.logger.info("Cache cleared")


# Test function
async def test_semantic_cache():
    """Test the SemanticCache functionality."""
    from .config import WitsV3Config
    from .llm_interface import get_llm_interface
    import os
    
    print("Testing SemanticCache...")
    
    # Load config
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    config_file_path = os.path.join(project_root_dir, "config.yaml")
    
    config = WitsV3Config.from_yaml(config_file_path)
    
    # Create LLM interface
    llm = get_llm_interface(config)
    
    # Create cache
    cache = SemanticCache(config, llm)
    
    # Test adding patterns
    print("\nAdding test patterns...")
    await cache.add_pattern(
        "How do I write a Python function to calculate Fibonacci numbers?",
        "Here's a recursive function to calculate Fibonacci numbers...",
        "python",
        0.7,
        0.5,
        0.9
    )
    
    await cache.add_pattern(
        "What's the best way to implement a Fibonacci sequence in Python?",
        "You can implement Fibonacci using dynamic programming...",
        "python",
        0.7,
        0.6,
        0.95
    )
    
    # Test finding similar patterns
    print("\nFinding similar patterns...")
    similar = await cache.find_similar("How to code Fibonacci in Python?")
    
    if similar:
        print(f"Found similar pattern with similarity: {similar['similarity']:.4f}")
        print(f"   Input: {similar['metadata']['input']}")
        print(f"   Path: {similar['metadata']['path']}")
        print(f"   Response: {similar['response'][:100]}...")
    else:
        print("No similar patterns found.")
    
    # Test user patterns
    print("\nUser patterns:")
    user_patterns = cache.get_user_patterns()
    print(f"   Module performance: {user_patterns['module_performance']}")
    
    # Test saving and loading
    print("\nSaving cache...")
    await cache._save_cache()
    
    print("\nClearing cache...")
    await cache.clear_cache()
    
    print("\nLoading cache...")
    await cache._load_cache()
    
    print("\nSemanticCache tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_semantic_cache())
