"""
Performance tracking module for the Adaptive LLM System.

This module handles tracking and logging of performance metrics.
"""

import logging
import asyncio
import time
import json
import os
from typing import Dict, List, Optional, Any


class PerformanceTracker:
    """Tracks performance metrics for the Adaptive LLM System."""
    
    def __init__(self, settings):
        """
        Initialize the PerformanceTracker.
        
        Args:
            settings: Adaptive LLM settings
        """
        self.settings = settings
        self.logger = logging.getLogger("WitsV3.PerformanceTracker")
        self.performance_log: List[Dict[str, Any]] = []
        
    def track_performance(
        self,
        prompt: str,
        response: str,
        module: str,
        complexity: float,
        generation_time: float,
        cache_hit: bool,
        similarity: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Track performance metrics.
        
        Args:
            prompt: The input prompt
            response: The generated response
            module: The module used
            complexity: The complexity score
            generation_time: The generation time in seconds
            cache_hit: Whether the response was from cache
            similarity: The similarity score (for cache hits)
            error: Error message (if any)
        """
        if not self.settings.enable_performance_tracking:
            return
            
        # Create performance entry
        entry = {
            'timestamp': time.time(),
            'prompt_length': len(prompt),
            'response_length': len(response),
            'module': module,
            'complexity': complexity,
            'generation_time': generation_time,
            'cache_hit': cache_hit,
        }
        
        if similarity is not None:
            entry['similarity'] = similarity
            
        if error is not None:
            entry['error'] = error
            
        # Add to log
        self.performance_log.append(entry)
        
        # Save periodically
        if len(self.performance_log) % 10 == 0:
            asyncio.create_task(self.save_performance_log())
    
    async def save_performance_log(self) -> None:
        """Save performance log to disk."""
        if not self.settings.enable_performance_tracking:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.settings.performance_log_path), exist_ok=True)
            
            # Save log
            with open(self.settings.performance_log_path, 'w') as f:
                json.dump(self.performance_log, f, indent=2)
                
            self.logger.debug(f"Performance log saved to {self.settings.performance_log_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving performance log: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.performance_log:
            return {
                'count': 0,
                'avg_generation_time': 0.0,
                'cache_hit_rate': 0.0,
                'module_usage': {},
                'complexity_distribution': {
                    'low': 0,
                    'medium': 0,
                    'high': 0,
                },
            }
            
        # Calculate statistics
        count = len(self.performance_log)
        avg_generation_time = sum(entry['generation_time'] for entry in self.performance_log) / count
        cache_hits = sum(1 for entry in self.performance_log if entry.get('cache_hit', False))
        cache_hit_rate = cache_hits / count
        
        # Module usage
        module_usage = {}
        for entry in self.performance_log:
            module = entry.get('module', 'unknown')
            if module not in module_usage:
                module_usage[module] = 0
            module_usage[module] += 1
            
        # Complexity distribution
        complexity_distribution = {
            'low': 0,
            'medium': 0,
            'high': 0,
        }
        
        for entry in self.performance_log:
            complexity = entry.get('complexity', 0.0)
            if complexity < 0.3:
                complexity_distribution['low'] += 1
            elif complexity < 0.7:
                complexity_distribution['medium'] += 1
            else:
                complexity_distribution['high'] += 1
                
        return {
            'count': count,
            'avg_generation_time': avg_generation_time,
            'cache_hit_rate': cache_hit_rate,
            'module_usage': module_usage,
            'complexity_distribution': complexity_distribution,
        }
    
    def get_recent_performance(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent performance entries.
        
        Args:
            n: Number of recent entries to return
            
        Returns:
            List of recent performance entries
        """
        return self.performance_log[-n:]