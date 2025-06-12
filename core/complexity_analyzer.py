"""
ComplexityAnalyzer for WitsV3 Adaptive LLM System.

This module implements a complexity analyzer that analyzes query complexity
and domain to route to appropriate specialized modules.
"""

import logging
import asyncio
import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from .config import WitsV3Config
from .llm_interface import BaseLLMInterface
from .adaptive_llm_config import ComplexityAnalyzerSettings

class ComplexityAnalyzer:
    """
    Analyzes query complexity and domain to route to appropriate modules.
    
    The ComplexityAnalyzer analyzes the complexity and domain of queries
    to route them to the most appropriate specialized module.
    """
    
    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        """
        Initialize the ComplexityAnalyzer.
        
        Args:
            config: System configuration
            llm_interface: LLM interface for embeddings
        """
        self.config = config
        self.settings = ComplexityAnalyzerSettings()  # Use defaults
        self.llm_interface = llm_interface
        self.logger = logging.getLogger("WitsV3.ComplexityAnalyzer")
        
        # Initialize model if available
        self.model = None
        self.tokenizer = None
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        self.logger.info("ComplexityAnalyzer initialized")
    
    async def analyze_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analyze the complexity of the given text.
        
        Args:
            text: The input text
            
        Returns:
            Dictionary with complexity analysis results
        """
        start_time = time.time()
        
        # Get embedding if enabled
        embedding = None
        if self.settings.use_embeddings:
            embedding = await self.llm_interface.get_embedding(text)
        
        # Calculate complexity factors
        factors = await self._calculate_complexity_factors(text, embedding)
        
        # Calculate overall complexity
        complexity = self._calculate_overall_complexity(factors)
        
        # Determine domain
        domain = await self._determine_domain(text, embedding)
        
        self.logger.debug(f"Complexity analysis completed in {time.time() - start_time:.2f}s")
        
        return {
            'complexity': complexity,
            'domain': domain,
            'factors': factors,
            'embedding': embedding,
        }
    
    async def _calculate_complexity_factors(self, text: str, embedding: Optional[List[float]]) -> Dict[str, float]:
        """
        Calculate complexity factors for the given text.
        
        Args:
            text: The input text
            embedding: Text embedding (optional)
            
        Returns:
            Dictionary with complexity factors
        """
        # Length factor (normalized by max expected length)
        length = min(len(text) / 1000, 1.0)
        
        # Vocabulary factor (based on unique words)
        words = text.lower().split()
        unique_words = len(set(words))
        vocab_complexity = min(unique_words / 200, 1.0)
        
        # Structure factor (based on sentence length and punctuation)
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(1, len(sentences))
        structure_complexity = min(avg_sentence_length / 20, 1.0)
        
        # Reasoning factor (based on keywords and patterns)
        reasoning_keywords = ['because', 'therefore', 'thus', 'hence', 'since', 'so', 'consequently']
        reasoning_count = sum(1 for word in words if word.lower() in reasoning_keywords)
        reasoning_complexity = min(reasoning_count / 5, 1.0)
        
        # Apply weights from settings
        factors = {
            'length': length * self.settings.length_weight,
            'vocabulary': vocab_complexity * self.settings.vocab_weight,
            'structure': structure_complexity * self.settings.structure_weight,
            'reasoning': reasoning_complexity * self.settings.reasoning_weight,
        }
        
        return factors
    
    def _calculate_overall_complexity(self, factors: Dict[str, float]) -> float:
        """
        Calculate overall complexity from individual factors.
        
        Args:
            factors: Dictionary with complexity factors
            
        Returns:
            Overall complexity score (0.0 to 1.0)
        """
        # Sum weighted factors
        complexity = sum(factors.values())
        
        # Normalize to 0-1 range
        complexity = min(max(complexity, 0.0), 1.0)
        
        return complexity
    
    async def _determine_domain(self, text: str, embedding: Optional[List[float]]) -> str:
        """
        Determine the domain of the given text.
        
        Args:
            text: The input text
            embedding: Text embedding (optional)
            
        Returns:
            Domain name
        """
        # Simple keyword-based domain detection
        # In a real implementation, this would use a more sophisticated approach
        
        text_lower = text.lower()
        
        # Check for programming domains
        if any(kw in text_lower for kw in ['python', 'code', 'function', 'class', 'programming']):
            return 'python'
            
        # Check for math domains
        if any(kw in text_lower for kw in ['math', 'calculate', 'equation', 'formula', 'computation']):
            return 'math'
            
        # Check for creative domains
        if any(kw in text_lower for kw in ['story', 'write', 'creative', 'imagine', 'fiction']):
            return 'creative'
            
        # Check for chat domains
        if any(kw in text_lower for kw in ['hello', 'hi', 'hey', 'how are you', 'what is your name']):
            return 'chat'
            
        # Default to general domain
        return 'general'
    
    async def route_to_module(self, text: str) -> str:
        """
        Route the given text to the appropriate module.
        
        Args:
            text: The input text
            
        Returns:
            Module name
        """
        # Analyze complexity
        analysis = await self.analyze_complexity(text)
        complexity = analysis['complexity']
        domain = analysis['domain']
        
        self.logger.info(f"Routing query with complexity {complexity:.2f} and domain '{domain}'")
        
        # Apply routing rules
        for module_name, rules in self.settings.routing_rules.items():
            # Check if this is the default module
            if rules.get('default', False):
                default_module = module_name
                continue
                
            # Check complexity constraints
            if 'complexity_min' in rules and complexity < rules['complexity_min']:
                continue
                
            if 'complexity_max' in rules and complexity > rules['complexity_max']:
                continue
                
            # Check domain constraints
            if 'domains' in rules and domain not in rules['domains']:
                continue
                
            # All constraints satisfied, use this module
            self.logger.debug(f"Selected module: {module_name}")
            return module_name
        
        # Fall back to default module
        self.logger.debug(f"Using default module: {default_module}")
        return default_module


# Test function
async def test_complexity_analyzer():
    """Test the ComplexityAnalyzer functionality."""
    from .config import WitsV3Config
    from .llm_interface import get_llm_interface
    import os
    
    print("Testing ComplexityAnalyzer...")
    
    # Load config
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    config_file_path = os.path.join(project_root_dir, "config.yaml")
    
    config = WitsV3Config.from_yaml(config_file_path)
    
    # Create LLM interface
    llm = get_llm_interface(config)
    
    # Create analyzer
    analyzer = ComplexityAnalyzer(config, llm)
    
    # Test queries
    test_queries = [
        "Hello, how are you today?",
        "Write a Python function to calculate the Fibonacci sequence recursively.",
        "Explain the philosophical implications of quantum mechanics on our understanding of reality.",
        "What is 2+2?",
        "Write a short story about a robot who discovers emotions."
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Analyze complexity
        analysis = await analyzer.analyze_complexity(query)
        
        print(f"Complexity: {analysis['complexity']:.2f}")
        print(f"Domain: {analysis['domain']}")
        print(f"Factors:")
        for factor, value in analysis['factors'].items():
            print(f"  {factor}: {value:.2f}")
            
        # Route to module
        module = await analyzer.route_to_module(query)
        print(f"Routed to module: {module}")
    
    print("\nComplexityAnalyzer tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_complexity_analyzer())
