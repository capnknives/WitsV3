"""
AdaptiveLLMInterface for WitsV3 Adaptive LLM System.

This module implements the main interface for the Adaptive LLM System,
integrating the complexity analyzer, dynamic module loader, and semantic cache.
"""

import logging
import asyncio
import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple, Union, AsyncGenerator

import torch
import numpy as np

from .config import WitsV3Config
from .llm_interface import BaseLLMInterface
from .complexity_analyzer import ComplexityAnalyzer
from .dynamic_module_loader import DynamicModuleLoader
from .semantic_cache import SemanticCache
from .adaptive_llm_config import AdaptiveLLMSettings

class AdaptiveLLMInterface(BaseLLMInterface):
    """
    Main interface for the Adaptive LLM System.
    
    The AdaptiveLLMInterface integrates the complexity analyzer, dynamic module
    loader, and semantic cache to provide an adaptive LLM system.
    """
    
    def __init__(self, config: WitsV3Config, llm_interface: BaseLLMInterface):
        """
        Initialize the AdaptiveLLMInterface.
        
        Args:
            config: System configuration
            llm_interface: Base LLM interface for embeddings and fallback
        """
        self.config = config
        self.settings = AdaptiveLLMSettings()  # Use defaults
        self.llm_interface = llm_interface
        self.logger = logging.getLogger("WitsV3.AdaptiveLLMInterface")
        
        # Initialize components
        self.complexity_analyzer = ComplexityAnalyzer(config, llm_interface)
        self.module_loader = DynamicModuleLoader(config)
        self.semantic_cache = SemanticCache(config, llm_interface)
        
        # Performance tracking
        self.performance_log = []
        
        # Fallback counter
        self.fallback_attempts = 0
        
        self.logger.info("AdaptiveLLMInterface initialized")
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text response to the given prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments
            
        Returns:
            The generated text response
        """
        result = await self.generate(prompt, stream=False, **kwargs)
        # Ensure we return a string, not an AsyncGenerator
        assert isinstance(result, str), "Expected string result from non-streaming generate"
        return result
    
    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream text response to the given prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments
            
        Yields:
            Chunks of the generated text response
        """
        result = await self.generate(prompt, stream=True, **kwargs)
        # Ensure we return an AsyncGenerator
        assert not isinstance(result, str), "Expected AsyncGenerator result from streaming generate"
        return result
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional arguments
            
        Returns:
            The generated response or an async generator for streaming
        """
        start_time = time.time()
        
        # Use default values if not provided
        max_tokens = max_tokens or self.settings.default_max_tokens
        temperature = temperature or self.settings.default_temperature
        
        # Check cache if enabled
        if self.settings.enable_caching:
            cached_response = await self.semantic_cache.find_similar(
                prompt,
                threshold=self.settings.cache_similarity_threshold
            )
            
            if cached_response:
                self.logger.info(f"Cache hit with similarity: {cached_response['similarity']:.4f}")
                
                # Track performance
                if self.settings.enable_performance_tracking:
                    self._track_performance(
                        prompt=prompt,
                        response=cached_response['response'],
                        module='cache',
                        complexity=cached_response['metadata'].get('complexity', 0.0),
                        generation_time=time.time() - start_time,
                        cache_hit=True,
                        similarity=cached_response['similarity']
                    )
                
                if stream:
                    return self._stream_cached_response(cached_response['response'])
                else:
                    return cached_response['response']
        
        # Analyze complexity
        analysis = await self.complexity_analyzer.analyze_complexity(prompt)
        complexity = analysis['complexity']
        domain = analysis['domain']
        
        self.logger.info(f"Prompt complexity: {complexity:.2f}, domain: {domain}")
        
        # Route to appropriate module
        module_name = await self.complexity_analyzer.route_to_module(prompt)
        
        try:
            # Load module
            module = await self.module_loader.load_module(module_name, complexity)
            
            # Tokenize input
            input_ids = await self._tokenize(prompt)
            
            # Generate response
            if stream:
                return self._stream_response(module, input_ids, max_tokens, temperature, **kwargs)
            else:
                response = await self._generate_response(module, input_ids, max_tokens, temperature, **kwargs)
                
                # Cache response
                if self.settings.enable_caching:
                    generation_time = time.time() - start_time
                    await self.semantic_cache.add_pattern(
                        prompt,
                        response,
                        module_name,
                        complexity,
                        generation_time
                    )
                
                # Track performance
                if self.settings.enable_performance_tracking:
                    self._track_performance(
                        prompt=prompt,
                        response=response,
                        module=module_name,
                        complexity=complexity,
                        generation_time=time.time() - start_time,
                        cache_hit=False
                    )
                
                return response
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            
            # Fallback to base LLM interface
            if self.settings.fallback_to_base_on_error and self.fallback_attempts < self.settings.max_fallback_attempts:
                self.fallback_attempts += 1
                self.logger.warning(f"Falling back to base LLM interface (attempt {self.fallback_attempts})")
                
                if stream:
                    # For streaming, return the AsyncGenerator directly
                    return self.llm_interface.stream_text(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
                else:
                    # For non-streaming, await the result
                    response = await self.llm_interface.generate_text(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
                    
                    # Track performance
                    if self.settings.enable_performance_tracking:
                        self._track_performance(
                            prompt=prompt,
                            response=response,
                            module='fallback',
                            complexity=complexity,
                            generation_time=time.time() - start_time,
                            cache_hit=False,
                            error=str(e)
                        )
                    
                    return response
            else:
                # Reset fallback counter
                self.fallback_attempts = 0
                raise
    
    async def _tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize the input text.
        
        Args:
            text: The input text
            
        Returns:
            Tensor of token IDs
        """
        # This is a placeholder for actual tokenization
        # In a real implementation, this would use the appropriate tokenizer
        
        # Simulate tokenization
        return torch.tensor([[101] + [ord(c) % 1000 for c in text[:100]] + [102]])
    
    async def _generate_response(
        self,
        module: Any,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """
        Generate a response using the given module.
        
        Args:
            module: The module to use
            input_ids: The tokenized input
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            The generated response
        """
        try:
            # Generate response
            output_ids = module.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                **kwargs
            )
            
            # Decode response
            response = await self._decode(output_ids[0, input_ids.shape[1]:])
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response with module: {e}")
            raise
    
    async def _stream_response(
        self,
        module: Any,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response using the given module.
        
        Args:
            module: The module to use
            input_ids: The tokenized input
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            An async generator that yields chunks of the generated response
        """
        try:
            # This is a placeholder for actual streaming
            # In a real implementation, this would use the appropriate streaming method
            
            # Simulate streaming
            response = await self._generate_response(module, input_ids, max_tokens, temperature, **kwargs)
            
            # Create and return an async generator
            async def response_generator():
                # Split response into chunks
                chunk_size = 10
                for i in range(0, len(response), chunk_size):
                    yield response[i:i+chunk_size]
                    await asyncio.sleep(0.1)  # Simulate delay
            
            return response_generator()
                
        except Exception as e:
            self.logger.error(f"Error streaming response with module: {e}")
            raise
    
    async def _stream_cached_response(self, response: str) -> AsyncGenerator[str, None]:
        """
        Stream a cached response.
        
        Args:
            response: The cached response
            
        Returns:
            An async generator that yields chunks of the cached response
        """
        # Create and return an async generator
        async def cached_response_generator():
            # Split response into chunks
            chunk_size = 10
            for i in range(0, len(response), chunk_size):
                yield response[i:i+chunk_size]
                await asyncio.sleep(0.1)  # Simulate delay
        
        return cached_response_generator()
    
    async def _decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: The token IDs
            
        Returns:
            The decoded text
        """
        # This is a placeholder for actual decoding
        # In a real implementation, this would use the appropriate tokenizer
        
        # Simulate decoding
        return ''.join([chr(t % 128) for t in token_ids.tolist()])
    
    def _track_performance(
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
            asyncio.create_task(self._save_performance_log())
    
    async def _save_performance_log(self) -> None:
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
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for the given text.
        
        Args:
            text: The input text
            
        Returns:
            The embedding vector
        """
        # Delegate to base LLM interface
        return await self.llm_interface.get_embedding(text)
    
    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """
        Get information about a module.
        
        Args:
            module_name: The name of the module
            
        Returns:
            Dictionary with module information
        """
        return self.module_loader.get_module_info(module_name)
    
    def get_resource_usage(self) -> Dict[str, float]:
        """
        Get current resource usage.
        
        Returns:
            Dictionary with resource usage information
        """
        return self.module_loader.get_resource_usage()
    
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
    
    async def shutdown(self) -> None:
        """Shutdown the AdaptiveLLMInterface."""
        self.logger.info("Shutting down AdaptiveLLMInterface")
        
        # Save performance log
        if self.settings.enable_performance_tracking:
            await self._save_performance_log()
            
        # Unload all modules
        await self.module_loader.unload_all()
        
        self.logger.info("AdaptiveLLMInterface shutdown complete")


    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status information.
        
        Returns:
            Dictionary with system status information
        """
        return {
            'resource_usage': self.get_resource_usage(),
            'performance_stats': self.get_performance_stats(),
            'modules': {
                name: self.get_module_info(name)
                for name in self.module_loader.module_registry
            }
        }

# Test function
async def test_adaptive_llm_interface():
    """Test the AdaptiveLLMInterface functionality."""
    from .config import WitsV3Config
    from .llm_interface import get_llm_interface
    import os
    
    print("Testing AdaptiveLLMInterface...")
    
    # Load config
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    config_file_path = os.path.join(project_root_dir, "config.yaml")
    
    config = WitsV3Config.from_yaml(config_file_path)
    
    # Create base LLM interface
    base_llm = get_llm_interface(config)
    
    # Create adaptive LLM interface
    adaptive_llm = AdaptiveLLMInterface(config, base_llm)
    
    # Test generation
    test_prompts = [
        "Hello, how are you today?",
        "Write a Python function to calculate the Fibonacci sequence recursively.",
        "Explain the philosophical implications of quantum mechanics on our understanding of reality.",
        "What is 2+2?",
        "Write a short story about a robot who discovers emotions."
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Generate response
        response = await adaptive_llm.generate(prompt)
        
        print(f"Response: {response[:100]}...")
    
    # Test performance stats
    print("\nPerformance stats:")
    stats = adaptive_llm.get_performance_stats()
    print(f"Count: {stats['count']}")
    print(f"Average generation time: {stats['avg_generation_time']:.2f}s")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2f}")
    print(f"Module usage: {stats['module_usage']}")
    print(f"Complexity distribution: {stats['complexity_distribution']}")
    
    # Test resource usage
    print("\nResource usage:")
    usage = adaptive_llm.get_resource_usage()
    print(f"VRAM usage: {usage['vram_usage'] / 1e9:.2f} GB / {usage['vram_budget'] / 1e9:.2f} GB ({usage['vram_percent']:.1f}%)")
    print(f"RAM usage: {usage['ram_usage'] / 1e9:.2f} GB / {usage['ram_budget'] / 1e9:.2f} GB ({usage['ram_percent']:.1f}%)")
    
    # Test shutdown
    print("\nShutting down...")
    await adaptive_llm.shutdown()
    
    print("\nAdaptiveLLMInterface tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_adaptive_llm_interface())
