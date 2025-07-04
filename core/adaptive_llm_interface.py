"""
AdaptiveLLMInterface for WitsV3 Adaptive LLM System.

This module implements the main interface for the Adaptive LLM System,
integrating the complexity analyzer, dynamic module loader, and semantic cache.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Union, AsyncGenerator

from .config import WitsV3Config
from .llm_interface import BaseLLMInterface
from .complexity_analyzer import ComplexityAnalyzer
from .dynamic_module_loader import DynamicModuleLoader
from .semantic_cache import SemanticCache
from .adaptive_llm_config import AdaptiveLLMSettings
from .adaptive import PerformanceTracker, AdaptiveTokenizer, ResponseGenerator


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

        # Initialize modular components
        self.tokenizer = AdaptiveTokenizer()
        self.response_generator = ResponseGenerator(self.tokenizer)
        self.performance_tracker = PerformanceTracker(self.settings)

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
        # When stream=True, self.generate returns an AsyncGenerator
        # which we need to properly await and yield from
        generator = await self.generate(prompt, stream=True, **kwargs)

        # Yield chunks from the generator
        async for chunk in generator:
            yield chunk

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
                    self.performance_tracker.track_performance(
                        prompt=prompt,
                        response=cached_response['response'],
                        module='cache',
                        complexity=cached_response['metadata'].get('complexity', 0.0),
                        generation_time=time.time() - start_time,
                        cache_hit=True,
                        similarity=cached_response['similarity']
                    )

                if stream:
                    return self.response_generator.stream_cached_response(cached_response['response'])
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
            input_ids = await self.tokenizer.tokenize(prompt)

            # Generate response
            if stream:
                return self.response_generator.stream_response(
                    module, input_ids, max_tokens, temperature, **kwargs
                )
            else:
                response = await self.response_generator.generate_response(
                    module, input_ids, max_tokens, temperature, **kwargs
                )

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
                    self.performance_tracker.track_performance(
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

                response = await self.response_generator.generate_with_fallback(
                    self.generate,  # Primary generator
                    self.llm_interface,  # Fallback generator
                    prompt,
                    max_tokens,
                    temperature,
                    stream,
                    **kwargs
                )

                # Track performance for non-streaming fallback
                if not stream and self.settings.enable_performance_tracking:
                    self.performance_tracker.track_performance(
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

    async def get_embedding(self, text: str, model=None) -> List[float]:
        # 'model' parameter is ignored but accepted for compatibility with MemoryManager
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
        return self.performance_tracker.get_performance_stats()

    async def shutdown(self) -> None:
        """Shutdown the AdaptiveLLMInterface."""
        self.logger.info("Shutting down AdaptiveLLMInterface")

        # Save performance log
        if self.settings.enable_performance_tracking:
            await self.performance_tracker.save_performance_log()

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
