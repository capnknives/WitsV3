"""
Response generator module for the Adaptive LLM System.

This module handles response generation and streaming.
"""

import logging
import asyncio
import torch
from typing import Any, AsyncGenerator, Optional

from .tokenizer import AdaptiveTokenizer


class ResponseGenerator:
    """Handles response generation for the Adaptive LLM System."""
    
    def __init__(self, tokenizer: AdaptiveTokenizer):
        """
        Initialize the ResponseGenerator.
        
        Args:
            tokenizer: The tokenizer to use
        """
        self.tokenizer = tokenizer
        self.logger = logging.getLogger("WitsV3.ResponseGenerator")
        
    async def generate_response(
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
            response = await self.tokenizer.decode(output_ids[0, input_ids.shape[1]:])
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response with module: {e}")
            raise
    
    async def stream_response(
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
            
        Yields:
            Chunks of the generated response
        """
        try:
            # This is a placeholder for actual streaming
            # In a real implementation, this would use the appropriate streaming method
            
            # Simulate streaming
            response = await self.generate_response(module, input_ids, max_tokens, temperature, **kwargs)
            
            # Split response into chunks
            chunk_size = 10
            for i in range(0, len(response), chunk_size):
                yield response[i:i+chunk_size]
                await asyncio.sleep(0.1)  # Simulate delay
                
        except Exception as e:
            self.logger.error(f"Error streaming response with module: {e}")
            raise
    
    async def stream_cached_response(self, response: str, chunk_size: int = 10) -> AsyncGenerator[str, None]:
        """
        Stream a cached response.
        
        Args:
            response: The cached response
            chunk_size: Size of chunks to yield
            
        Yields:
            Chunks of the cached response
        """
        # Split response into chunks
        for i in range(0, len(response), chunk_size):
            yield response[i:i+chunk_size]
            await asyncio.sleep(0.1)  # Simulate delay
    
    async def generate_with_fallback(
        self,
        primary_generator,
        fallback_generator,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        **kwargs
    ):
        """
        Generate response with fallback option.
        
        Args:
            primary_generator: Primary generation function
            fallback_generator: Fallback generation function
            prompt: The input prompt
            max_tokens: Maximum number of tokens
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional arguments
            
        Returns:
            Generated response or async generator
        """
        try:
            return await primary_generator(prompt, max_tokens, temperature, stream, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary generation failed: {e}, using fallback")
            
            if stream:
                # For streaming, return the AsyncGenerator directly
                return fallback_generator.stream_text(
                    prompt, 
                    max_tokens=max_tokens, 
                    temperature=temperature, 
                    **kwargs
                )
            else:
                # For non-streaming, await the result
                return await fallback_generator.generate_text(
                    prompt, 
                    max_tokens=max_tokens, 
                    temperature=temperature, 
                    **kwargs
                )
    
    def prepare_generation_kwargs(self, **kwargs) -> dict:
        """
        Prepare keyword arguments for generation.
        
        Args:
            **kwargs: Raw keyword arguments
            
        Returns:
            Processed keyword arguments
        """
        # Filter out None values
        generation_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Set defaults
        defaults = {
            'top_p': 0.95,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'length_penalty': 1.0,
            'num_beams': 1,
        }
        
        for key, value in defaults.items():
            if key not in generation_kwargs:
                generation_kwargs[key] = value
                
        return generation_kwargs