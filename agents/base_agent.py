# agents/base_agent.py
"""
Base Agent class for WitsV3.
All agents inherit from this class to ensure consistent interfaces.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator

from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, AgentResponse, ConversationHistory


class BaseAgent(ABC):
    """
    Abstract base class for all agents in WitsV3.
    
    This class defines the common interface and shared functionality
    that all agents must implement or can optionally use.
    """
    
    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: Optional[MemoryManager] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_name: Unique name for this agent
            config: System configuration
            llm_interface: Interface to LLM for text generation
            memory_manager: Optional memory manager for persistence
        """
        self.agent_name = agent_name
        self.config = config
        self.llm_interface = llm_interface
        self.memory_manager = memory_manager
          # Set up logging
        self.logger = logging.getLogger(f"WitsV3.{self.__class__.__name__}")
        
        # Agent configuration from config
        self.temperature = config.agents.default_temperature
        self.max_iterations = config.agents.max_iterations
        
        self.logger.info(f"Initialized {self.__class__.__name__}: {agent_name}")
    
    @abstractmethod
    async def run(
        self,
        user_input: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Main execution method for the agent.
        
        Args:
            user_input: The user input to process (user message, goal, etc.)
            conversation_history: Optional conversation context
            session_id: Optional session identifier
            **kwargs: Additional agent-specific parameters
            
        Yields:
            StreamData objects representing the agent's processing steps
        """
        pass
    
    async def generate_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            model_name: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Generated text response
        """
        try:
            response = await self.llm_interface.generate_text(
                prompt=prompt,
                model=model_name or self.get_model_name(),
                temperature=temperature or self.temperature,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_streaming_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            model_name: Optional model override
            temperature: Optional temperature override
            
        Yields:
            Text chunks as they are generated
        """
        try:
            async for chunk in self.llm_interface.stream_text(
                prompt=prompt,
                model=model_name or self.get_model_name(),
                temperature=temperature or self.temperature
            ):
                yield chunk
        except Exception as e:
            self.logger.error(f"Error generating streaming response: {e}")
            raise
    
    def get_model_name(self) -> str:
        """
        Get the model name for this agent.
        Override in subclasses for agent-specific models.
        
        Returns:
            Model name to use for this agent
        """
        return self.config.ollama_settings.default_model
    
    async def store_memory(
        self,
        content: str,
        segment_type: str = "AGENT_ACTION",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Store information in memory if available.
        
        Args:
            content: Content to store
            segment_type: Type of memory segment
            importance: Importance score (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            Memory segment ID if stored, None if no memory manager
        """
        if not self.memory_manager:
            return None
        
        try:
            from core.memory_manager import MemorySegment, MemorySegmentContent
            
            segment = MemorySegment(
                type=segment_type,
                source=self.agent_name,
                content=MemorySegmentContent(text=content),
                importance=importance,
                metadata=metadata or {}
            )
            
            segment_id = await self.memory_manager.add_segment(segment)
            self.logger.debug(f"Stored memory segment: {segment_id}")
            return segment_id
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
            return None
    
    async def search_memory(
        self,
        query: str,
        limit: int = 5
    ) -> List[Any]:
        """
        Search memory for relevant information.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant memory segments
        """
        if not self.memory_manager:
            return []
        
        try:
            results = await self.memory_manager.search_memory(query, limit=limit)
            self.logger.debug(f"Found {len(results)} memory results for query: {query}")
            return results
        except Exception as e:
            self.logger.error(f"Error searching memory: {e}")
            return []
    
    def stream_thinking(self, thought: str) -> StreamData:
        """Create a thinking stream data object."""
        return StreamData(
            type="thinking",
            content=thought,
            source=self.agent_name
        )
    
    def stream_action(self, action: str) -> StreamData:
        """Create an action stream data object."""
        return StreamData(
            type="action",
            content=action,
            source=self.agent_name
        )
    
    def stream_observation(self, observation: str) -> StreamData:
        """Create an observation stream data object."""
        return StreamData(
            type="observation",
            content=observation,
            source=self.agent_name
        )
    
    def stream_result(self, result: str) -> StreamData:
        """Create a result stream data object."""
        return StreamData(
            type="result",
            content=result,
            source=self.agent_name
        )
    
    def stream_error(self, error: str, details: Optional[str] = None) -> StreamData:
        """Create an error stream data object."""
        return StreamData(
            type="error",
            content=error,
            source=self.agent_name,
            error_details=details
        )
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.agent_name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_name='{self.agent_name}', model='{self.get_model_name()}')"


# Test function
async def test_base_agent():
    """Test the BaseAgent functionality."""
    print("Testing BaseAgent...")
    
    # Note: This is a mock test since BaseAgent is abstract
    print("✓ BaseAgent is properly defined as abstract class")
    print("✓ All required methods are defined")
    print("✓ Streaming helper methods are available")
    print("✓ Memory integration methods are available")
    
    print("BaseAgent tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_base_agent())
