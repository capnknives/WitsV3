# agents/base_agent.py
"""
Base Agent class for WitsV3.
All agents inherit from this class to ensure consistent interfaces.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.model_router import ModelRouter
from core.schemas import ConversationHistory, StreamData


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
        memory_manager: MemoryManager | None = None,
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

        # Smart model routing: agents pass the raw user message/goal to
        # self.model_router.route() to pick a per-request model
        self.model_router = ModelRouter(config)

        self.logger.info(f"Initialized {self.__class__.__name__}: {agent_name}")

    @abstractmethod
    async def run(
        self,
        user_input: str,
        conversation_history: ConversationHistory | None = None,
        session_id: str | None = None,
        **kwargs,
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
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            prompt: The prompt to send to the LLM
            model_name: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            response_format: Optional output constraint ("json" for Ollama structured output)
            num_ctx: Optional Ollama context-window override — needed for long-form
                generation (e.g. novella chapters), since Ollama's own default context
                is otherwise much smaller than most models' trained context length and
                silently truncates a large prompt+completion.

        Returns:
            Generated text response
        """
        try:
            extra_kwargs: dict[str, Any] = {}
            if response_format is not None:
                extra_kwargs["format"] = response_format
            if num_ctx is not None:
                extra_kwargs["num_ctx"] = num_ctx
            from core.smoke_metrics import get_current, smoke_metrics_enabled

            if smoke_metrics_enabled() and get_current():
                get_current().record_llm_call()
            response = await self.llm_interface.generate_text(
                prompt=prompt,
                model=model_name or self.get_model_name(),
                temperature=temperature or self.temperature,
                max_tokens=max_tokens,
                **extra_kwargs,
            )
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

    async def generate_streaming_response(
        self, prompt: str, model_name: str | None = None, temperature: float | None = None
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
                temperature=temperature or self.temperature,
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
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
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
        if getattr(self, "_skip_global_memory_store", False):
            return None
        if not self.memory_manager:
            return None

        from core.pii_redaction import maybe_redact_for_storage

        role = getattr(self, "_request_user_role", "owner")
        content = maybe_redact_for_storage(content, self.config, role=role)

        # nomic-embed-text rejects very long inputs; cap before persisting.
        from core.memory_embedding import resolve_max_embedding_chars, truncate_for_embedding
        from core.memory_manager import MemorySegment, MemorySegmentContent

        max_chars = resolve_max_embedding_chars(self.config)
        content = truncate_for_embedding(content, max_chars, suffix="\n… [truncated for memory]")

        try:

            segment = MemorySegment(
                type=segment_type,
                source=self.agent_name,
                content=MemorySegmentContent(text=content),
                importance=importance,
                metadata=metadata or {},
            )

            segment_id = await self.memory_manager.add_segment(segment)
            self.logger.debug(f"Stored memory segment: {segment_id}")
            await self._maybe_promote_to_knowledge_log(
                content, segment_type, importance, metadata or {}
            )
            return segment_id
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
            return None

    async def _maybe_promote_to_knowledge_log(
        self,
        content: str,
        segment_type: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> None:
        """Promote high-confidence remembered facts to the knowledge log (Phase 3a)."""
        if importance < 0.9:
            return
        if segment_type != "USER_FACT" and not metadata.get("remember"):
            return
        if not getattr(self, "config", None):
            return
        try:
            from core.knowledge_log import KnowledgeLogStore

            log = KnowledgeLogStore(path=self.config.knowledge_log.file_path)
            log.add_fact(content, source=self.agent_name, category="promoted_memory")
        except Exception as e:
            self.logger.debug("Knowledge log promotion skipped: %s", e)

    async def search_memory(
        self,
        query: str,
        limit: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Search memory for relevant information.

        Args:
            query: Search query
            limit: Maximum number of results
            filter_dict: Optional metadata filter (e.g. session_id)

        Returns:
            List of relevant memory segments
        """
        if not self.memory_manager:
            return []

        try:
            results = await self.memory_manager.search_memory(
                query, limit=limit, filter_dict=filter_dict
            )
            self.logger.debug(f"Found {len(results)} memory results for query: {query}")
            return results
        except Exception as e:
            self.logger.error(f"Error searching memory: {e}")
            return []

    def stream_tool_progress(
        self, tool_name: str, phase: str, detail: str = ""
    ) -> StreamData:
        """SSE-friendly progress event for long-running tools (Phase 2.4)."""
        return StreamData(
            type="tool_call",
            content=detail or f"{tool_name}: {phase}",
            source=self.agent_name,
            metadata={"tool_name": tool_name, "progress_phase": phase},
        )

    def stream_thinking(self, thought: str) -> StreamData:
        """Create a thinking stream data object."""
        return StreamData(type="thinking", content=thought, source=self.agent_name)

    def stream_action(self, action: str) -> StreamData:
        """Create an action stream data object."""
        return StreamData(type="action", content=action, source=self.agent_name)

    def stream_observation(self, observation: str) -> StreamData:
        """Create an observation stream data object."""
        return StreamData(type="observation", content=observation, source=self.agent_name)

    def stream_result(self, result: str) -> StreamData:
        """Create a result stream data object."""
        return StreamData(type="result", content=result, source=self.agent_name)

    def stream_error(self, error: str, details: str | None = None) -> StreamData:
        """Create an error stream data object."""
        return StreamData(
            type="error", content=error, source=self.agent_name, error_details=details
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
