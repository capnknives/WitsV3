"""
Conversation Handler for WitsV3

Handles casual conversation while integrating with enhanced task capabilities.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator

from core.config import WitsV3Config
from core.schemas import StreamData
from core.llm_interface import OllamaInterface
from core.memory_manager import MemoryManager
from core.concrete_meta_reasoning import WitsV3MetaReasoningEngine
from core.tool_composition import IntelligentToolComposer

logger = logging.getLogger(__name__)

class ConversationHandler:
    """
    Handles casual conversation while integrating with enhanced task capabilities.

    This module serves as a bridge between casual conversation and the enhanced
    task-oriented capabilities of WitsV3.
    """

    def __init__(
        self,
        config: WitsV3Config,
        llm_interface: OllamaInterface,
        memory_manager: Optional[MemoryManager] = None,
        meta_reasoning: Optional[WitsV3MetaReasoningEngine] = None,
        tool_composer: Optional[IntelligentToolComposer] = None
    ):
        self.config = config
        self.llm_interface = llm_interface
        self.memory_manager = memory_manager

        # Enhanced capabilities
        self.meta_reasoning = meta_reasoning
        self.tool_composer = tool_composer

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []

    async def handle_message(
        self,
        user_message: str
    ) -> AsyncGenerator[StreamData, None]:
        """
        Handle a user message, whether casual or task-oriented.

        Args:
            user_message: The user's message

        Yields:
            StreamData objects with the response
        """
        yield StreamData(thinking=f"Processing message: {user_message}")

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Determine if this is a task or casual conversation
        is_task = await self._is_task_oriented(user_message)

        if is_task:
            # Handle as a task using enhanced capabilities
            yield StreamData(thinking="Detected task-oriented request. Using enhanced capabilities.")
            async for data in self._handle_task(user_message):
                yield data
        else:
            # Handle as casual conversation
            yield StreamData(thinking="Detected casual conversation.")
            async for data in self._handle_conversation(user_message):
                yield data

    async def _is_task_oriented(self, message: str) -> bool:
        """
        Determine if a message is task-oriented or casual.

        Args:
            message: The user's message

        Returns:
            True if task-oriented, False if casual
        """
        # Simple heuristic based on verbs and complexity
        task_indicators = [
            "create", "make", "build", "analyze", "find", "search",
            "calculate", "compute", "implement", "develop", "generate",
            "summarize", "extract", "convert", "transform"
        ]

        question_indicators = ["?", "what", "how", "why", "when", "where", "who"]

        # Check for task indicators
        if any(indicator in message.lower() for indicator in task_indicators):
            return True

        # Check for complexity
        if len(message.split()) > 15:  # Longer messages tend to be tasks
            return True

        # Short questions are usually casual
        if any(indicator in message.lower() for indicator in question_indicators) and len(message.split()) < 10:
            return False

        # Default to casual for short messages
        return len(message.split()) > 8

    async def _handle_task(self, task: str) -> AsyncGenerator[StreamData, None]:
        """
        Handle a task-oriented message using enhanced capabilities.

        Args:
            task: The task message

        Yields:
            StreamData objects with the response
        """
        if not self.meta_reasoning:
            yield StreamData(thinking="Meta-reasoning engine not available, falling back to basic processing.")
            # Fall back to conversation handling
            async for data in self._handle_conversation(task):
                yield data
            return

        try:
            # Use meta-reasoning to analyze the problem
            context = {"source": "conversation"}
            problem_space = await self.meta_reasoning.analyze_problem_space(task, context)

            yield StreamData(thinking=f"Analyzed task as {problem_space.complexity.value} complexity.")

            # For now, just provide the analysis
            yield StreamData(thinking="Planning task execution...")

            # Create a response
            response = (
                f"I understand you'd like me to {task}. "
                f"This appears to be a {problem_space.complexity.value} task that will "
                f"require approximately {problem_space.estimated_steps} steps. "
                f"I'll help you with this task."
            )

            yield StreamData(content=response)

            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})

            # In a full implementation, would use tool composition and execute the task

        except Exception as e:
            self.logger.error(f"Error in task handling: {e}")
            yield StreamData(error=f"I encountered an error while processing your task: {e}")

            # Fall back to conversation
            yield StreamData(thinking="Falling back to conversational mode due to error.")
            async for data in self._handle_conversation(task):
                yield data

    async def _handle_conversation(self, message: str) -> AsyncGenerator[StreamData, None]:
        """
        Handle a casual conversation message.

        Args:
            message: The user's message

        Yields:
            StreamData objects with the response
        """
        try:
            # Prepare conversation history for the LLM
            formatted_history = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in self.conversation_history[-5:]  # Last 5 messages
            ])

            # Create a prompt for the LLM
            prompt = (
                f"You are WitsV3, a helpful and friendly AI assistant. Respond conversationally to this message.\n\n"
                f"CONVERSATION HISTORY:\n{formatted_history}\n\n"
                f"ASSISTANT: "
            )

            # Generate a response
            response = await self.llm_interface.generate_text(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7
            )

            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})

            yield StreamData(content=response)

        except Exception as e:
            self.logger.error(f"Error in conversation handling: {e}")
            yield StreamData(error=f"I encountered an error while responding: {e}")

async def test_conversation_handler():
    """Test the conversation handler"""
    from core.config import WitsV3Config
    from core.llm_interface import OllamaInterface

    # Load config
    config = WitsV3Config.from_yaml('config.yaml')

    # Create components
    llm_interface = OllamaInterface(config)

    # Create conversation handler
    handler = ConversationHandler(config, llm_interface)

    # Test casual conversation
    print("\n=== Testing casual conversation ===")
    message = "Hi there! How are you today?"
    print(f"USER: {message}")

    async for data in handler.handle_message(message):
        if data.thinking:
            print(f"[Thinking] {data.thinking}")
        if data.content:
            print(f"ASSISTANT: {data.content}")

    # Test task-oriented message
    print("\n=== Testing task-oriented message ===")
    message = "Can you analyze this dataset and create a summary of the key trends?"
    print(f"USER: {message}")

    async for data in handler.handle_message(message):
        if data.thinking:
            print(f"[Thinking] {data.thinking}")
        if data.content:
            print(f"ASSISTANT: {data.content}")

if __name__ == "__main__":
    asyncio.run(test_conversation_handler())
