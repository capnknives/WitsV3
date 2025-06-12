"""Minimal self-repair agent used for testing."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncGenerator, Optional

from .base_agent import BaseAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, ConversationHistory


class SelfRepairAgent(BaseAgent):
    """Simplified implementation that streams a basic response."""

    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: Optional[MemoryManager] = None,
        **_: Any,
    ) -> None:
        super().__init__(agent_name, config, llm_interface, memory_manager)

    async def run(
        self,
        user_input: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamData, None]:
        """Return a very small fixed response for tests."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        yield StreamData(type="thinking", content="Analyzing request", source=self.agent_name)
        response = await self.llm_interface.generate_text(user_input)
        yield StreamData(type="result", content=response, source=self.agent_name)

