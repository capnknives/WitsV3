import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the project root to sys.path for test discovery
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import asyncio
from typing import AsyncGenerator, List, Optional
from agents.self_repair_agent import SelfRepairAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.schemas import StreamData


class MockLLM(BaseLLMInterface):
    def __init__(self, config: WitsV3Config):
        super().__init__(config)
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        return '{"task_type": "health_check", "urgency": "medium", "scope": "system", "focus_areas": ["performance", "reliability"], "parameters": {}}'
    
    async def stream_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        yield '{"task_type": "health_check"}'
        yield ' more text'
    
    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding


async def main():
    config = WitsV3Config()
    llm = MockLLM(config)
    agent = SelfRepairAgent('test_agent', config, llm)
    print('Agent initialized successfully')
    
    # Test running the agent
    print('Testing agent run method...')
    async for stream_data in agent.run("Perform a health check"):
        if isinstance(stream_data, StreamData):
            print(f"{stream_data.type}: {stream_data.content[:50]}...")
    
    print('Test completed successfully')


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
