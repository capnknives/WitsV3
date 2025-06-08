"""
Background Agent for WitsV3
Runs in Docker container for isolated execution
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, AsyncGenerator
from aiohttp import web

from agents.base_agent import BaseAgent
from core.config import WitsV3Config
from core.llm_interface import OllamaInterface
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry
from core.schemas import StreamData

logger = logging.getLogger("WitsV3.BackgroundAgent")

class BackgroundAgent(BaseAgent):
    """
    Agent that runs in a Docker container for isolated execution.
    Handles environment-specific configuration and tool access.
    """
    
    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: OllamaInterface,
        memory_manager: Optional[MemoryManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        docker_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_name, config, llm_interface, memory_manager)
        
        self.tool_registry = tool_registry
        self.docker_config = docker_config or {}
        
        # Docker-specific settings
        self.container_id = os.getenv("HOSTNAME", "unknown")
        self.is_docker = os.getenv("WITSV3_DOCKER_ENV", "false").lower() == "true"
        self.cursor_integration = os.getenv("CURSOR_INTEGRATION", "false").lower() == "true"
        
        # Initialize Docker-specific tools if available
        if self.tool_registry and self.is_docker:
            self._initialize_docker_tools()
        
        logger.info(f"Background agent initialized in {'Docker' if self.is_docker else 'local'} environment")
        if self.cursor_integration:
            logger.info("Cursor integration enabled")
    
    def _initialize_docker_tools(self):
        """Initialize tools specific to Docker environment"""
        # Add Docker-specific tools to registry
        # These tools will be available only in Docker environment
        pass
    
    async def run(
        self,
        task: str,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Execute a task in the background agent.
        
        Args:
            task: The task to execute
            **kwargs: Additional parameters
            
        Yields:
            StreamData objects showing progress
        """
        try:
            # Log container info
            yield self.stream_thinking(f"Running in container: {self.container_id}")
            
            # Execute task with Docker-specific handling
            if self.is_docker:
                yield self.stream_thinking("Executing in Docker environment")
                # Add Docker-specific execution logic here
            
            # Process the task
            yield self.stream_thinking(f"Processing task: {task}")
            
            # Use tools if available
            if self.tool_registry:
                yield self.stream_thinking("Using available tools")
                # Add tool execution logic here
            
            # Complete task
            yield self.stream_result("Task completed successfully")
            
        except Exception as e:
            logger.error(f"Error in background agent: {e}")
            yield self.stream_error(f"Task failed: {str(e)}")

async def health_check(request):
    """Health check endpoint for Docker"""
    return web.Response(text="OK")

async def start_web_server():
    """Start web server for Cursor integration"""
    app = web.Application()
    app.router.add_get('/health', health_check)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8000)
    await site.start()
    
    logger.info("Web server started on port 8000")

async def main():
    """Main entry point for background agent"""
    # Load configuration
    config = WitsV3Config.from_yaml()
    
    # Initialize components
    llm_interface = OllamaInterface(config)
    memory_manager = MemoryManager(config, llm_interface)
    await memory_manager.initialize()
    
    tool_registry = ToolRegistry()
    
    # Create background agent
    agent = BackgroundAgent(
        agent_name="BackgroundAgent",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager,
        tool_registry=tool_registry
    )
    
    # Start web server if Cursor integration is enabled
    if os.getenv("CURSOR_INTEGRATION", "false").lower() == "true":
        await start_web_server()
    
    # Run the agent
    async for stream in agent.run("Initialize background processing"):
        if stream.type == "result":
            logger.info(stream.content)
        elif stream.type == "error":
            logger.error(stream.content)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the agent
    asyncio.run(main()) 