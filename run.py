# run.py
"""
Main entry point for WitsV3.
CLI-first AI orchestration system with LLM wrapper architecture.
"""

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

from core.config import load_config, WitsV3Config
from core.llm_interface import OllamaInterface
from core.memory_manager import MemoryManager, BasicMemoryBackend
from core.tool_registry import ToolRegistry
from core.schemas import ConversationHistory, StreamData
from agents.wits_control_center_agent import WitsControlCenterAgent
from agents.llm_driven_orchestrator import LLMDrivenOrchestrator
from tools.base_tool import FileReadTool, FileWriteTool, ListDirectoryTool, DateTimeTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/witsv3.log') if Path('logs').exists() else logging.NullHandler()
    ]
)

logger = logging.getLogger("WitsV3.Main")


class WitsV3System:
    """
    Main WitsV3 system class that orchestrates all components.
    """
    
    def __init__(self, config: WitsV3Config):
        """
        Initialize the WitsV3 system.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.session_histories = {}
        
        # Initialize core components
        self.llm_interface = None
        self.memory_manager = None
        self.tool_registry = None
        self.control_center = None
        self.orchestrator = None
        
        logger.info(f"Initializing WitsV3 system v{config.version}")
    
    async def initialize(self):
        """Initialize all system components."""
        try:
            # Initialize LLM interface
            self.llm_interface = OllamaInterface(
                config=self.config
            )            
            # Test LLM connection (simple test)
            logger.info("LLM interface initialized")
              
            # Initialize memory manager
            if self.config.memory_manager.backend == "basic":
                self.memory_manager = MemoryManager(
                    config=self.config,
                    llm_interface=self.llm_interface
                )
                await self.memory_manager.initialize()
                logger.info("Memory manager initialized with basic backend")
            else:
                logger.warning(f"Memory backend '{self.config.memory_manager.backend}' not implemented, continuing without memory")
            
            # Initialize tool registry
            self.tool_registry = ToolRegistry()
            
            # Register additional tools
            self.tool_registry.register_tool(FileReadTool())
            self.tool_registry.register_tool(FileWriteTool())
            self.tool_registry.register_tool(ListDirectoryTool())
            self.tool_registry.register_tool(DateTimeTool())
            
            logger.info(f"Tool registry initialized with {len(self.tool_registry.tools)} tools")
            
            # Initialize orchestrator
            self.orchestrator = LLMDrivenOrchestrator(
                agent_name="LLMOrchestrator",
                config=self.config,
                llm_interface=self.llm_interface,
                memory_manager=self.memory_manager,
                tool_registry=self.tool_registry
            )
            logger.info("LLM-driven orchestrator initialized")
            
            # Initialize control center
            self.control_center = WitsControlCenterAgent(
                agent_name="WitsControlCenter",
                config=self.config,
                llm_interface=self.llm_interface,
                memory_manager=self.memory_manager,
                orchestrator_agent=self.orchestrator
            )
            logger.info("WITS Control Center initialized")
            
            logger.info("WitsV3 system fully initialized and ready!")
            
        except Exception as e:
            logger.error(f"Failed to initialize WitsV3 system: {e}")
            raise
    
    async def process_user_input(
        self, 
        user_input: str, 
        session_id: Optional[str] = None
    ) -> str:
        """
        Process user input and return response.
        
        Args:
            user_input: User's input message
            session_id: Optional session identifier
            
        Returns:
            Complete response from the system
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get or create conversation history
        if session_id not in self.session_histories:
            self.session_histories[session_id] = ConversationHistory(session_id=session_id)
        
        conversation = self.session_histories[session_id]
        conversation.add_message("user", user_input)
        
        # Process through control center
        response_parts = []
        
        try:
            async for stream_data in self.control_center.run(
                user_input=user_input,
                conversation_history=conversation,
                session_id=session_id
            ):
                if self.config.cli.show_thoughts or stream_data.type != "thinking":
                    response_parts.append(self._format_stream_data(stream_data))
            
            final_response = "\n".join(response_parts)
            conversation.add_message("assistant", final_response)
            
            return final_response
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logger.error(error_msg)
            conversation.add_message("assistant", error_msg)
            return error_msg
    
    def _format_stream_data(self, stream_data: StreamData) -> str:
        """
        Format stream data for CLI display.
        
        Args:
            stream_data: Stream data to format
            
        Returns:
            Formatted string for display
        """
        if stream_data.type == "thinking":
            return f"[THINKING] {stream_data.content}"
        elif stream_data.type == "action":
            return f"[ACTION] {stream_data.content}"
        elif stream_data.type == "observation":
            return f"[OBSERVATION] {stream_data.content}"
        elif stream_data.type == "result":
            return f"[RESULT] {stream_data.content}"
        elif stream_data.type == "error":
            return f"[ERROR] {stream_data.content}"
        elif stream_data.type == "clarification":
            return f"[CLARIFICATION] {stream_data.content}"
        else:
            return f"[INFO] {stream_data.content}"
    
    async def shutdown(self):
        """Shutdown the system gracefully."""
        logger.info("Shutting down WitsV3 system...")
        
        if self.memory_manager:
            # Save any pending memory operations
            pass
        
        logger.info("WitsV3 system shutdown complete")


async def run_cli():
    """Run WitsV3 in CLI mode."""
    print("Starting WitsV3 - CLI Mode")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config("config.yaml")
        
        # Initialize system
        wits_system = WitsV3System(config)
        await wits_system.initialize()
        
        print("\nWitsV3 is ready! Type 'quit' or 'exit' to stop.")
        print("Use 'help' for assistance, or just start chatting!\n")
        
        session_id = str(uuid.uuid4())
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye! Thanks for using WitsV3!")
                    break
                
                if user_input.lower() == 'help':
                    print_help()
                    continue
                
                if user_input.lower() == 'clear':
                    # Clear session history
                    if session_id in wits_system.session_histories:
                        del wits_system.session_histories[session_id]
                    session_id = str(uuid.uuid4())
                    print("Session cleared! Starting fresh.")
                    continue
                
                # Process the input
                print("\nWits:", end=" ")
                response = await wits_system.process_user_input(user_input, session_id)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for using WitsV3!")
                break
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                logger.error(f"CLI error: {e}", exc_info=True)
    
    except Exception as e:
        print(f"Failed to start WitsV3: {str(e)}")
        logger.error(f"Startup error: {e}", exc_info=True)
        return 1
    
    finally:
        if 'wits_system' in locals():
            await wits_system.shutdown()
    
    return 0


def print_help():
    """Print help information."""
    help_text = """
WitsV3 CLI Commands:
  help       - Show this help message
  clear      - Clear conversation history and start fresh
  quit/exit  - Exit WitsV3

Usage Tips:
  - Just type naturally! WitsV3 will understand your requests
  - Ask for clarification if needed
  - WitsV3 can help with various tasks using its tools
  - Conversation history is maintained within each session

Available Tools:
  - File operations (read, write, list directories)
  - Calculations
  - Date/time information
  - Memory search and storage
"""
    print(help_text)


async def main():
    """Main entry point."""
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Run CLI by default
    return await run_cli()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
