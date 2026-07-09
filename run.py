# run.py
"""
Main entry point for WitsV3.
CLI-first AI orchestration system with LLM wrapper architecture.
"""

import asyncio
# Fix Unicode encoding issues
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
# Set UTF-8 encoding for stdout/stderr (Windows consoles default to cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import logging
import os
import sys
import time
import uuid
import subprocess
from pathlib import Path
from typing import Optional, List

from core.config import load_config, WitsV3Config
from core.runtime_paths import ensure_runtime_layout, main_log_path
from core.conversation_compaction import maybe_flush_conversation_memory
from core.llm_interface import OllamaInterface, BaseLLMInterface
from core.memory_manager import MemoryManager, BasicMemoryBackend
from core.tool_registry import ToolRegistry
from core.schemas import ConversationHistory, StreamData
from core.user_errors import format_cli_error
from agents.wits_control_center_agent import WitsControlCenterAgent
from agents.llm_driven_orchestrator import LLMDrivenOrchestrator
from tools.mcp_tool_registry import MCPToolRegistry
# Note: Tool imports removed - tools are auto-discovered by ToolRegistry

# Import file watcher
try:
    from core.file_watcher import AsyncFileWatcher
    HAS_FILE_WATCHER = True
except ImportError:
    HAS_FILE_WATCHER = False
    logger = logging.getLogger("WitsV3.Main")
    logger.warning("File watcher not available. Install watchdog package for auto-restart functionality.")

# Flag to indicate if this is a restart or test mode
IS_RESTART = "--restart" in sys.argv
IS_TEST_MODE = any(arg in sys.argv for arg in ["--test", "--validate", "--check"])
IS_CURSOR_TEST = os.getenv('CURSOR_TEST_MODE') == '1' or os.getenv('CI') == 'true'

# Ensure runtime dirs exist before attaching the log file handler.
ensure_runtime_layout()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(main_log_path()), encoding='utf-8'),
    ]
)

# Custom filter to handle emoji characters in logs
class EmojiFilter(logging.Filter):
    def filter(self, record):
        if hasattr(record, 'msg') and isinstance(record, logging.LogRecord) and isinstance(record.msg, str):
            # Replace emoji with text representation
            record.msg = record.msg.encode('ascii', 'replace').decode('ascii')
        return True

# Add the filter to the root logger
logging.getLogger().addFilter(EmojiFilter())

logger = logging.getLogger("WitsV3.Main")


# Flag to indicate if this is a restart
IS_RESTART = "--restart" in sys.argv

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
        self.session_histories = {}        # Initialize core components
        self.file_watcher = None
        self.self_repair_scheduler = None
        self.llm_interface: Optional[BaseLLMInterface] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.control_center: Optional[WitsControlCenterAgent] = None
        self.orchestrator = None
        self.neural_web = None
        self.mcp_registry = None

        # Specialized agents
        self.book_writing_agent = None
        self.coding_agent = None
        self.self_repair_agent = None

        logger.info(f"Initializing WitsV3 system v{config.version}")

    async def initialize(self):
        """Initialize all system components."""
        try:
            from core.session_store import load_persisted_sessions_into

            root = self.config.runtime_paths.root
            restored = load_persisted_sessions_into(self.session_histories, root)
            if restored:
                logger.info("Restored %s persisted chat session(s) from disk", restored)
            # Initialize LLM interface with model reliability
            from core.enhanced_llm_interface import get_enhanced_llm_interface
            self.llm_interface = get_enhanced_llm_interface(self.config)
            logger.info("Enhanced LLM interface with model reliability initialized")
                # Initialize memory manager
            self.memory_manager = MemoryManager(
                config=self.config,
                llm_interface=self.llm_interface
            )
            await self.memory_manager.initialize()
            logger.info(f"Memory manager initialized with {self.config.memory_manager.backend} backend")
              # Initialize neural web if using neural backend
            if self.config.memory_manager.backend == "neural":
                from core.neural_web_core import NeuralWeb
                from core.neural_memory_backend import NeuralMemoryBackend
                # Neural web is initialized within the NeuralMemoryBackend
                if isinstance(self.memory_manager.backend, NeuralMemoryBackend):
                    self.neural_web = self.memory_manager.backend.neural_web
                    logger.info("Neural web integration enabled")
              # Initialize tool registry with auto-discovery
            self.tool_registry = ToolRegistry()

            # Initialize MCP tool registry (external server connections are
            # skipped unless mcp_connect_on_startup is enabled — connecting to
            # unavailable node servers adds ~20s of failed attempts at boot)
            if self.config.tool_system.enable_mcp_tools and self.config.tool_system.mcp_connect_on_startup:
                self.mcp_registry = MCPToolRegistry(self.config.tool_system.mcp_tool_definitions_path)
                await self.mcp_registry.initialize(self.tool_registry)
            else:
                self.mcp_registry = None
                logger.info("MCP server connections skipped at startup (tool_system.mcp_connect_on_startup=false)")

            # Note: Tools are automatically discovered from tools/file_tools.py module
            # No manual registration needed for: FileReadTool, FileWriteTool, ListDirectoryTool, DateTimeTool

            logger.info(f"Tool registry initialized with {len(self.tool_registry.tools)} tools (auto-discovered)")

            # Wire shared dependencies into tools that opt in (e.g. document tools
            # need the live MemoryManager rather than their own instance)
            for tool in self.tool_registry.tools.values():
                if hasattr(tool, "set_dependencies"):
                    tool.set_dependencies(
                        self.config,
                        self.llm_interface,
                        self.memory_manager,
                        tool_registry=self.tool_registry,
                    )

            # Document RAG: ensure the documents folder exists and optionally
            # ingest new/changed files at startup
            if self.config.document_rag.enabled:
                Path(self.config.document_rag.documents_path).mkdir(parents=True, exist_ok=True)
                if self.config.document_rag.auto_ingest_on_startup:
                    ingest_tool = self.tool_registry.get_tool("ingest_documents")
                    if ingest_tool:
                        try:
                            result = await ingest_tool.execute()
                            logger.info(
                                "Document ingest: %s scanned, %s ingested, %s unchanged, "
                                "%s removed, %s chunks added",
                                result.get("files_scanned", 0), result.get("files_ingested", 0),
                                result.get("files_unchanged", 0), result.get("files_removed", 0),
                                result.get("chunks_added", 0),
                            )
                        except Exception as e:
                            logger.warning(f"Document ingest at startup failed (non-fatal): {e}")

            # Initialize specialized agents first
            await self._initialize_specialized_agents()
            logger.info("Specialized agents initialized")

            # Initialize orchestrator with neural capabilities if available
            if self.neural_web and self.config.memory_manager.backend == "neural":
                from agents.neural_orchestrator_agent import NeuralOrchestratorAgent
                self.orchestrator = NeuralOrchestratorAgent(
                    agent_name="NeuralOrchestrator",
                    config=self.config,
                    llm_interface=self.llm_interface,
                    memory_manager=self.memory_manager,
                    tool_registry=self.tool_registry,
                    neural_web=self.neural_web
                )
                logger.info("Neural orchestrator initialized")
            else:
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
                orchestrator_agent=self.orchestrator,
                specialized_agents={
                    "book_writing": self.book_writing_agent,
                    "coding": self.coding_agent,
                    "self_repair": self.self_repair_agent
                }
            )
            logger.info("WITS Control Center initialized with specialized agents")

            logger.info("WitsV3 system fully initialized")

            # Initialize file watcher if available
            if HAS_FILE_WATCHER and self.config.auto_restart_on_file_change:
                await self._initialize_file_watcher()

            if self.config.self_repair.daily_schedule_enabled:
                self._initialize_self_repair_schedule()

        except Exception as e:
            logger.error(f"Failed to initialize WitsV3 system: {e}")
            raise

    def _initialize_self_repair_schedule(self):
        """Run the self-repair agent's autonomous scan-and-fix on a cron
        schedule (default: daily at 3am). Guarded by
        self_repair.daily_schedule_enabled; restart_after_fix stays off by
        default so a scheduled repair never surprises an active session."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger

            self.self_repair_scheduler = AsyncIOScheduler()
            self.self_repair_scheduler.add_job(
                self._run_scheduled_self_repair,
                CronTrigger.from_crontab(self.config.self_repair.daily_schedule_cron),
                id="daily_self_repair",
                replace_existing=True,
            )
            self.self_repair_scheduler.start()
            logger.info(
                "Scheduled autonomous self-repair: cron '%s'",
                self.config.self_repair.daily_schedule_cron,
            )
        except Exception as e:
            logger.warning(f"Failed to schedule self-repair job (non-fatal): {e}")

    async def _run_scheduled_self_repair(self):
        """The actual job body: run the self-repair agent's autonomous scan
        and log the outcome. Never raises — a failed repair attempt should
        never take down the scheduler."""
        if not self.self_repair_agent:
            return
        logger.info("Scheduled self-repair scan starting...")
        try:
            results = []
            async for stream_data in self.self_repair_agent.run(
                "Scan for recent errors and fix any that can be safely resolved.",
                session_id=f"scheduled-self-repair-{uuid.uuid4().hex[:8]}",
            ):
                if stream_data.type in ("result", "observation"):
                    results.append(f"[{stream_data.type}] {stream_data.content}")
            logger.info("Scheduled self-repair scan finished:\n%s", "\n".join(results))
        except Exception as e:
            logger.error(f"Scheduled self-repair scan failed: {e}")

    async def _initialize_file_watcher(self):
        """Initialize the file watcher for auto-restart functionality."""
        try:
            # Create file watcher
            self.file_watcher = AsyncFileWatcher(
                restart_callback=self._restart_system,
                patterns=["*.py"]  # Watch Python files
            )

            # Start file watcher
            watch_paths = [
                os.path.join(os.getcwd(), "agents"),
                os.path.join(os.getcwd(), "core"),
                os.path.join(os.getcwd(), "tools"),
                os.path.join(os.getcwd(), "gui")
            ]
            await self.file_watcher.start(watch_paths)

            logger.info(f"File watcher initialized and watching for changes in Python files")
        except Exception as e:
            logger.error(f"Failed to initialize file watcher: {e}")

    def _restart_system(self):
        """Restart the system when a file changes."""
        logger.info("Restarting WitsV3 system due to file changes...")

        try:
            # Get the current script path
            script_path = sys.argv[0]

            # Build the restart command
            cmd = [sys.executable, script_path, "--restart"]

            # Add any other command line arguments
            for arg in sys.argv[1:]:
                if arg != "--restart":  # Avoid duplicate restart flags
                    cmd.append(arg)

            # Start the new process
            subprocess.Popen(cmd)

            # Exit the current process
            logger.info("New WitsV3 process started, shutting down current process...")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to restart system: {e}")

    async def _initialize_specialized_agents(self):
        """Initialize specialized agents with neural web integration"""
        if not self.llm_interface or not self.memory_manager or not self.tool_registry:
            raise RuntimeError("Core components must be initialized before specialized agents")

        try:
            # Book Writing Agent with fallback
            from agents.enhanced_book_agent_with_fallback import EnhancedBookWritingAgent
            self.book_writing_agent = EnhancedBookWritingAgent(
                agent_name="BookWriter",
                config=self.config,
                llm_interface=self.llm_interface,
                memory_manager=self.memory_manager,
                neural_web=self.neural_web,
                tool_registry=self.tool_registry
            )

            # Advanced Coding Agent
            from agents.advanced_coding_agent import AdvancedCodingAgent
            self.coding_agent = AdvancedCodingAgent(
                agent_name="CodingExpert",
                config=self.config,
                llm_interface=self.llm_interface,
                memory_manager=self.memory_manager,
                neural_web=self.neural_web,
                tool_registry=self.tool_registry
            )

            # Self-Repair Agent
            from agents.self_repair_agent import SelfRepairAgent
            self.self_repair_agent = SelfRepairAgent(
                agent_name="SystemDoctor",
                config=self.config,
                llm_interface=self.llm_interface,
                memory_manager=self.memory_manager,
                neural_web=self.neural_web,
                tool_registry=self.tool_registry
            )

            logger.info("Specialized agents initialized")

        except Exception as e:
            logger.error(f"Error initializing specialized agents: {e}")

    async def test_system(self) -> bool:
        """Run system tests without user interaction."""
        logger.info("Running WitsV3 system tests...")

        try:
            # Test LLM connection
            response = await self.llm_interface.generate_text("Say hello in one short sentence.")
            logger.info(f"LLM test: {response}")

            # Test a basic tool
            calculator_tool = self.tool_registry.get_tool("calculator")

            if calculator_tool:
                result = await calculator_tool.execute(expression="2+2")
                logger.info(f"Calculator tool test: {result}")

            # Test memory system
            test_memory = f"This is a test memory created during system test at {time.time()}"
            segment_id = await self.memory_manager.add_memory(
                type="SYSTEM_MESSAGE",
                source="system_test",
                content_text=test_memory,
                importance=0.5,
                metadata={"source": "system_test"}
            )
            logger.info(f"Memory test: Created segment {segment_id}")

            # Test control center with a simple query
            test_query = "What is 2+2?"
            logger.info(f"Testing control center with query: {test_query}")

            response_parts = []
            async for stream_data in self.control_center.run(
                user_input=test_query,
                goal="Answer the question directly"
            ):
                if stream_data.type in ["result", "error"]:
                    response_parts.append(stream_data.content)
                elif stream_data.type == "thinking":
                    logger.debug(f"Thinking: {stream_data.content}")
                elif stream_data.type == "action":
                    logger.debug(f"Action: {stream_data.content}")

            response = "".join(response_parts)
            logger.info(f"Control center test response: {response}")

            logger.info("All system tests passed!")
            return True

        except Exception as e:
            logger.error(f"System test failed: {e}")
            return False
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

        if not self.control_center:
            raise RuntimeError("Control center not initialized")

        try:
            await maybe_flush_conversation_memory(
                conversation,
                history_window=self.config.agents.history_window,
                memory_manager=self.memory_manager,
                llm_interface=self.llm_interface,
                session_id=session_id,
            )
            async for stream_data in self.control_center.run(
                user_input=user_input,
                conversation_history=conversation,
                session_id=session_id
            ):
                if self.config.cli.show_thoughts or stream_data.type != "thinking":
                    response_parts.append(self._format_stream_data(stream_data))

            final_response = "\n".join(response_parts)
            conversation.add_message("assistant", final_response)
            try:
                from core.session_store import persist_session

                persist_session(conversation, self.config.runtime_paths.root)
            except Exception as persist_err:
                logger.warning("Session persist failed: %s", persist_err)

            return final_response

        except Exception as e:
            error_msg = format_cli_error(e, self.config.ollama_settings.url)
            logger.error(f"Error processing request: {e}")
            conversation.add_message("assistant", error_msg)
            try:
                from core.session_store import persist_session

                persist_session(conversation, self.config.runtime_paths.root)
            except Exception as persist_err:
                logger.warning("Session persist failed: %s", persist_err)
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

        # Stop file watcher if running
        if self.file_watcher:
            await self.file_watcher.stop()
            logger.info("File watcher shutdown complete")

        if self.self_repair_scheduler:
            self.self_repair_scheduler.shutdown(wait=False)
            logger.info("Self-repair scheduler shutdown complete")

        if self.memory_manager:
            # Save any pending memory operations
            pass

        if self.mcp_registry:
            await self.mcp_registry.shutdown()
            logger.info("MCP registry shutdown complete")

        logger.info("WitsV3 system shutdown complete")


async def run_test_mode():
    """Run WitsV3 in test mode to verify system integrity."""
    try:
        logger.info("Starting WitsV3 in test mode...")

        # Load configuration
        config = load_config()

        # Initialize system
        system = WitsV3System(config)
        await system.initialize()

        # Run system tests
        success = await system.test_system()

        # Shutdown system
        await system.shutdown()

        if success:
            logger.info("✅ WitsV3 system tests PASSED")
            return 0
        else:
            logger.error("❌ WitsV3 system tests FAILED")
            return 1

    except Exception as e:
        logger.error(f"Error during test mode: {e}")
        return 1
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
                ollama_url = wits_system.config.ollama_settings.url if wits_system.config else "http://localhost:11434"
                print(f"\n{format_cli_error(e, ollama_url)}")
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
    # Check if we're in test mode
    if IS_TEST_MODE or IS_CURSOR_TEST:
        return await run_test_mode()

    # Otherwise, run in CLI mode
    return await run_cli()



if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
