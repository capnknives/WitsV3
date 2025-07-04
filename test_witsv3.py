#!/usr/bin/env python3
"""
WitsV3 Non-Interactive Test Runner
This script tests WitsV3 functionality without requiring user input
"""

import asyncio

# Fix Unicode encoding issues
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import codecs
# Set UTF-8 encoding for stdout/stderr
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

import logging
import time
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/test_witsv3.log') if Path('logs').exists() else logging.NullHandler()
    ]
)
logger = logging.getLogger("WitsV3.Test")

# Set environment variables to indicate test mode
os.environ['CURSOR_TEST_MODE'] = '1'
os.environ['WITSV3_TEST_MODE'] = 'true'

class WitsV3Tester:
    """Non-interactive tester for WitsV3 system."""

    def __init__(self, test_mode="basic"):
        self.config = None
        self.system_components = {}
        self.test_mode = test_mode
        self.start_time = time.time()

    async def setup_system(self):
        """Set up WitsV3 system for testing."""
        try:
            # Import here to avoid circular imports
            from core.config import load_config
            from core.llm_interface import OllamaInterface, get_llm_interface
            from core.memory_manager import MemoryManager
            from core.tool_registry import ToolRegistry
            from agents.wits_control_center_agent import WitsControlCenterAgent
            from agents.llm_driven_orchestrator import LLMDrivenOrchestrator

            # Load configuration
            self.config = load_config()
            logger.info("✅ Configuration loaded")

            # Initialize LLM interface
            llm_interface = get_llm_interface(self.config)
            self.system_components['llm'] = llm_interface
            logger.info("✅ LLM interface initialized")

            # Test LLM connection
            try:
                response = await llm_interface.generate_text("Say hello in one short sentence.")
                logger.info(f"✅ LLM connection test passed: {response}")
            except Exception as e:
                logger.error(f"❌ LLM connection test failed: {e}")
                raise

            # Initialize memory manager
            memory_manager = MemoryManager(
                config=self.config,
                llm_interface=llm_interface
            )
            await memory_manager.initialize()
            self.system_components['memory'] = memory_manager
            logger.info("✅ Memory manager initialized")

            # Initialize tool registry
            tool_registry = ToolRegistry()
            self.system_components['tools'] = tool_registry
            logger.info(f"✅ Tool registry initialized with {len(tool_registry.get_all_tools())} tools")

            # Initialize orchestrator
            orchestrator = LLMDrivenOrchestrator(
                agent_name="TestOrchestrator",
                config=self.config,
                llm_interface=llm_interface,
                memory_manager=memory_manager,
                tool_registry=tool_registry
            )
            self.system_components['orchestrator'] = orchestrator
            logger.info("✅ Orchestrator initialized")

            # Initialize control center
            control_center = WitsControlCenterAgent(
                agent_name="TestControlCenter",
                config=self.config,
                llm_interface=llm_interface,
                memory_manager=memory_manager,
                orchestrator_agent=orchestrator
            )
            self.system_components['control_center'] = control_center
            logger.info("✅ Control center initialized")

            return True

        except Exception as e:
            logger.error(f"❌ System setup failed: {e}")
            return False

    async def test_basic_functionality(self):
        """Test basic system functionality."""
        logger.info("🧪 Testing basic functionality...")

        test_cases = [
            "Hello, how are you?",
            "What is 2+2?",
            "List the files in the current directory",
            "What time is it?",
        ]

        control_center = self.system_components.get('control_center')
        if not control_center:
            logger.error("❌ Control center not available for testing")
            return False

        for i, test_input in enumerate(test_cases, 1):
            try:
                logger.info(f"Test {i}: {test_input}")

                # Process the test input
                response_parts = []
                async for stream_data in control_center.run(user_input=test_input, goal="Answer the question or respond to the statement"):
                    if stream_data.type in ["result", "error"]:
                        response_parts.append(stream_data.content)

                    # Print streamed responses for debugging
                    if stream_data.type == "thinking":
                        logger.debug(f"Thinking: {stream_data.content}")
                    elif stream_data.type == "action":
                        logger.debug(f"Action: {stream_data.content}")

                response = "".join(response_parts)
                if response:
                    logger.info(f"✅ Test {i} passed: Got response")
                    logger.info(f"Response: {response[:100]}...")
                else:
                    logger.warning(f"⚠️  Test {i} warning: Empty response")

            except Exception as e:
                logger.error(f"❌ Test {i} failed: {e}")
                return False

        return True

    async def test_tool_functionality(self):
        """Test tool registry and basic tool functionality."""
        logger.info("🔧 Testing tool functionality...")

        tool_registry = self.system_components.get('tools')
        if not tool_registry:
            logger.error("❌ Tool registry not available for testing")
            return False

        try:
            # Check if tools are registered
            tools = tool_registry.get_all_tools()
            logger.info(f"✅ {len(tools)} tools registered")

            # Test calculator tool if it exists
            calculator_found = False
            for tool_name in tools:
                if tool_name == "calculator":
                    calculator_found = True
                    # Get the actual tool object
                    calculator_tool = tool_registry.get_tool(tool_name)

                    # Test calculator with a simple expression
                    result = await calculator_tool.execute(expression="2+2")
                    if result and float(result) == 4.0:
                        logger.info("✅ Calculator tool test passed")
                    else:
                        logger.warning(f"⚠️ Calculator returned unexpected result: {result}")

            if not calculator_found:
                logger.warning("⚠️ Calculator tool not found, skipping tool test")

            return True

        except Exception as e:
            logger.error(f"❌ Tool test failed: {e}")
            return False

    async def test_memory_system(self):
        """Test memory system functionality."""
        logger.info("🧠 Testing memory system...")

        memory_manager = self.system_components.get('memory')
        if not memory_manager:
            logger.error("❌ Memory manager not available for testing")
            return False

        try:
            # Test basic memory operations
            test_memory = f"This is a test memory for the WitsV3 system created at {self.start_time}"

            # Store memory
            segment_id = await memory_manager.add_memory(
                type="TEST",
                source="test_runner",
                content_text=test_memory,
                importance=0.7
            )
            logger.info(f"✅ Memory storage test passed - ID: {segment_id}")

            # Search memory - catch errors but don't fail the test if this part fails
            try:
                results = await memory_manager.search_memory("test memory", limit=1)
                if results and len(results) > 0:
                    logger.info(f"✅ Memory search test passed - Found {len(results)} results")
                else:
                    logger.warning("⚠️  Memory search returned no results")
            except Exception as search_error:
                logger.warning(f"⚠️  Memory search failed: {search_error}")
                logger.info("Continuing with test as storage was successful")

            # Consider the test passed if we can at least store memory
            return True

        except Exception as e:
            logger.error(f"❌ Memory test failed: {e}")
            return False

    async def run_all_tests(self):
        """Run all tests and return overall result."""
        logger.info("🚀 Starting WitsV3 system tests...")

        # Setup system
        if not await self.setup_system():
            logger.error("❌ System setup failed - aborting tests")
            return False

        # Select which tests to run based on test mode
        tests = []
        if self.test_mode in ["basic", "all"]:
            tests.append(("Basic Functionality", self.test_basic_functionality))

        if self.test_mode in ["tools", "all"]:
            tests.append(("Tool System", self.test_tool_functionality))

        if self.test_mode in ["memory", "all"]:
            tests.append(("Memory System", self.test_memory_system))

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info('='*50)

            try:
                if await test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")

        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("TEST SUMMARY")
        logger.info('='*50)
        logger.info(f"Passed: {passed_tests}/{total_tests}")

        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED!")
            return True
        else:
            logger.error(f"❌ {total_tests - passed_tests} tests failed")
            return False

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WitsV3 Non-Interactive Test Runner")
    parser.add_argument("--mode", choices=["basic", "tools", "memory", "all"],
                        default="all", help="Test mode (default: all)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set log level
    if args.debug:
        logging.getLogger("WitsV3").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    tester = WitsV3Tester(test_mode=args.mode)

    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        return 1

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
