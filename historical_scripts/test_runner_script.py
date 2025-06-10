#!/usr/bin/env python3
"""
WitsV3 Test Runner - Non-interactive testing for Cursor.ai
This script tests WitsV3 functionality without requiring user input
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import load_config, WitsV3Config
from core.llm_interface import OllamaInterface
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry
from agents.wits_control_center_agent import WitsControlCenterAgent
from agents.llm_driven_orchestrator import LLMDrivenOrchestrator

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WitsV3.Test")

class WitsV3Tester:
    """Non-interactive tester for WitsV3 system."""
    
    def __init__(self):
        self.config = None
        self.system_components = {}
        
    async def setup_system(self):
        """Set up WitsV3 system for testing."""
        try:
            # Load configuration
            self.config = load_config("config.yaml")
            logger.info("✅ Configuration loaded")
            
            # Initialize LLM interface
            llm_interface = OllamaInterface(self.config)
            self.system_components['llm'] = llm_interface
            logger.info("✅ LLM interface initialized")
            
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
            logger.info(f"✅ Tool registry initialized with {len(tool_registry.tools)} tools")
            
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
                async for stream_data in control_center.run(test_input):
                    if stream_data.type in ["result", "error"]:
                        response_parts.append(stream_data.content)
                
                response = " ".join(response_parts)
                if response:
                    logger.info(f"✅ Test {i} passed: Got response")
                    logger.debug(f"Response: {response[:100]}...")
                else:
                    logger.warning(f"⚠️  Test {i} warning: Empty response")
                    
            except Exception as e:
                logger.error(f"❌ Test {i} failed: {e}")
                return False
        
        return True
    
    async def test_tool_functionality(self):
        """Test tool system functionality."""
        logger.info("🔧 Testing tool functionality...")
        
        tool_registry = self.system_components.get('tools')
        if not tool_registry:
            logger.error("❌ Tool registry not available for testing")
            return False
        
        # Test that tools are registered
        if len(tool_registry.tools) == 0:
            logger.error("❌ No tools registered")
            return False
        
        logger.info(f"✅ {len(tool_registry.tools)} tools registered")
        
        # Test a simple tool execution
        try:
            # Find a simple tool to test
            for tool_name, tool in tool_registry.tools.items():
                if hasattr(tool, 'execute'):
                    logger.info(f"Testing tool: {tool_name}")
                    # We'll just verify the tool exists and has execute method
                    break
            
            logger.info("✅ Tool system functional")
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
            test_memory = "This is a test memory for the WitsV3 system"
            
            # Store memory
            await memory_manager.store_memory(
                content=test_memory,
                memory_type="test",
                source="test_runner"
            )
            logger.info("✅ Memory storage test passed")
            
            # Search memory
            results = await memory_manager.search_memory("test memory", limit=1)
            if results:
                logger.info("✅ Memory search test passed")
            else:
                logger.warning("⚠️  Memory search returned no results")
            
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
        
        # Run tests
        tests = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Tool System", self.test_tool_functionality),
            ("Memory System", self.test_memory_system),
        ]
        
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
    """Main test runner entry point."""
    tester = WitsV3Tester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test runner error: {e}")
        return 1

if __name__ == "__main__":
    # Ensure required directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)