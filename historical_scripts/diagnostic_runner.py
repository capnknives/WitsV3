#!/usr/bin/env python
"""
WitsV3 Diagnostic Tool
Performs automated diagnostics on WitsV3 system components.
"""

import os
import sys
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("WitsV3_Diagnostic")

# Import WitsV3 modules
try:
    from core.config import load_config, WitsV3Config
    from core.llm_interface import OllamaInterface, get_llm_interface
    from core.memory_manager import MemoryManager
    from core.tool_registry import ToolRegistry
except ImportError as e:
    logger.error(f"Failed to import WitsV3 modules: {e}")
    sys.exit(1)

class WitsV3Diagnostic:
    """Diagnostic tool for WitsV3 system."""

    def __init__(self):
        self.config = None
        self.llm_interface = None
        self.memory_manager = None
        self.tool_registry = None
        self.results = {
            "config": {"status": "Not checked", "details": {}},
            "llm_interface": {"status": "Not checked", "details": {}},
            "memory_manager": {"status": "Not checked", "details": {}},
            "tool_registry": {"status": "Not checked", "details": {}},
            "overall": {"status": "Not checked", "details": {}}
        }

    async def check_config(self) -> bool:
        """Check configuration loading."""
        logger.info("Checking configuration...")
        try:
            self.config = load_config()
            logger.info(f"Configuration loaded: version {self.config.version}")
            self.results["config"] = {
                "status": "Pass",
                "details": {
                    "version": self.config.version,
                    "project_name": self.config.project_name,
                    "llm_provider": self.config.llm_interface.default_provider,
                    "ollama_url": self.config.ollama_settings.url,
                    "default_model": self.config.ollama_settings.default_model
                }
            }
            return True
        except Exception as e:
            logger.error(f"Configuration check failed: {e}")
            self.results["config"] = {
                "status": "Fail",
                "details": {"error": str(e)}
            }
            return False

    async def check_llm_interface(self) -> bool:
        """Check LLM interface functionality."""
        if not self.config:
            logger.error("Cannot check LLM interface: Configuration not loaded")
            self.results["llm_interface"] = {
                "status": "Fail",
                "details": {"error": "Configuration not loaded"}
            }
            return False

        logger.info("Checking LLM interface...")
        try:
            self.llm_interface = get_llm_interface(self.config)
            logger.info(f"LLM interface initialized with provider: {self.config.llm_interface.default_provider}")

            # Test basic generation
            test_prompt = "Say hello world in one short sentence."
            logger.info("Testing text generation...")
            response = await self.llm_interface.generate_text(test_prompt)
            logger.info(f"Generated text: {response[:50]}...")

            # Test streaming (collect first 3 chunks)
            logger.info("Testing text streaming...")
            stream_chunks = []
            async for chunk in self.llm_interface.stream_text("Count from 1 to 5 briefly."):
                stream_chunks.append(chunk)
                if len(stream_chunks) >= 3:
                    logger.info(f"Streaming works, received {len(stream_chunks)} chunks")
                    break

            # Test embedding
            logger.info("Testing embeddings...")
            embedding = await self.llm_interface.get_embedding("Test embedding")
            embedding_size = len(embedding)
            logger.info(f"Generated embedding with {embedding_size} dimensions")

            self.results["llm_interface"] = {
                "status": "Pass",
                "details": {
                    "provider": self.config.llm_interface.default_provider,
                    "model": self.config.ollama_settings.default_model,
                    "text_generation": "Success" if response else "Fail",
                    "streaming": "Success" if stream_chunks else "Fail",
                    "embedding_size": embedding_size
                }
            }
            return True
        except Exception as e:
            logger.error(f"LLM interface check failed: {e}")
            self.results["llm_interface"] = {
                "status": "Fail",
                "details": {"error": str(e)}
            }
            return False

    async def check_memory_manager(self) -> bool:
        """Check memory manager functionality."""
        if not self.config:
            logger.error("Cannot check memory manager: Configuration not loaded")
            self.results["memory_manager"] = {
                "status": "Fail",
                "details": {"error": "Configuration not loaded"}
            }
            return False

        logger.info("Checking memory manager...")
        try:
            if self.llm_interface:
                self.memory_manager = MemoryManager(self.config, self.llm_interface)
            else:
                # If LLM interface test failed, create a new one just for this test
                temp_llm = get_llm_interface(self.config)
                self.memory_manager = MemoryManager(self.config, temp_llm)

            # Test memory storage
            test_content = f"Test memory content created during diagnostic at {asyncio.get_event_loop().time()}"
            test_memory = await self.memory_manager.add_memory_segment(
                content=test_content,
                metadata={"source": "diagnostic", "importance": 0.5}
            )
            logger.info(f"Memory stored with ID: {test_memory.id}")

            # Test memory retrieval
            search_results = await self.memory_manager.search_memory("test diagnostic", limit=5)
            logger.info(f"Memory search returned {len(search_results)} results")

            self.results["memory_manager"] = {
                "status": "Pass",
                "details": {
                    "backend": self.config.memory_manager.backend,
                    "total_segments": len(await self.memory_manager.get_memory_segments()),
                    "store_memory": "Success" if test_memory else "Fail",
                    "search_memory": "Success" if search_results else "Fail"
                }
            }
            return True
        except Exception as e:
            logger.error(f"Memory manager check failed: {e}")
            self.results["memory_manager"] = {
                "status": "Fail",
                "details": {"error": str(e)}
            }
            return False

    async def check_tool_registry(self) -> bool:
        """Check tool registry functionality."""
        if not self.config:
            logger.error("Cannot check tool registry: Configuration not loaded")
            self.results["tool_registry"] = {
                "status": "Fail",
                "details": {"error": "Configuration not loaded"}
            }
            return False

        logger.info("Checking tool registry...")
        try:
            self.tool_registry = ToolRegistry()

            # Get list of registered tools
            tools = self.tool_registry.get_all_tools()
            tool_names = [tool.name for tool in tools]
            logger.info(f"Tool registry has {len(tools)} tools: {', '.join(tool_names[:5])}...")

            # Test basic tools - find calculator tool
            from tools.calculator_tool import CalculatorTool
            calc_tool = CalculatorTool()
            calc_result = await calc_tool.execute("2 + 2")
            logger.info(f"Calculator result: {calc_result}")

            # Test datetime tool
            from tools.datetime_tool import DateTimeTool
            dt_tool = DateTimeTool()
            dt_result = await dt_tool.execute("current_time")
            logger.info(f"DateTime result: {dt_result}")

            self.results["tool_registry"] = {
                "status": "Pass",
                "details": {
                    "total_tools": len(tools),
                    "tool_names": tool_names[:5],
                    "calculator_test": "Success" if calc_result == "4" else "Fail",
                    "datetime_test": "Success" if dt_result else "Fail"
                }
            }
            return True
        except Exception as e:
            logger.error(f"Tool registry check failed: {e}")
            self.results["tool_registry"] = {
                "status": "Fail",
                "details": {"error": str(e)}
            }
            return False

    async def run_diagnostics(self):
        """Run all diagnostic checks."""
        logger.info("Starting WitsV3 diagnostics...")

        # Run checks
        config_ok = await self.check_config()
        llm_ok = await self.check_llm_interface()
        memory_ok = await self.check_memory_manager()
        tools_ok = await self.check_tool_registry()

        # Determine overall status
        all_checks = [config_ok, llm_ok, memory_ok, tools_ok]
        all_components = ["config", "llm_interface", "memory_manager", "tool_registry"]

        if all(all_checks):
            overall_status = "Pass"
            logger.info("All diagnostic checks passed!")
        elif not any(all_checks):
            overall_status = "Critical Failure"
            logger.error("All diagnostic checks failed!")
        else:
            overall_status = "Partial Failure"
            failed = [comp for i, comp in enumerate(all_components) if not all_checks[i]]
            logger.warning(f"Some diagnostic checks failed: {', '.join(failed)}")

        self.results["overall"] = {
            "status": overall_status,
            "details": {
                "components_checked": len(all_checks),
                "components_passed": sum(all_checks),
                "components_failed": len(all_checks) - sum(all_checks)
            }
        }

        # Output results
        logger.info("Diagnostic Results:")
        logger.info(json.dumps(self.results, indent=2))

        # Save results to file
        results_file = Path("diagnostic_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Diagnostic results saved to {results_file}")

        return overall_status == "Pass"

async def main():
    """Main function to run diagnostics."""
    diagnostic = WitsV3Diagnostic()
    success = await diagnostic.run_diagnostics()
    return 0 if success else 1

if __name__ == "__main__":
    print("WitsV3 Diagnostic Tool")
    print("=" * 50)
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nDiagnostic interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Diagnostic failed with error: {e}")
        sys.exit(1)
