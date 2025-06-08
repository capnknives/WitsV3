"""
Background Agent for WitsV3
Handles scheduled tasks and system maintenance
"""

import asyncio
import logging
import os
import yaml
import psutil
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, AsyncGenerator, cast
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
from apscheduler.triggers.cron import CronTrigger  # type: ignore

from agents.base_agent import BaseAgent
from core.config import WitsV3Config
from core.llm_interface import OllamaInterface
from core.memory_manager import MemoryManager, BasicMemoryBackend
from core.tool_registry import ToolRegistry
from core.schemas import StreamData
from core.metrics import MetricsManager

logger = logging.getLogger("WitsV3.BackgroundAgent")

class BackgroundAgent(BaseAgent):
    """
    Agent that handles scheduled tasks and system maintenance.
    Runs in Docker container for isolated execution.
    """

    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: OllamaInterface,
        memory_manager: Optional[MemoryManager] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        super().__init__(agent_name, config, llm_interface, memory_manager)

        self.tool_registry = tool_registry
        self.scheduler = AsyncIOScheduler()
        self.metrics = MetricsManager()
        self.tasks_config = self._load_tasks_config()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.running = False

        logger.info("Background agent initialized")

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
            yield self.stream_thinking(f"Starting task: {task}")

            if task in self.tasks_config.get("tasks", {}):
                task_config = self.tasks_config["tasks"][task]
                if task_config.get("enabled", False):
                    await self._execute_task(task, task_config)
                    yield self.stream_result(f"Task {task} completed successfully")
                else:
                    yield self.stream_error(f"Task {task} is disabled")
            else:
                yield self.stream_error(f"Unknown task: {task}")

        except Exception as e:
            logger.error(f"Error in task {task}: {e}")
            yield self.stream_error(f"Task failed: {str(e)}")

    def _load_tasks_config(self) -> Dict[str, Any]:
        """Load background agent configuration"""
        config_path = os.path.join("config", "background_agent.yaml")
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load background agent config: {e}")
            return {"enabled": False}

    async def start(self):
        """Start the background agent and schedule tasks"""
        if self.running:
            return

        self.running = True
        self.scheduler.start()

        # Schedule all enabled tasks
        for task_name, task_config in self.tasks_config.get("tasks", {}).items():
            if task_config.get("enabled", False):
                self.scheduler.add_job(
                    self._run_task,
                    CronTrigger.from_crontab(task_config["schedule"]),
                    args=[task_name],
                    id=task_name,
                    replace_existing=True
                )
                logger.info(f"Scheduled task: {task_name}")

        # Start continuous monitoring
        asyncio.create_task(self._monitor_system_resources())

    async def stop(self):
        """Stop the background agent and all tasks"""
        self.running = False
        self.scheduler.shutdown()
        for task in self.active_tasks.values():
            task.cancel()
        self.active_tasks.clear()
        logger.info("Background agent stopped")

    async def _run_task(self, task_name: str):
        """Execute a scheduled task"""
        if task_name in self.active_tasks and not self.active_tasks[task_name].done():
            logger.warning(f"Task {task_name} is already running")
            return

        task_config = self.tasks_config["tasks"][task_name]
        self.active_tasks[task_name] = asyncio.create_task(
            self._execute_task(task_name, task_config)
        )

    async def _execute_task(self, task_name: str, task_config: Dict[str, Any]):
        """Execute a specific task with its configuration"""
        try:
            logger.info(f"Starting task: {task_name}")

            if task_name == "memory_maintenance":
                await self._maintain_memory(task_config["settings"])
            elif task_name == "semantic_cache_optimization":
                await self._optimize_semantic_cache(task_config["settings"])
            elif task_name == "system_monitoring":
                await self._monitor_system(task_config["settings"])
            elif task_name == "knowledge_graph_construction":
                await self._build_knowledge_graph(task_config["settings"])

            logger.info(f"Completed task: {task_name}")

        except Exception as e:
            logger.error(f"Error in task {task_name}: {e}")
            self.metrics.record_error(f"task_{task_name}")
        finally:
            if task_name in self.active_tasks:
                del self.active_tasks[task_name]

    async def _maintain_memory(self, settings: Dict[str, Any]):
        """Maintain memory by pruning and optimizing"""
        if not self.memory_manager:
            return

        # Prune old and low-importance segments
        cutoff_date = datetime.now() - timedelta(days=settings["max_age_days"])
        segments = await self.memory_manager.get_recent_memory(
            limit=settings["batch_size"]
        )

        for segment in segments:
            if (segment.timestamp < cutoff_date or
                segment.importance < settings["min_importance_threshold"]):
                # Remove segment
                pass  # Implement segment removal

    async def _optimize_semantic_cache(self, settings: Dict[str, Any]):
        """Optimize semantic cache by cleaning and reorganizing"""
        # Implement cache optimization
        pass

    async def _monitor_system(self, settings: Dict[str, Any]):
        """Monitor system resources and Ollama health"""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Store metrics
            self.metrics.record_metric(
                "system",
                {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent
                }
            )

            # Check Ollama health
            async with aiohttp.ClientSession() as session:
                ollama_interface = cast(OllamaInterface, self.llm_interface)
                start_time = datetime.now()
                async with session.get(f"{ollama_interface.ollama_settings.url}/") as response:
                    is_healthy = response.status == 200 and "Ollama is running" in await response.text()
                    response_time = (datetime.now() - start_time).total_seconds()

                    # Store Ollama metrics
                    self.metrics.record_metric(
                        "llm",
                        {
                            "status": is_healthy,
                            "response_time": response_time
                        }
                    )

                    if not is_healthy:
                        logger.warning("Ollama health check failed")
                        # Implement recovery logic here if needed

        except Exception as e:
            logger.error(f"Error checking Ollama health: {e}")
            self.metrics.record_error("system_monitoring")

    async def _build_knowledge_graph(self, settings: Dict[str, Any]):
        """Build semantic knowledge graph from memory segments"""
        # Implement knowledge graph construction
        pass

    async def _monitor_system_resources(self):
        """Continuously monitor system resources"""
        while self.running:
            try:
                settings = self.tasks_config["tasks"]["system_monitoring"]["settings"]
                await self._monitor_system(settings)
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

async def main():
    """Main function to run the background agent"""
    # Load configuration
    config_path = os.path.join("config", "background_agent.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    app_config = WitsV3Config.from_yaml("config.yaml")
    llm_interface = OllamaInterface(app_config)
    memory_manager = MemoryManager(app_config, llm_interface)

    # Create and start background agent
    agent = BackgroundAgent("background", app_config, llm_interface, memory_manager)
    await agent.start()

    try:
        # Keep the agent running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
