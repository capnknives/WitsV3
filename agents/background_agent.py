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
from aiohttp import web
from datetime import datetime, timedelta, timezone
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
            elif task_name == "self_repair":
                await self._run_self_repair(task_config["settings"])

            logger.info(f"Completed task: {task_name}")

        except Exception as e:
            logger.error(f"Error in task {task_name}: {e}")
            self.metrics.record_error(f"task_{task_name}")
        finally:
            if task_name in self.active_tasks:
                del self.active_tasks[task_name]

    async def _maintain_memory(self, settings: Dict[str, Any]):
        """Maintain memory by pruning old and low-importance segments."""
        if not self.memory_manager:
            return

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=settings["max_age_days"])
        segments = await self.memory_manager.get_recent_memory(
            limit=settings["batch_size"]
        )

        removed = 0
        for segment in segments:
            timestamp = segment.timestamp
            if timestamp is not None and timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            is_old = timestamp is not None and timestamp < cutoff_date
            is_low_importance = (segment.importance or 0.0) < settings["min_importance_threshold"]
            if is_old or is_low_importance:
                removed += await self.memory_manager.delete_segments({"id": segment.id})

        logger.info(
            "Memory maintenance: removed %d segment(s) (older than %dd or importance < %.2f)",
            removed, settings["max_age_days"], settings["min_importance_threshold"],
        )

    async def _optimize_semantic_cache(self, settings: Dict[str, Any]):
        """Not implemented — the semantic cache belongs to the deprecated
        adaptive-LLM stack (see planning/archive/adaptive_llm/README.md).
        Disabled by default in config/background_agent.yaml; logs rather
        than silently reporting success on a no-op."""
        logger.warning(
            "semantic_cache_optimization is not implemented (adaptive LLM stack "
            "is deprecated in favor of core/model_router.py) — skipping"
        )

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
        """Not implemented — knowledge graph construction belongs to the
        dormant neural-web stack (only active when memory_manager.backend:
        neural). Disabled by default in config/background_agent.yaml; logs
        rather than silently reporting success on a no-op."""
        logger.warning(
            "knowledge_graph_construction is not implemented (neural web stack "
            "is dormant unless memory_manager.backend: neural) — skipping"
        )

    async def _run_self_repair(self, settings: Dict[str, Any]):
        """Run the self-repair agent's autonomous scan-and-fix (Docker-deployment
        parity for the same job WitsV3System schedules in-process — see
        run.py's _run_scheduled_self_repair)."""
        if not self.config.self_repair.enabled:
            return
        from agents.self_repair_agent import SelfRepairAgent

        agent = SelfRepairAgent(
            agent_name="SystemDoctor",
            config=self.config,
            llm_interface=self.llm_interface,
            memory_manager=self.memory_manager,
            tool_registry=self.tool_registry,
        )
        results = []
        async for stream_data in agent.run(
            "Scan for recent errors and fix any that can be safely resolved."
        ):
            if stream_data.type in ("result", "observation"):
                results.append(f"[{stream_data.type}] {stream_data.content}")
        logger.info("Background self-repair scan finished:\n%s", "\n".join(results))

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

_BACKGROUND_AGENT_KEY: web.AppKey["BackgroundAgent"] = web.AppKey("background_agent", BackgroundAgent)


async def _health_handler(request):
    """Real /health endpoint for the docker-compose healthcheck — previously
    nothing listened on port 8000 at all, so the healthcheck failed forever."""
    agent: "BackgroundAgent" = request.app[_BACKGROUND_AGENT_KEY]
    return web.json_response({
        "status": "ok",
        "running": agent.running,
        "active_tasks": list(agent.active_tasks.keys()),
    })


async def _start_health_server(agent: "BackgroundAgent", host: str = "0.0.0.0", port: int = 8000):
    """Minimal aiohttp server — just enough for the compose healthcheck and
    a manual `curl http://host:8000/health`, not a full API."""
    app = web.Application()
    app[_BACKGROUND_AGENT_KEY] = agent
    app.router.add_get("/health", _health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(f"Background agent health endpoint listening on http://{host}:{port}/health")
    return runner


async def main():
    """Main function to run the background agent"""
    # Initialize components
    app_config = WitsV3Config.from_yaml("config.yaml")
    llm_interface = OllamaInterface(app_config)
    memory_manager = MemoryManager(app_config, llm_interface)
    await memory_manager.initialize()

    # A real ToolRegistry, wired the same way run.py wires it — previously
    # main() passed no tool_registry at all, so self_repair (if enabled here)
    # would silently degrade to a plain LLM passthrough with none of its
    # actual diagnose/fix/verify capability.
    tool_registry = ToolRegistry()
    for tool in tool_registry.tools.values():
        if hasattr(tool, "set_dependencies"):
            tool.set_dependencies(app_config, llm_interface, memory_manager, tool_registry=tool_registry)

    # Create and start background agent
    agent = BackgroundAgent("background", app_config, llm_interface, memory_manager, tool_registry=tool_registry)
    await agent.start()
    health_runner = await _start_health_server(agent)

    try:
        # Keep the agent running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await agent.stop()
        await health_runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
