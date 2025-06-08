"""
Tests for the background agent
"""

import os
import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from agents.background_agent import BackgroundAgent
from core.config import WitsV3Config
from core.llm_interface import OllamaInterface
from core.memory_manager import MemoryManager
from core.tool_registry import ToolRegistry

@pytest.fixture
async def config():
    """Create a test configuration"""
    return WitsV3Config.from_yaml()

@pytest.fixture
async def llm_interface(config):
    """Create a mock LLM interface"""
    return Mock(spec=OllamaInterface)

@pytest.fixture
async def memory_manager(config, llm_interface):
    """Create a mock memory manager"""
    return Mock(spec=MemoryManager)

@pytest.fixture
async def tool_registry():
    """Create a mock tool registry"""
    return Mock(spec=ToolRegistry)

@pytest.fixture
async def background_agent(config, llm_interface, memory_manager, tool_registry):
    """Create a background agent instance"""
    return BackgroundAgent(
        agent_name="TestBackgroundAgent",
        config=config,
        llm_interface=llm_interface,
        memory_manager=memory_manager,
        tool_registry=tool_registry
    )

@pytest.mark.asyncio
async def test_agent_initialization(background_agent):
    """Test that the agent initializes correctly"""
    assert background_agent.agent_name == "TestBackgroundAgent"
    assert background_agent.scheduler is not None
    assert hasattr(background_agent.metrics, 'metrics')
    assert isinstance(background_agent.metrics.metrics, dict)
    assert "system" in background_agent.metrics.metrics
    assert "llm" in background_agent.metrics.metrics
    assert "errors" in background_agent.metrics.metrics

@pytest.mark.asyncio
async def test_task_execution(background_agent):
    """Test that tasks can be executed"""
    # Mock the task execution
    background_agent._execute_task = Mock()

    # Run a test task
    async for stream in background_agent.run("test_task"):
        assert stream.type in ["thinking", "result", "error"]

    # Verify task was executed
    background_agent._execute_task.assert_called_once()

@pytest.mark.asyncio
async def test_system_monitoring(background_agent):
    """Test system monitoring functionality"""
    # Mock psutil functions
    with patch("psutil.cpu_percent", return_value=50.0), \
         patch("psutil.virtual_memory", return_value=Mock(percent=60.0)), \
         patch("psutil.disk_usage", return_value=Mock(percent=70.0)):

        # Run system monitoring
        await background_agent._monitor_system({
            "ollama_health_check_interval": 300,
            "resource_alert_threshold": {
                "cpu_percent": 80,
                "memory_percent": 85,
                "disk_percent": 90
            }
        })

        # Verify metrics were recorded
        assert len(background_agent.metrics.metrics["system"]) > 0
        latest_metrics = background_agent.metrics.metrics["system"][-1]
        assert latest_metrics["cpu_percent"] == 50.0
        assert latest_metrics["memory_percent"] == 60.0
        assert latest_metrics["disk_percent"] == 70.0

@pytest.mark.asyncio
async def test_memory_maintenance(background_agent, memory_manager):
    """Test memory maintenance functionality"""
    # Mock memory manager
    memory_manager.get_recent_memory.return_value = [
        Mock(
            timestamp=datetime.now() - timedelta(days=31),
            importance=0.2
        ),
        Mock(
            timestamp=datetime.now() - timedelta(days=15),
            importance=0.8
        )
    ]

    # Run memory maintenance
    await background_agent._maintain_memory({
        "max_age_days": 30,
        "min_importance_threshold": 0.3,
        "batch_size": 100
    })

    # Verify memory manager was called
    memory_manager.get_recent_memory.assert_called_once_with(limit=100)

@pytest.mark.asyncio
async def test_agent_start_stop(background_agent):
    """Test agent start and stop functionality"""
    # Start the agent
    await background_agent.start()
    assert background_agent.scheduler.running

    # Stop the agent
    await background_agent.stop()
    assert not background_agent.scheduler.running
    assert len(background_agent.active_tasks) == 0
