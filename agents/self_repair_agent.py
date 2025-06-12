"""
Self-Repair and Evolution Agent for WitsV3
Monitors system health, fixes issues, and evolves capabilities
"""

import asyncio
import json
import os
import traceback
import uuid
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
from typing import Dict, List, Optional, Any, AsyncGenerator, Set
from datetime import datetime, timedelta
from pathlib import Path

from agents.base_agent import BaseAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, ConversationHistory
from core.neural_web_core import NeuralWeb

# Import models and handlers
from agents.self_repair_models import SystemIssue, SystemMetrics, EvolutionSuggestion
from agents.self_repair_handlers import (
    handle_health_check,
    handle_issue_diagnosis,
    handle_system_repair,
    handle_performance_optimization,
    handle_capability_evolution,
    handle_failure_learning,
    handle_tool_monitoring,
    handle_general_maintenance
)
from agents.self_repair_utils import (
    suggest_capability_enhancement,
    suggest_integration,
    suggest_optimization,
    suggest_user_experience_improvement
)


class SelfRepairAgent(BaseAgent):
    """
    Agent responsible for:
    - System health monitoring
    - Issue detection and diagnosis
    - Automatic repair and recovery
    - Performance optimization
    - Capability evolution
    - Learning from failures
    """
    
    def __init__(
        self,
        agent_name: 
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: Optional[MemoryManager] = None,
        neural_web: Optional[NeuralWeb] = None,
        tool_registry: Optional[Any] = None,
        system_components: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_name, config, llm_interface, memory_manager)
        
        # Neural web integration for system intelligence
   f.neural_web = neural_web
        if self.neural_web:
            self.enable_system_monitoring = True
            self.logger.info("Neural web enabled for system monitoring")
            asyncio.create_task(self._initialize_system_patterns())
        else:
            self.enable_system_monitoring = False
        
        self.tool_registry = tool_registry
        self.system_componystem_components or {}
        
        # Monitoring state
        self.detected_issues: Dict[str, Syste= {}
        self.system_metrics: List[SystemMetrics] = []
        self.evolution_suggestions: Dict[str, EvolutionSuggestion] = {}
        
        # Configuration
        self.monitoring_interval = 60  # seconds
    asattr(config, 'agents'):
            agent_config = getattr(config, 'agents')
            if hasattr(agent_config, 'self_repair_agent'):
                repair_config = getattr(agent_config, 'self_repair_agent')
                if hasattr(repair_config, 'health_check_interval'):
                    self.monitoring_interval = repair_config.health_check_interval
        
        self.metrics_retention = 1000  # number of metric samples to keep
        selfx_enabled = True
        self.learning_enabled = True
        
        # Health thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
    'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,  # seconds
            'error_rate': 0.05,  # 5%
            'tool_failure_rate': 0.1  # 10%
        }
        
        # Map repair strategies to utility functions
        self.repair_strategies = {
            leak': 'fix_memory_leak',
            'high_cpu': 'fix_high_cpu',
            'disk_space': 'fix_disk_space',
            'configuration_error': 'fix_configuration_error',
            'tool_failure': 'fix_tool_failure',
            'performance_degradation': 'fix_performance_issue'
        }
        
        # Map evolution patterns to utility functions
        self.evolution_patterns = {
            'capabil: suggest_capability_enhancement,
            'integration_opportunity': suggest_integration,
            'optimization_opportunity': suggest_optimization,
            'user_pattern': suggest_user_experience_improvement
        }
        
        # Start continuous monitoring if enabled
        self.monitoring_task = None
        if self.enable_system_mon
            self.logger.info(f"Starting continuous system monitoring (interval: {self.monitoring_interval}s)")
            self.monitoring_task = asyncio.create_task(self.start_continuous_monitoring())
        
        # Track tool failures if tool registry is available
        self.tool_failure_counts = {}
        if self.tool_regis         self.logger.info("Tool registry monitoring enabled")
            self._setup_tool_monitoring()
        
        self.logger.info("Self-Repair Agent initialized")
    
    def _setup_tool_monitoring(self):
        """Set up monitoring  failures"""
        if not self.tool_registry:
            rn
        
        # Initialize failure counts for all tools
        for tool_name in self.tool_registry.tools:
            self.tool_failure_countsme] = 0
    
    def __del__(self):
        """Clean up resources when the agent is destroyed"""
        if self.monitoring_task and not self.monitoring_task.():
            self.monitoring_task.cancel()
    
    async def _initialize_system_patterns(self):
        """Initialize neural web with system patterns"""
        self.logger.info("Initializing systetterns")
        # Simplified implementation
        await asyncio.sleep(0.1)
    
    async def start_continuous_monitoring(self):
        """Start continuous system monitoring"""
        self.logger.info("Starting continuous monitoring")     while True:
            try:
                # Collect metrics
                metrics = await self._collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # Keep only recent metrics
                if len(self.system_metrics) > self.metrics_retention:
                    self.system_metrics trics[-self.metrics_retention:]
                
                # Wait for next check
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
              fo("Monitoring task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring task: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        
        try:
            # Get system metrics
        if PSUTIL_AVAILABLE and psutil:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = prtual_memory()
                try:
                    disk = psutil.disk_usage('/')
                    disk_usage = disk.percent
                except:
                    disk_usage = 50.0  # Fallback if path doesn't exist
                
                cpu_usage = cpu_percent
                memory_usage = memory.percent
            else:
                # Fallback values when psutil is not available
             0.0   # Simulated CPU usage
                memory_usage = 30.0  # Simulated memory usage
                disk_usage = 50.0    # Simulated disk usage
            
            # Application metrics
            response_time = 0.5  # Would measure actual response times
            error_rate = 0.01   # Would calculate from error logs
            uptime =  Would track actual uptime
            active_agents = len(self.system_components.get('agents', []))
            
            # Tool failures
            tool_failures = sum(self.tool_failure_counts.values()) if self.tool_registry else 0
            
            return SystemMetrics(
                cpu_usage=c              memory_usage=memory_usage,
                disk_usage=disk_usage,
                response_time=response_time,
    error_rate=error_rate,
                uptime=uptime,
                active_agents=active_agents,
                tool_failures=tool_failures
            )
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            # Return default metrics on error
            return SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                response_time=0.0,
                error_rate=0.0,
                uptime=0.0,
                active_agents=0,
                tool_failures=0
            )
    
    async def _analyze_maintenance_task(self, request: str) -> Dict[str, Any]:
        """Analyze the maintenance request to determine task type"""
        
        analysis_prompt = f"""
        Analyze this system maintenance ret:
        
        Request: {request}
        
        Respond with JSON containing:
        {{
            "task_type": "health_check" | "diagnose_issrepair_system" | "optimize_performance" | "evolve_capabilities" | "learn_from_failuronitor_tools" | "general_maintena          "urgency": "low" | "medium" | "high" | "critical",
            "scope": "component" | "system" | "global",
            "focus_areas": ["performance", "reliability", "security", "functionality"],
            "parameters": {{additional specific parameters}}
        }}
        """
        
        try:
            response = await self.generate_response(analysis_prompt, temperature=0.3)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loadatch.group(0))
        except Exception as e:
            self.logger.warning(f"Failed to parse maintenance task analysis: {e}")
        
        return {
            "task_type": "health_check",
            "urgency": "medium",
            "scope": "system",
            "focus_areas": ["performance", "reliability"],
            "parameters": {}
        }
    
    async def run(
        self  request: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Process self-repair evolution requests
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        yield self.stream_thinking("Analyzing system health request...")
        
        # Parse the request to understand what type of maintenance is needed
        task_analysis = await self._analyze_maintenance_task(request)
        
        yield self.stream_thinking(fied task: {task_analysis['task_type']}")
        
        # Route to approprdler
        if task_analysis['task_type'] == 'health_check':
            async for stream in handle_health_check(self, session_id):
                yield        
        elif task_analysis['task_type'] == 'diagnose_issues':
            asynceam in handle_issue_diagnosis(self, session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'repair_system':
            async for stream in handle_system_repair(se_analysis, session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'optimize_performance':
            async for stream in handle_pee_optimization(self, session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'evolve_capabilities':
            async for stream in handle_capavolution(self, session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'learn_from_failures':
            async for stream in handle_failure_learn, session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'monitor_tools':
            async for stream in handle_tool_monitoring(self, sessi                yield stream
        
        else:
            async for stream in handle_general_maintenance(self, task_analysis, session_id):
                yield stream
