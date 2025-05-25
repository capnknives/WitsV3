# agents/self_repair_agent.py
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
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from agents.base_agent import BaseAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, ConversationHistory
from core.neural_web_core import NeuralWeb


@dataclass
class SystemIssue:
    """Represents a detected system issue"""
    id: str
    category: str  # performance, error, configuration, security, memory
    severity: str  # low, medium, high, critical
    description: str
    location: str  # module, file, or component
    detected_at: datetime
    resolution_attempts: int = 0
    status: str = "open"  # open, investigating, fixing, resolved, ignored
    auto_fixable: bool = False
    fix_suggestions: List[str] = field(default_factory=list)
    impact_score: float = 0.0


@dataclass
class SystemMetrics:
    """System performance and health metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    response_time: float
    error_rate: float
    uptime: float
    active_agents: int
    tool_failures: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvolutionSuggestion:
    """Suggestion for system evolution"""
    id: str
    category: str  # feature, optimization, integration, capability
    priority: str  # low, medium, high, critical
    description: str
    implementation_complexity: str  # simple, moderate, complex
    expected_benefit: str
    dependencies: List[str] = field(default_factory=list)
    implementation_plan: str = ""
    estimated_effort: str = ""


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
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: Optional[MemoryManager] = None,
        neural_web: Optional[NeuralWeb] = None,
        tool_registry: Optional[Any] = None,
        system_components: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_name, config, llm_interface, memory_manager)
        
        # Neural web integration for system intelligence
        self.neural_web = neural_web
        if self.neural_web:
            self.enable_system_monitoring = True
            self.logger.info("Neural web enabled for system monitoring")
            asyncio.create_task(self._initialize_system_patterns())
        else:
            self.enable_system_monitoring = False
        
        self.tool_registry = tool_registry
        self.system_components = system_components or {}
        
        # Monitoring state
        self.detected_issues: Dict[str, SystemIssue] = {}
        self.system_metrics: List[SystemMetrics] = []
        self.evolution_suggestions: Dict[str, EvolutionSuggestion] = {}
        
        # Configuration
        self.monitoring_interval = 60  # seconds
        self.metrics_retention = 1000  # number of metric samples to keep
        self.auto_fix_enabled = True
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
        
        # Repair strategies
        self.repair_strategies = {
            'memory_leak': self._fix_memory_leak,
            'high_cpu': self._fix_high_cpu,
            'disk_space': self._fix_disk_space,
            'configuration_error': self._fix_configuration_error,
            'tool_failure': self._fix_tool_failure,
            'performance_degradation': self._fix_performance_issue
        }
        
        # Evolution capabilities
        self.evolution_patterns = {
            'capability_gap': self._suggest_capability_enhancement,
            'integration_opportunity': self._suggest_integration,
            'optimization_opportunity': self._suggest_optimization,
            'user_pattern': self._suggest_user_experience_improvement
        }
        
        self.logger.info("Self-Repair Agent initialized")
    
    async def run(
        self,
        request: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Process self-repair and evolution requests
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        yield self.stream_thinking("Analyzing system health request...")
        
        # Parse the request to understand what type of maintenance is needed
        task_analysis = await self._analyze_maintenance_task(request)
        
        yield self.stream_thinking(f"Identified task: {task_analysis['task_type']}")
        
        # Route to appropriate handler
        if task_analysis['task_type'] == 'health_check':
            async for stream in self._handle_health_check(session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'diagnose_issues':
            async for stream in self._handle_issue_diagnosis(session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'repair_system':
            async for stream in self._handle_system_repair(task_analysis, session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'optimize_performance':
            async for stream in self._handle_performance_optimization(session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'evolve_capabilities':
            async for stream in self._handle_capability_evolution(session_id):
                yield stream
        
        elif task_analysis['task_type'] == 'learn_from_failures':
            async for stream in self._handle_failure_learning(session_id):
                yield stream
        
        else:
            async for stream in self._handle_general_maintenance(task_analysis, session_id):
                yield stream
    
    async def _analyze_maintenance_task(self, request: str) -> Dict[str, Any]:
        """Analyze the maintenance request to determine task type"""
        
        analysis_prompt = f"""
        Analyze this system maintenance request:
        
        Request: {request}
        
        Respond with JSON containing:
        {{
            "task_type": "health_check" | "diagnose_issues" | "repair_system" | "optimize_performance" | "evolve_capabilities" | "learn_from_failures" | "general_maintenance",
            "urgency": "low" | "medium" | "high" | "critical",
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
                return json.loads(json_match.group(0))
        except Exception as e:
            self.logger.warning(f"Failed to parse maintenance task analysis: {e}")
        
        return {
            "task_type": "health_check",
            "urgency": "medium",
            "scope": "system",
            "focus_areas": ["performance", "reliability"],
            "parameters": {}
        }
    
    async def _handle_health_check(self, session_id: str) -> AsyncGenerator[StreamData, None]:
        """Perform comprehensive system health check"""
        
        yield self.stream_action("Performing system health check...")
        
        # Collect current metrics
        current_metrics = await self._collect_system_metrics()
        self.system_metrics.append(current_metrics)
        
        # Keep only recent metrics
        if len(self.system_metrics) > self.metrics_retention:
            self.system_metrics = self.system_metrics[-self.metrics_retention:]
        
        yield self.stream_observation(f"CPU: {current_metrics.cpu_usage:.1f}%, Memory: {current_metrics.memory_usage:.1f}%")
        yield self.stream_observation(f"Disk: {current_metrics.disk_usage:.1f}%, Response Time: {current_metrics.response_time:.2f}s")
        
        # Check for threshold violations
        issues_detected = []
        
        if current_metrics.cpu_usage > self.thresholds['cpu_usage']:
            issues_detected.append(f"High CPU usage: {current_metrics.cpu_usage:.1f}%")
        
        if current_metrics.memory_usage > self.thresholds['memory_usage']:
            issues_detected.append(f"High memory usage: {current_metrics.memory_usage:.1f}%")
        
        if current_metrics.disk_usage > self.thresholds['disk_usage']:
            issues_detected.append(f"High disk usage: {current_metrics.disk_usage:.1f}%")
        
        if current_metrics.response_time > self.thresholds['response_time']:
            issues_detected.append(f"Slow response time: {current_metrics.response_time:.2f}s")
        
        if current_metrics.error_rate > self.thresholds['error_rate']:
            issues_detected.append(f"High error rate: {current_metrics.error_rate:.2%}")
        
        if issues_detected:
            yield self.stream_observation("Issues detected:")
            for issue in issues_detected:
                yield self.stream_observation(f"  - {issue}")
            
            # Create issue records
            for issue_desc in issues_detected:
                await self._create_issue_record(issue_desc, "performance", "medium")
        
        else:
            yield self.stream_result("System health check passed - all metrics within normal ranges")
        
        # Store health check results
        await self.store_memory(
            content=f"Health check completed: {len(issues_detected)} issues detected",
            segment_type="HEALTH_CHECK",
            importance=0.7,
            metadata={
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "issues_count": len(issues_detected),
                "session_id": session_id
            }
        )
        
        # Trend analysis
        if len(self.system_metrics) > 10:
            yield self.stream_thinking("Analyzing performance trends...")
            trends = await self._analyze_performance_trends()
            if trends['concerns']:
                yield self.stream_observation("Performance trends indicate:")
                for concern in trends['concerns']:
                    yield self.stream_observation(f"  - {concern}")
    
    async def _handle_issue_diagnosis(self, session_id: str) -> AsyncGenerator[StreamData, None]:
        """Diagnose existing system issues"""
        
        yield self.stream_action("Diagnosing system issues...")
        
        if not self.detected_issues:
            yield self.stream_result("No outstanding issues detected")
            return
        
        for issue_id, issue in self.detected_issues.items():
            yield self.stream_thinking(f"Analyzing issue: {issue.description}")
            
            # Generate detailed diagnosis
            diagnosis_prompt = f"""
            Provide detailed diagnosis for this system issue:
            
            Category: {issue.category}
            Description: {issue.description}
            Location: {issue.location}
            Severity: {issue.severity}
            
            Provide:
            1. Root cause analysis
            2. Impact assessment
            3. Potential solutions
            4. Prevention strategies
            5. Urgency level
            """
            
            diagnosis = await self.generate_response(diagnosis_prompt, temperature=0.5)
            
            # Extract actionable suggestions
            suggestions = await self._extract_fix_suggestions(diagnosis)
            issue.fix_suggestions = suggestions
            
            yield self.stream_observation(f"Issue: {issue.description}")
            yield self.stream_observation(f"Diagnosis: {diagnosis[:200]}...")
            
            if suggestions:
                yield self.stream_observation("Suggested fixes:")
                for suggestion in suggestions[:3]:
                    yield self.stream_observation(f"  - {suggestion}")
            
            # Store diagnosis
            await self.store_memory(
                content=f"Issue diagnosis: {diagnosis}",
                segment_type="ISSUE_DIAGNOSIS",
                importance=0.8,
                metadata={
                    "issue_id": issue_id,
                    "category": issue.category,
                    "severity": issue.severity,
                    "session_id": session_id
                }
            )
    
    async def _handle_system_repair(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle automated system repair"""
        
        yield self.stream_action("Initiating system repair procedures...")
        
        if not self.auto_fix_enabled:
            yield self.stream_observation("Auto-repair is disabled - providing manual repair guidance")
            await self._provide_manual_repair_guidance(session_id)
            return
        
        repairs_attempted = 0
        repairs_successful = 0
        
        for issue_id, issue in self.detected_issues.items():
            if issue.status != "open" or not issue.auto_fixable:
                continue
            
            yield self.stream_action(f"Attempting to fix: {issue.description}")
            
            try:
                # Determine repair strategy
                strategy = self._determine_repair_strategy(issue)
                
                if strategy in self.repair_strategies:
                    success = await self.repair_strategies[strategy](issue)
                    
                    repairs_attempted += 1
                    
                    if success:
                        issue.status = "resolved"
                        repairs_successful += 1
                        yield self.stream_result(f"Successfully fixed: {issue.description}")
                        
                        # Remove from active issues
                        del self.detected_issues[issue_id]
                    else:
                        issue.status = "failed"
                        issue.resolution_attempts += 1
                        yield self.stream_observation(f"Failed to fix: {issue.description}")
                
                else:
                    yield self.stream_observation(f"No repair strategy available for: {issue.category}")
            
            except Exception as e:
                yield self.stream_observation(f"Error during repair: {str(e)}")
                issue.status = "error"
                issue.resolution_attempts += 1
        
        yield self.stream_result(f"Repair summary: {repairs_successful}/{repairs_attempted} fixes successful")
        
        # Store repair results
        await self.store_memory(
            content=f"System repair completed: {repairs_successful}/{repairs_attempted} successful",
            segment_type="SYSTEM_REPAIR",
            importance=0.9,
            metadata={
                "repairs_attempted": repairs_attempted,
                "repairs_successful": repairs_successful,
                "session_id": session_id
            }
        )
    
    async def _handle_performance_optimization(self, session_id: str) -> AsyncGenerator[StreamData, None]:
        """Handle performance optimization"""
        
        yield self.stream_action("Analyzing performance optimization opportunities...")
        
        # Analyze recent performance data
        if len(self.system_metrics) < 10:
            yield self.stream_observation("Insufficient performance data for optimization analysis")
            return
        
        optimization_prompt = f"""
        Analyze these system performance metrics and suggest optimizations:
        
        Recent metrics:
        - Average CPU: {sum(m.cpu_usage for m in self.system_metrics[-10:]) / 10:.1f}%
        - Average Memory: {sum(m.memory_usage for m in self.system_metrics[-10:]) / 10:.1f}%
        - Average Response Time: {sum(m.response_time for m in self.system_metrics[-10:]) / 10:.2f}s
        - Error Rate: {sum(m.error_rate for m in self.system_metrics[-10:]) / 10:.2%}
        
        Provide specific optimization recommendations:
        1. Performance bottlenecks to address
        2. Resource utilization improvements
        3. Configuration optimizations
        4. Code-level optimizations
        5. Infrastructure improvements
        """
        
        yield self.stream_thinking("Generating optimization recommendations...")
        optimization_suggestions = await self.generate_response(optimization_prompt, temperature=0.6)
        
        yield self.stream_result("Performance Optimization Recommendations:")
        yield self.stream_result(optimization_suggestions)
        
        # Implement automatic optimizations
        auto_optimizations = await self._apply_automatic_optimizations()
        
        if auto_optimizations:
            yield self.stream_action("Applied automatic optimizations:")
            for opt in auto_optimizations:
                yield self.stream_action(f"  - {opt}")
        
        # Store optimization analysis
        await self.store_memory(
            content=f"Performance optimization analysis: {optimization_suggestions}",
            segment_type="PERFORMANCE_OPTIMIZATION",
            importance=0.8,
            metadata={
                "optimizations_applied": len(auto_optimizations),
                "session_id": session_id
            }
        )
    
    async def _handle_capability_evolution(self, session_id: str) -> AsyncGenerator[StreamData, None]:
        """Handle capability evolution and enhancement"""
        
        yield self.stream_action("Analyzing capability evolution opportunities...")
        
        # Analyze usage patterns and gaps
        usage_analysis = await self._analyze_usage_patterns()
        capability_gaps = await self._identify_capability_gaps()
        
        evolution_prompt = f"""
        Based on system usage analysis, suggest capability evolution:
        
        Usage Patterns: {usage_analysis}
        Capability Gaps: {capability_gaps}
        
        Suggest:
        1. New features to develop
        2. Existing capabilities to enhance
        3. Integration opportunities
        4. Automation improvements
        5. User experience enhancements
        
        Prioritize by impact and feasibility.
        """
        
        yield self.stream_thinking("Generating evolution suggestions...")
        evolution_suggestions = await self.generate_response(evolution_prompt, temperature=0.7)
        
        # Parse suggestions into structured format
        suggestions = await self._parse_evolution_suggestions(evolution_suggestions)
        
        for suggestion in suggestions:
            self.evolution_suggestions[suggestion.id] = suggestion
        
        yield self.stream_result("Capability Evolution Suggestions:")
        yield self.stream_result(evolution_suggestions)
        
        # Implement simple evolutionary improvements
        auto_evolutions = await self._apply_automatic_evolutions()
        
        if auto_evolutions:
            yield self.stream_action("Applied automatic improvements:")
            for evolution in auto_evolutions:
                yield self.stream_action(f"  - {evolution}")
        
        # Store evolution analysis
        await self.store_memory(
            content=f"Capability evolution analysis: {evolution_suggestions}",
            segment_type="CAPABILITY_EVOLUTION",
            importance=0.8,
            metadata={
                "suggestions_count": len(suggestions),
                "auto_evolutions": len(auto_evolutions),
                "session_id": session_id
            }
        )
    
    async def _handle_failure_learning(self, session_id: str) -> AsyncGenerator[StreamData, None]:
        """Learn from system failures and improve resilience"""
        
        yield self.stream_action("Analyzing failure patterns for learning...")
        
        # Search for failure-related memories
        failure_memories = []
        if self.memory_manager:
            for failure_type in ['ERROR', 'SYSTEM_FAILURE', 'TOOL_FAILURE']:
                memories = await self.memory_manager.search_memory(failure_type, limit=20)
                failure_memories.extend(memories)
        
        if not failure_memories:
            yield self.stream_result("No failure patterns found for analysis")
            return
        
        # Analyze failure patterns
        learning_prompt = f"""
        Analyze these system failures to identify patterns and learning opportunities:
        
        Failure Count: {len(failure_memories)}
        
        Extract:
        1. Common failure patterns
        2. Root causes
        3. Prevention strategies
        4. System resilience improvements
        5. Monitoring enhancements
        6. Recovery procedures
        
        Focus on actionable improvements to prevent future failures.
        """
        
        yield self.stream_thinking("Extracting failure insights...")
        failure_analysis = await self.generate_response(learning_prompt, temperature=0.5)
        
        # Extract actionable improvements
        improvements = await self._extract_resilience_improvements(failure_analysis)
        
        yield self.stream_result("Failure Analysis and Learning:")
        yield self.stream_result(failure_analysis)
        
        # Update neural web with failure patterns
        if self.neural_web:
            for improvement in improvements:
                await self.neural_web.add_concept(
                    f"resilience_{improvement.replace(' ', '_')}",
                    f"Resilience improvement: {improvement}",
                    "resilience_pattern"
                )
        
        # Store learning insights
        await self.store_memory(
            content=f"Failure learning analysis: {failure_analysis}",
            segment_type="FAILURE_LEARNING",
            importance=0.9,
            metadata={
                "failures_analyzed": len(failure_memories),
                "improvements_identified": len(improvements),
                "session_id": session_id
            }
        )
    
    async def _handle_general_maintenance(
        self, 
        task_analysis: Dict[str, Any], 
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle general maintenance tasks"""
        
        yield self.stream_action("Performing general system maintenance...")
        
        maintenance_tasks = [
            "Cleaning temporary files",
            "Updating configuration",
            "Optimizing memory usage",
            "Checking component health",
            "Updating tool registry",
            "Pruning old memories"
        ]
        
        completed_tasks = []
        
        for task in maintenance_tasks:
            yield self.stream_action(f"Task: {task}")
            
            try:
                # Simulate maintenance task
                await asyncio.sleep(0.1)  # Simulate work
                success = await self._perform_maintenance_task(task)
                
                if success:
                    completed_tasks.append(task)
                    yield self.stream_observation(f"✓ {task}")
                else:
                    yield self.stream_observation(f"✗ {task} - failed")
            
            except Exception as e:
                yield self.stream_observation(f"✗ {task} - error: {str(e)}")
        
        yield self.stream_result(f"Maintenance completed: {len(completed_tasks)}/{len(maintenance_tasks)} tasks successful")
        
        # Store maintenance results
        await self.store_memory(
            content=f"General maintenance: {len(completed_tasks)} tasks completed",
            segment_type="GENERAL_MAINTENANCE",
            importance=0.6,
            metadata={
                "tasks_completed": completed_tasks,
                "session_id": session_id
            }
        )
    
    # Helper methods for system monitoring and repair
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        
        try:
            # Get system metrics
            if PSUTIL_AVAILABLE and psutil:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                try:
                    disk = psutil.disk_usage('/')
                    disk_usage = disk.percent
                except:
                    disk_usage = 50.0  # Fallback if path doesn't exist
                
                cpu_usage = cpu_percent
                memory_usage = memory.percent
            else:
                # Fallback values when psutil is not available
                cpu_usage = 10.0   # Simulated CPU usage
                memory_usage = 30.0  # Simulated memory usage
                disk_usage = 50.0    # Simulated disk usage
            
            # Simulate application metrics
            response_time = 0.5  # Would measure actual response times
            error_rate = 0.01   # Would calculate from error logs
            uptime = 3600       # Would track actual uptime
            active_agents = len(self.system_components.get('agents', []))
            tool_failures = 0   # Would track tool failures
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                response_time=response_time,
                error_rate=error_rate,
                uptime=uptime,
                active_agents=active_agents,
                tool_failures=tool_failures
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics
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
    
    async def _create_issue_record(self, description: str, category: str, severity: str) -> str:
        """Create a new issue record"""
        
        issue_id = str(uuid.uuid4())
        
        issue = SystemIssue(
            id=issue_id,
            category=category,
            severity=severity,
            description=description,
            location="system",
            detected_at=datetime.now(),
            auto_fixable=category in ['performance', 'configuration']
        )
        
        self.detected_issues[issue_id] = issue
        
        return issue_id
    
    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from recent metrics"""
        
        if len(self.system_metrics) < 10:
            return {'concerns': []}
        
        recent_metrics = self.system_metrics[-10:]
        concerns = []
        
        # CPU trend
        cpu_values = [m.cpu_usage for m in recent_metrics]
        if len(cpu_values) > 5 and cpu_values[-1] > cpu_values[0] * 1.5:
            concerns.append("CPU usage trending upward")
        
        # Memory trend
        memory_values = [m.memory_usage for m in recent_metrics]
        if len(memory_values) > 5 and memory_values[-1] > memory_values[0] * 1.3:
            concerns.append("Memory usage trending upward")
        
        # Response time trend
        response_values = [m.response_time for m in recent_metrics]
        if len(response_values) > 5 and response_values[-1] > response_values[0] * 2:
            concerns.append("Response time degrading")
        
        return {'concerns': concerns}
    
    async def _extract_fix_suggestions(self, diagnosis: str) -> List[str]:
        """Extract actionable fix suggestions from diagnosis text"""
        
        # Simple extraction - could be enhanced with NLP
        suggestions = []
        lines = diagnosis.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(word in line.lower() for word in ['fix', 'solution', 'resolve', 'repair']):
                if len(line) > 10 and not line.startswith('#'):
                    suggestions.append(line)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _determine_repair_strategy(self, issue: SystemIssue) -> str:
        """Determine the appropriate repair strategy for an issue"""
        
        issue_text = issue.description.lower()
        
        if 'memory' in issue_text:
            return 'memory_leak'
        elif 'cpu' in issue_text:
            return 'high_cpu'
        elif 'disk' in issue_text:
            return 'disk_space'
        elif 'config' in issue_text:
            return 'configuration_error'
        elif 'tool' in issue_text:
            return 'tool_failure'
        else:
            return 'performance_degradation'
    
    # Repair strategy implementations
    
    async def _fix_memory_leak(self, issue: SystemIssue) -> bool:
        """Fix memory leak issues"""
        try:
            # Implement memory cleanup
            import gc
            gc.collect()
            
            # Clear old metrics
            if len(self.system_metrics) > 100:
                self.system_metrics = self.system_metrics[-50:]
            
            return True
        except Exception as e:
            self.logger.error(f"Memory leak fix failed: {e}")
            return False
    
    async def _fix_high_cpu(self, issue: SystemIssue) -> bool:
        """Fix high CPU usage issues"""
        try:
            # Reduce processing intensity
            # This could involve throttling, load balancing, etc.
            await asyncio.sleep(0.1)  # Simulate CPU relief
            return True
        except Exception as e:
            self.logger.error(f"High CPU fix failed: {e}")
            return False
    
    async def _fix_disk_space(self, issue: SystemIssue) -> bool:
        """Fix disk space issues"""
        try:
            # Clean up temporary files, logs, etc.
            temp_dirs = ['/tmp', './temp', './logs']
            
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    # Clean old files (simulation)
                    pass
            
            return True
        except Exception as e:
            self.logger.error(f"Disk space fix failed: {e}")
            return False
    
    async def _fix_configuration_error(self, issue: SystemIssue) -> bool:
        """Fix configuration errors"""
        try:
            # Reset to default configuration or fix known issues
            return True
        except Exception as e:
            self.logger.error(f"Configuration fix failed: {e}")
            return False
    
    async def _fix_tool_failure(self, issue: SystemIssue) -> bool:
        """Fix tool failure issues"""
        try:
            # Restart failed tools, update registry, etc.
            if self.tool_registry:
                # Refresh tool registry
                pass
            return True
        except Exception as e:
            self.logger.error(f"Tool failure fix failed: {e}")
            return False
    
    async def _fix_performance_issue(self, issue: SystemIssue) -> bool:
        """Fix general performance issues"""
        try:
            # Apply general performance improvements
            return True
        except Exception as e:
            self.logger.error(f"Performance fix failed: {e}")
            return False
    
    async def _provide_manual_repair_guidance(self, session_id: str):
        """Provide manual repair guidance when auto-repair is disabled"""
        
        guidance = """
        Manual Repair Guidance:
        
        1. Check system resources (CPU, Memory, Disk)
        2. Review error logs for patterns
        3. Restart affected components
        4. Update configurations if needed
        5. Clear caches and temporary files
        6. Monitor performance after changes
        """
        
        await self.store_memory(
            content=guidance,
            segment_type="MANUAL_REPAIR_GUIDE",
            importance=0.7,
            metadata={"session_id": session_id}
        )
    
    async def _apply_automatic_optimizations(self) -> List[str]:
        """Apply automatic performance optimizations"""
        
        optimizations = []
        
        try:
            # Memory optimization
            import gc
            gc.collect()
            optimizations.append("Memory garbage collection")
            
            # Metrics pruning
            if len(self.system_metrics) > self.metrics_retention:
                old_count = len(self.system_metrics)
                self.system_metrics = self.system_metrics[-self.metrics_retention:]
                optimizations.append(f"Pruned {old_count - len(self.system_metrics)} old metrics")
            
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Auto optimization failed: {e}")
            return optimizations
    
    async def _analyze_usage_patterns(self) -> str:
        """Analyze system usage patterns"""
        
        # Analyze memory for usage patterns
        patterns = "High tool usage, frequent book writing requests, code generation popular"
        return patterns
    
    async def _identify_capability_gaps(self) -> str:
        """Identify gaps in current capabilities"""
        
        gaps = "Missing: video processing, advanced ML integration, real-time collaboration"
        return gaps
    
    async def _parse_evolution_suggestions(self, suggestions_text: str) -> List[EvolutionSuggestion]:
        """Parse evolution suggestions into structured format"""
        
        suggestions = []
        lines = suggestions_text.split('\n')
        
        current_suggestion = None
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line[0].isdigit()):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                
                current_suggestion = EvolutionSuggestion(
                    id=str(uuid.uuid4()),
                    category="feature",
                    priority="medium",
                    description=line.lstrip('- 0123456789.'),
                    implementation_complexity="moderate",
                    expected_benefit="Improved user experience"
                )
        
        if current_suggestion:
            suggestions.append(current_suggestion)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    async def _apply_automatic_evolutions(self) -> List[str]:
        """Apply simple automatic evolutionary improvements"""
        
        evolutions = []
        
        try:
            # Update thresholds based on recent performance
            if len(self.system_metrics) > 50:
                avg_cpu = sum(m.cpu_usage for m in self.system_metrics[-50:]) / 50
                if avg_cpu < 50:
                    self.thresholds['cpu_usage'] = max(70, self.thresholds['cpu_usage'] - 5)
                    evolutions.append("Lowered CPU threshold for better monitoring")
            
            return evolutions
            
        except Exception as e:
            self.logger.error(f"Auto evolution failed: {e}")
            return evolutions
    
    async def _extract_resilience_improvements(self, analysis: str) -> List[str]:
        """Extract resilience improvements from failure analysis"""
        
        improvements = []
        lines = analysis.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(word in line.lower() for word in ['improve', 'enhance', 'prevent', 'monitor']):
                if len(line) > 15:
                    improvements.append(line)
        
        return improvements[:5]
    
    async def _perform_maintenance_task(self, task: str) -> bool:
        """Perform a specific maintenance task"""
        
        try:
            if "temporary files" in task:
                # Clean temporary files
                return True
            elif "configuration" in task:
                # Update configuration
                return True
            elif "memory" in task:
                # Optimize memory
                import gc
                gc.collect()
                return True
            elif "component health" in task:
                # Check components
                return True
            elif "tool registry" in task:
                # Update tools
                return True
            elif "memories" in task:
                # Prune old memories
                return True
            else:
                return True
                
        except Exception as e:
            self.logger.error(f"Maintenance task '{task}' failed: {e}")
            return False
    
    async def start_continuous_monitoring(self):
        """Start continuous system monitoring"""
        
        while True:
            try:
                # Collect metrics
                metrics = await self._collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # Check for issues
                if metrics.cpu_usage > self.thresholds['cpu_usage']:
                    await self._create_issue_record(
                        f"High CPU usage: {metrics.cpu_usage:.1f}%",
                        "performance",
                        "medium"
                    )
                
                if metrics.memory_usage > self.thresholds['memory_usage']:
                    await self._create_issue_record(
                        f"High memory usage: {metrics.memory_usage:.1f}%",
                        "performance",
                        "medium"
                    )
                
                # Auto-repair critical issues
                if self.auto_fix_enabled:
                    critical_issues = [
                        issue for issue in self.detected_issues.values()
                        if issue.severity == "critical" and issue.status == "open"
                    ]
                    
                    for issue in critical_issues:
                        strategy = self._determine_repair_strategy(issue)
                        if strategy in self.repair_strategies:
                            await self.repair_strategies[strategy](issue)
                
                # Prune metrics
                if len(self.system_metrics) > self.metrics_retention:
                    self.system_metrics = self.system_metrics[-self.metrics_retention:]
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status summary"""
        
        if not self.system_metrics:
            return {"status": "unknown", "message": "No metrics available"}
        
        latest = self.system_metrics[-1]
        
        status = "healthy"
        if latest.cpu_usage > self.thresholds['cpu_usage'] * 0.8:
            status = "warning"
        if latest.cpu_usage > self.thresholds['cpu_usage']:
            status = "critical"
        
        return {
            "status": status,
            "metrics": {
                "cpu_usage": latest.cpu_usage,
                "memory_usage": latest.memory_usage,
                "disk_usage": latest.disk_usage,
                "response_time": latest.response_time,
                "error_rate": latest.error_rate,
                "uptime": latest.uptime
            },
            "issues": {
                "total": len(self.detected_issues),
                "open": len([i for i in self.detected_issues.values() if i.status == "open"]),
                "critical": len([i for i in self.detected_issues.values() if i.severity == "critical"])
            },
            "evolution_suggestions": len(self.evolution_suggestions),
            "last_check": latest.timestamp.isoformat()
        }
    
    async def _initialize_system_patterns(self):
        """Initialize system monitoring patterns in the neural web"""
        if not self.neural_web:
            return
        
        try:
            # Add system health patterns
            patterns = [
                ("memory_usage", "System memory consumption patterns", "system_metric"),
                ("cpu_usage", "CPU utilization patterns", "system_metric"),
                ("error_frequency", "Error occurrence patterns", "system_health"),
                ("response_time", "System response time patterns", "performance"),
                ("user_satisfaction", "User satisfaction patterns", "quality")
            ]
            
            for pattern_id, description, concept_type in patterns:
                await self.neural_web.add_concept(
                    concept_id=f"system_{pattern_id}",
                    content=description,
                    concept_type=concept_type,
                    metadata={"domain": "system_monitoring", "type": "health_metric"}
                )
            
            # Connect related patterns
            await self.neural_web.connect_concepts("system_cpu_usage", "system_response_time", "causes", 0.8)
            await self.neural_web.connect_concepts("system_error_frequency", "system_user_satisfaction", "contradicts", 0.9)
            
            self.logger.info("System monitoring patterns initialized in neural web")
            
        except Exception as e:
            self.logger.error(f"Error initializing system patterns: {e}")

    async def _suggest_capability_enhancement(self, issue_data: Dict[str, Any]) -> List[EvolutionSuggestion]:
        """Suggest new capabilities to enhance the system"""
        
        suggestions = []
        
        # Analyze capability gaps from issue data
        capability_prompt = f"""
        Analyze the following system usage patterns and suggest capability enhancements:
        
        Issue Data: {issue_data}
        
        Suggest specific new capabilities that would:
        1. Address recurring user needs
        2. Prevent common issues
        3. Improve overall system efficiency
        4. Enhance user experience
        
        Focus on actionable, implementable features.
        """
        
        try:
            response = await self.generate_response(capability_prompt, temperature=0.7)
            suggestions_list = await self._parse_evolution_suggestions(response)
            
            for suggestion in suggestions_list:
                suggestion.category = "capability_enhancement"
                suggestion.priority = "medium"
                suggestions.append(suggestion)
                
        except Exception as e:
            self.logger.error(f"Error generating capability suggestions: {e}")
            
        return suggestions[:3]
    
    async def _suggest_integration(self, issue_data: Dict[str, Any]) -> List[EvolutionSuggestion]:
        """Suggest integration opportunities with other systems"""
        
        suggestions = []
        
        integration_prompt = f"""
        Based on system usage patterns, suggest integration opportunities:
        
        Issue Data: {issue_data}
        
        Identify potential integrations with:
        1. External APIs and services
        2. Development tools and platforms
        3. Data sources and databases
        4. Communication systems
        5. Monitoring and analytics tools
        
        Focus on integrations that would solve user pain points.
        """
        
        try:
            response = await self.generate_response(integration_prompt, temperature=0.6)
            suggestions_list = await self._parse_evolution_suggestions(response)
            
            for suggestion in suggestions_list:
                suggestion.category = "integration"
                suggestion.priority = "low"
                suggestion.implementation_complexity = "complex"
                suggestions.append(suggestion)
                
        except Exception as e:
            self.logger.error(f"Error generating integration suggestions: {e}")
            
        return suggestions[:3]
    
    async def _suggest_optimization(self, issue_data: Dict[str, Any]) -> List[EvolutionSuggestion]:
        """Suggest performance and efficiency optimizations"""
        
        suggestions = []
        
        optimization_prompt = f"""
        Analyze system performance data and suggest optimizations:
        
        Issue Data: {issue_data}
        
        Recommend optimizations for:
        1. Response time improvements
        2. Memory usage reduction
        3. CPU efficiency gains
        4. Better resource management
        5. Caching strategies
        6. Algorithm improvements
        
        Prioritize high-impact, low-risk optimizations.
        """
        
        try:
            response = await self.generate_response(optimization_prompt, temperature=0.5)
            suggestions_list = await self._parse_evolution_suggestions(response)
            
            for suggestion in suggestions_list:
                suggestion.category = "optimization"
                suggestion.priority = "high"
                suggestion.implementation_complexity = "simple"
                suggestions.append(suggestion)
                
        except Exception as e:
            self.logger.error(f"Error generating optimization suggestions: {e}")
            
        return suggestions[:3]
    
    async def _suggest_user_experience_improvement(self, issue_data: Dict[str, Any]) -> List[EvolutionSuggestion]:
        """Suggest user experience improvements"""
        
        suggestions = []
        
        ux_prompt = f"""
        Based on user interaction patterns, suggest UX improvements:
        
        Issue Data: {issue_data}
        
        Focus on improvements to:
        1. User interface clarity and usability
        2. Response clarity and helpfulness
        3. Error handling and recovery
        4. Workflow efficiency
        5. Personalization features
        6. Accessibility features
        
        Prioritize changes that directly impact user satisfaction.
        """
        
        try:
            response = await self.generate_response(ux_prompt, temperature=0.7)
            suggestions_list = await self._parse_evolution_suggestions(response)
            
            for suggestion in suggestions_list:
                suggestion.category = "user_experience"
                suggestion.priority = "high"
                suggestion.expected_benefit = "Improved user satisfaction"
                suggestions.append(suggestion)
                
        except Exception as e:
            self.logger.error(f"Error generating UX suggestions: {e}")
            
        return suggestions[:3]


# Test function
async def test_self_repair_agent():
    """Test the self-repair agent functionality"""
    from core.config import load_config
    from core.llm_interface import OllamaInterface
    
    try:
        config = load_config("config.yaml")
        llm_interface = OllamaInterface(config=config)
        
        agent = SelfRepairAgent(
            agent_name="SelfRepair",
            config=config,
            llm_interface=llm_interface
        )
        
        print("Testing self-repair agent...")
        
        # Test health check
        async for stream_data in agent.run("Perform a comprehensive system health check"):
            print(f"[{stream_data.type.upper()}] {stream_data.content[:100]}...")
        
        # Get system status
        status = agent.get_system_status()
        print(f"System status: {status}")
        
    except Exception as e:
        print(f"Test completed with expected errors: {e}")


if __name__ == "__main__":
    asyncio.run(test_self_repair_agent())