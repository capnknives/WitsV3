"""
Meta-Reasoning Framework for WitsV3

This module provides the base classes and interfaces for meta-reasoning capabilities.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

from .config import WitsV3Config
from .schemas import StreamData

logger = logging.getLogger(__name__)


class ProblemComplexity(Enum):
    """Levels of problem complexity"""
    SIMPLE = "simple"          # Single step, clear solution
    MODERATE = "moderate"      # Multiple steps, standard approach
    COMPLEX = "complex"        # Multi-agent, requires planning
    RESEARCH = "research"      # Unknown solution, requires exploration
    CREATIVE = "creative"      # Creative/generative tasks


class ExecutionStrategy(Enum):
    """Strategies for plan execution"""
    SEQUENTIAL = "sequential"   # Execute steps one after another
    PARALLEL = "parallel"      # Execute steps simultaneously
    ADAPTIVE = "adaptive"      # Adapt strategy based on results
    ITERATIVE = "iterative"    # Repeat steps until success
    EXPLORATORY = "exploratory" # Try multiple approaches
    DIRECT = "direct"          # Direct execution for simple tasks


@dataclass
class ExecutionStep:
    """A single step in an execution plan"""
    step_id: str = field(default_factory=lambda: str(uuid4()))
    agent_name: str = ""
    agent_type: str = ""
    action: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: float = 60.0
    retry_count: int = 3
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class ProblemSpace:
    """Represents a problem to be solved"""
    goal: str
    complexity: ProblemComplexity
    constraints: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    estimated_steps: int = 1
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """A plan for executing a solution"""
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    problem_space: Optional[ProblemSpace] = None
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    steps: List[ExecutionStep] = field(default_factory=list)
    contingency_plans: Dict[str, 'ExecutionPlan'] = field(default_factory=dict)
    estimated_duration: float = 0.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionMetrics:
    """Metrics from plan execution"""
    plan_id: str
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    success_rate: float = 0.0
    execution_time: float = 0.0
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)


class MetaReasoningEngine(ABC):
    """
    Abstract base class for meta-reasoning engines.

    Meta-reasoning engines analyze problems, create execution plans,
    monitor progress, and adapt strategies based on results.
    """

    def __init__(self, config: WitsV3Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def analyze_problem_space(self, goal: str, context: Dict[str, Any]) -> ProblemSpace:
        """
        Analyze a problem to understand its nature and requirements.

        Args:
            goal: The goal to achieve
            context: Additional context about the problem

        Returns:
            ProblemSpace object with analysis results
        """
        pass

    @abstractmethod
    async def generate_execution_plan(
        self,
        problem_space: ProblemSpace,
        available_agents: List[str]
    ) -> ExecutionPlan:
        """
        Generate an execution plan based on problem analysis.

        Args:
            problem_space: The analyzed problem space
            available_agents: List of available agent names

        Returns:
            ExecutionPlan object
        """
        pass

    @abstractmethod
    async def monitor_execution(
        self,
        plan: ExecutionPlan,
        real_time: bool = True
    ) -> ExecutionMetrics:
        """
        Monitor plan execution and gather metrics.

        Args:
            plan: The execution plan to monitor
            real_time: Whether to monitor in real-time

        Returns:
            ExecutionMetrics object
        """
        pass

    @abstractmethod
    async def adapt_plan(
        self,
        original_plan: ExecutionPlan,
        metrics: ExecutionMetrics,
        failure_info: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Adapt the execution plan based on results.

        Args:
            original_plan: The original execution plan
            metrics: Execution metrics
            failure_info: Information about any failures

        Returns:
            Adapted ExecutionPlan
        """
        pass

    def evaluate_progress(
        self,
        plan: ExecutionPlan,
        completed_steps: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Evaluate progress and identify bottlenecks.

        Args:
            plan: The execution plan
            completed_steps: List of completed step IDs

        Returns:
            Tuple of (progress_percentage, bottlenecks)
        """
        if not plan.steps:
            return 0.0, []

        completed_count = len(completed_steps)
        total_count = len(plan.steps)
        progress = completed_count / total_count

        # Identify bottlenecks (simplified)
        bottlenecks = []
        for step in plan.steps:
            if step.step_id not in completed_steps:
                # Check if dependencies are met
                deps_met = all(dep in completed_steps for dep in step.dependencies)
                if deps_met:
                    bottlenecks.append(f"Step {step.action} ready but not executed")

        return progress, bottlenecks

    async def stream_thinking(self, content: str) -> None:
        """Stream thinking output"""
        # This would integrate with the streaming system
        self.logger.info(f"💭 {content}")

    async def stream_action(self, content: str) -> None:
        """Stream action output"""
        # This would integrate with the streaming system
        self.logger.info(f"🎯 {content}")

    async def stream_observation(self, content: str) -> None:
        """Stream observation output"""
        # This would integrate with the streaming system
        self.logger.info(f"👁️ {content}")
