"""
Meta-Reasoning Framework for WitsV3

This module implements meta-reasoning capabilities - the ability to reason about
reasoning, plan execution strategies, and monitor/adapt agent performance.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

from .schemas import StreamData
from .config import WitsV3Config

logger = logging.getLogger(__name__)


class ProblemComplexity(Enum):
    """Problem complexity levels"""
    SIMPLE = "simple"          # Single step, direct answer
    MODERATE = "moderate"      # Multiple steps, clear path
    COMPLEX = "complex"        # Many steps, unclear path
    RESEARCH = "research"      # Requires exploration
    CREATIVE = "creative"      # Open-ended, multiple solutions


class ExecutionStrategy(Enum):
    """Execution strategies for problem solving"""
    DIRECT = "direct"              # Single agent, direct execution
    SEQUENTIAL = "sequential"      # Multiple agents in sequence
    PARALLEL = "parallel"          # Multiple agents in parallel
    ITERATIVE = "iterative"        # Repeated refinement
    EXPLORATORY = "exploratory"    # Trial and error with backtracking


@dataclass
class ProblemSpace:
    """Represents the analyzed problem space"""
    problem_id: str = field(default_factory=lambda: str(uuid4()))
    goal: str = ""
    complexity: ProblemComplexity = ProblemComplexity.SIMPLE
    constraints: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    estimated_steps: int = 1
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass 
class ExecutionStep:
    """Single step in an execution plan"""
    step_id: str = field(default_factory=lambda: str(uuid4()))
    agent_type: str = ""
    action: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite steps
    optional: bool = False
    timeout_seconds: int = 300
    retry_count: int = 3


@dataclass
class ExecutionPlan:
    """Complete execution plan for solving a problem"""
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    problem_space: ProblemSpace = field(default_factory=ProblemSpace)
    strategy: ExecutionStrategy = ExecutionStrategy.DIRECT
    steps: List[ExecutionStep] = field(default_factory=list)
    contingency_plans: Dict[str, 'ExecutionPlan'] = field(default_factory=dict)
    estimated_duration: float = 0.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_executable_steps(self, completed_steps: List[str]) -> List[ExecutionStep]:
        """Get steps that are ready to execute based on completed dependencies"""
        ready_steps = []
        for step in self.steps:
            if step.step_id not in completed_steps:
                # Check if all dependencies are satisfied
                if all(dep_id in completed_steps for dep_id in step.dependencies):
                    ready_steps.append(step)
        return ready_steps


@dataclass
class ExecutionMetrics:
    """Metrics for monitoring execution progress"""
    plan_id: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    execution_time: float = 0.0
    success_rate: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class StateCheckpoint:
    """Checkpoint for execution state"""
    checkpoint_id: str = field(default_factory=lambda: str(uuid4()))
    plan_id: str = ""
    completed_steps: List[str] = field(default_factory=list)
    agent_states: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class MetaReasoningEngine(ABC):
    """
    Abstract base class for meta-reasoning engines.
    
    Meta-reasoning involves:
    1. Analyzing problems to understand their nature
    2. Planning execution strategies
    3. Monitoring execution progress
    4. Adapting plans based on results
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
            Analyzed problem space
        """
        pass
    
    @abstractmethod
    async def generate_execution_plan(
        self, 
        problem_space: ProblemSpace,
        available_agents: List[str]
    ) -> ExecutionPlan:
        """
        Generate an execution plan for the problem.
        
        Args:
            problem_space: The analyzed problem
            available_agents: List of available agent types
            
        Returns:
            Execution plan with steps and contingencies
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
            plan: The execution plan being monitored
            real_time: Whether to monitor in real-time
            
        Returns:
            Execution metrics
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
            metrics: Current execution metrics
            failure_info: Information about any failures
            
        Returns:
            Adapted execution plan
        """
        pass
    
    async def checkpoint_state(
        self,
        plan_id: str,
        completed_steps: List[str],
        agent_states: Dict[str, Any],
        context: Dict[str, Any]
    ) -> StateCheckpoint:
        """
        Create a checkpoint of the current execution state.
        
        Args:
            plan_id: ID of the execution plan
            completed_steps: List of completed step IDs
            agent_states: Current state of all agents
            context: Execution context
            
        Returns:
            State checkpoint
        """
        checkpoint = StateCheckpoint(
            plan_id=plan_id,
            completed_steps=completed_steps.copy(),
            agent_states=agent_states.copy(),
            context=context.copy()
        )
        
        self.logger.info(f"Created checkpoint {checkpoint.checkpoint_id} for plan {plan_id}")
        return checkpoint
    
    def evaluate_progress(
        self,
        plan: ExecutionPlan,
        completed_steps: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Evaluate progress on the execution plan.
        
        Args:
            plan: The execution plan
            completed_steps: List of completed step IDs
            
        Returns:
            Tuple of (progress percentage, bottlenecks)
        """
        if not plan.steps:
            return 100.0, []
        
        progress = (len(completed_steps) / len(plan.steps)) * 100
        
        # Identify bottlenecks
        bottlenecks = []
        ready_steps = plan.get_executable_steps(completed_steps)
        
        if not ready_steps and progress < 100:
            # Find steps blocking progress
            for step in plan.steps:
                if step.step_id not in completed_steps:
                    blocking_deps = [
                        dep for dep in step.dependencies 
                        if dep not in completed_steps
                    ]
                    if blocking_deps:
                        bottlenecks.append(
                            f"Step {step.step_id} blocked by: {blocking_deps}"
                        )
        
        return progress, bottlenecks
    
    async def stream_thinking(
        self,
        thought: str,
        source: str = "MetaReasoning"
    ) -> StreamData:
        """Stream a thinking/reasoning update"""
        return StreamData(
            type="thinking",
            content=thought,
            source=source,
            metadata={"engine": self.__class__.__name__}
        )
    
    async def stream_action(
        self,
        action: str,
        source: str = "MetaReasoning"
    ) -> StreamData:
        """Stream an action update"""
        return StreamData(
            type="action",
            content=action,
            source=source,
            metadata={"engine": self.__class__.__name__}
        )


# Test function
async def test_meta_reasoning():
    """Test the meta-reasoning framework"""
    from .config import load_config
    
    print("Testing Meta-Reasoning Framework...")
    
    # Create test problem space
    problem = ProblemSpace(
        goal="Create a Python web scraper",
        complexity=ProblemComplexity.MODERATE,
        constraints=["Must handle rate limiting", "Must be async"],
        required_capabilities=["code_generation", "web_knowledge", "testing"],
        success_criteria=["Working code", "Handles errors", "Has tests"]
    )
    
    print(f"✓ Created problem space: {problem.goal}")
    print(f"  Complexity: {problem.complexity.value}")
    print(f"  Capabilities: {problem.required_capabilities}")
    
    # Create test execution plan
    plan = ExecutionPlan(
        problem_space=problem,
        strategy=ExecutionStrategy.SEQUENTIAL,
        steps=[
            ExecutionStep(
                agent_type="research_agent",
                action="research_web_scraping_best_practices",
                expected_outputs=["best_practices", "libraries"]
            ),
            ExecutionStep(
                agent_type="coding_agent",
                action="generate_scraper_code",
                inputs={"requirements": problem.constraints},
                dependencies=[],  # Would reference first step ID
                expected_outputs=["scraper.py"]
            ),
            ExecutionStep(
                agent_type="testing_agent",
                action="generate_tests",
                dependencies=[],  # Would reference second step ID
                expected_outputs=["test_scraper.py"]
            )
        ]
    )
    
    print(f"✓ Created execution plan with {len(plan.steps)} steps")
    print(f"  Strategy: {plan.strategy.value}")
    
    # Test getting executable steps
    ready = plan.get_executable_steps([])
    print(f"✓ Ready steps with no completions: {len(ready)}")
    
    # Create test checkpoint
    checkpoint = StateCheckpoint(
        plan_id=plan.plan_id,
        completed_steps=["step1"],
        agent_states={"coding_agent": {"status": "ready"}},
        context={"iteration": 1}
    )
    
    print(f"✓ Created checkpoint: {checkpoint.checkpoint_id}")
    
    print("\nMeta-Reasoning Framework tests completed! 🎉")


if __name__ == "__main__":
    asyncio.run(test_meta_reasoning())