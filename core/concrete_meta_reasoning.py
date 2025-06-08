"""
Concrete implementation of the Meta-Reasoning Framework for WitsV3

This module provides a concrete implementation of the meta-reasoning capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .meta_reasoning import (
    MetaReasoningEngine, ProblemSpace, ProblemComplexity, 
    ExecutionPlan, ExecutionStrategy, ExecutionStep, ExecutionMetrics
)
from .config import WitsV3Config
from .schemas import StreamData

logger = logging.getLogger(__name__)


class WitsV3MetaReasoningEngine(MetaReasoningEngine):
    """
    Concrete implementation of meta-reasoning for WitsV3.
    
    This engine analyzes problems, creates execution plans,
    monitors progress, and adapts strategies based on results.
    """
    
    def __init__(self, config: WitsV3Config):
        super().__init__(config)
        self.execution_history: Dict[str, ExecutionMetrics] = {}
        self.checkpoints: Dict[str, List[Any]] = {}
        
    async def analyze_problem_space(self, goal: str, context: Dict[str, Any]) -> ProblemSpace:
        """
        Analyze a problem to understand its nature and requirements.
        
        Uses heuristics and pattern matching to categorize problems.
        """
        await self.stream_thinking(f"Analyzing problem: {goal}")
        
        # Analyze complexity based on keywords and patterns
        complexity = await self._determine_complexity(goal, context)
        
        # Extract constraints from context
        constraints = context.get('constraints', [])
        if 'time_limit' in context:
            constraints.append(f"Complete within {context['time_limit']} seconds")
            
        # Determine required capabilities
        capabilities = await self._determine_capabilities(goal, complexity)
        
        # Define success criteria
        success_criteria = await self._define_success_criteria(goal, context)
        
        # Estimate steps needed
        estimated_steps = await self._estimate_steps(complexity, capabilities)
        
        # Calculate confidence
        confidence = await self._calculate_confidence(goal, context, capabilities)
        
        problem_space = ProblemSpace(
            goal=goal,
            complexity=complexity,
            constraints=constraints,
            required_capabilities=capabilities,
            success_criteria=success_criteria,
            estimated_steps=estimated_steps,
            confidence=confidence,
            metadata=context
        )
        
        await self.stream_action(
            f"Problem analyzed: {complexity.value} complexity, "
            f"{estimated_steps} steps, {confidence:.0%} confidence"
        )
        
        return problem_space
    
    async def generate_execution_plan(
        self, 
        problem_space: ProblemSpace,
        available_agents: List[str]
    ) -> ExecutionPlan:
        """
        Generate an execution plan based on problem analysis.
        """
        await self.stream_thinking("Generating execution plan...")
        
        # Select strategy based on complexity
        strategy = await self._select_strategy(problem_space, available_agents)
        
        # Generate steps based on strategy
        steps = await self._generate_steps(problem_space, strategy, available_agents)
        
        # Create contingency plans for complex problems
        contingencies = {}
        if problem_space.complexity in [ProblemComplexity.COMPLEX, ProblemComplexity.RESEARCH]:
            contingencies = await self._generate_contingencies(problem_space, available_agents)
        
        # Estimate duration
        estimated_duration = sum(step.timeout_seconds for step in steps)
        
        # Determine resource requirements
        resources = await self._estimate_resources(steps)
        
        plan = ExecutionPlan(
            problem_space=problem_space,
            strategy=strategy,
            steps=steps,
            contingency_plans=contingencies,
            estimated_duration=estimated_duration,
            resource_requirements=resources
        )
        
        await self.stream_action(
            f"Generated {strategy.value} plan with {len(steps)} steps"
        )
        
        return plan
    
    async def monitor_execution(
        self,
        plan: ExecutionPlan,
        real_time: bool = True
    ) -> ExecutionMetrics:
        """
        Monitor plan execution and gather metrics.
        """
        metrics = ExecutionMetrics(
            plan_id=plan.plan_id,
            total_steps=len(plan.steps)
        )
        
        if real_time:
            # In real-time mode, would hook into actual execution
            # For now, simulate monitoring
            await self.stream_thinking("Monitoring execution in real-time...")
            
        # Analyze execution history if available
        if plan.plan_id in self.execution_history:
            previous_metrics = self.execution_history[plan.plan_id]
            metrics.completed_steps = previous_metrics.completed_steps
            metrics.failed_steps = previous_metrics.failed_steps
            
        # Calculate success rate
        if metrics.total_steps > 0:
            metrics.success_rate = metrics.completed_steps / metrics.total_steps
            
        # Identify bottlenecks
        progress, bottlenecks = self.evaluate_progress(plan, [])
        metrics.bottlenecks = bottlenecks
        
        # Generate recommendations
        metrics.recommendations = await self._generate_recommendations(plan, metrics)
        
        # Store metrics
        self.execution_history[plan.plan_id] = metrics
        
        return metrics
    
    async def adapt_plan(
        self,
        original_plan: ExecutionPlan,
        metrics: ExecutionMetrics,
        failure_info: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Adapt the execution plan based on results.
        """
        await self.stream_thinking("Adapting plan based on execution results...")
        
        # Check if we should use a contingency plan
        if failure_info and original_plan.contingency_plans:
            failure_type = failure_info.get('type', 'unknown')
            if failure_type in original_plan.contingency_plans:
                await self.stream_action(f"Switching to contingency plan: {failure_type}")
                return original_plan.contingency_plans[failure_type]
        
        # Otherwise, create an adapted plan
        adapted_plan = ExecutionPlan(
            problem_space=original_plan.problem_space,
            strategy=original_plan.strategy,
            steps=original_plan.steps.copy()
        )
        
        # Adapt based on metrics
        if metrics.success_rate < 0.5:
            # Low success rate - try different strategy
            adapted_plan.strategy = await self._select_alternative_strategy(
                original_plan.strategy,
                original_plan.problem_space
            )
            await self.stream_action(f"Switching strategy to: {adapted_plan.strategy.value}")
            
        if metrics.bottlenecks:
            # Address bottlenecks
            await self._address_bottlenecks(adapted_plan, metrics.bottlenecks)
            
        if failure_info:
            # Add retry or alternative steps for failures
            await self._add_failure_recovery(adapted_plan, failure_info)
            
        return adapted_plan
    
    # Helper methods
    
    async def _determine_complexity(self, goal: str, context: Dict[str, Any]) -> ProblemComplexity:
        """Determine problem complexity based on analysis"""
        goal_lower = goal.lower()
        
        # Simple patterns
        if any(word in goal_lower for word in ['what', 'when', 'who', 'define']):
            return ProblemComplexity.SIMPLE
            
        # Research patterns
        if any(word in goal_lower for word in ['research', 'analyze', 'compare', 'investigate']):
            return ProblemComplexity.RESEARCH
            
        # Creative patterns
        if any(word in goal_lower for word in ['create', 'design', 'write', 'generate']):
            if 'complex' in goal_lower or 'advanced' in goal_lower:
                return ProblemComplexity.COMPLEX
            return ProblemComplexity.CREATIVE
            
        # Complex patterns
        if any(word in goal_lower for word in ['implement', 'build', 'develop', 'integrate']):
            return ProblemComplexity.COMPLEX
            
        # Default to moderate
        return ProblemComplexity.MODERATE
    
    async def _determine_capabilities(self, goal: str, complexity: ProblemComplexity) -> List[str]:
        """Determine required capabilities"""
        capabilities = []
        goal_lower = goal.lower()
        
        # Coding capabilities
        if any(word in goal_lower for word in ['code', 'program', 'script', 'function', 'implement']):
            capabilities.extend(['code_generation', 'debugging', 'testing'])
            
        # Research capabilities
        if any(word in goal_lower for word in ['research', 'find', 'search', 'investigate']):
            capabilities.append('web_search')
            
        # Writing capabilities
        if any(word in goal_lower for word in ['write', 'document', 'explain', 'describe']):
            capabilities.append('writing')
            
        # Analysis capabilities
        if any(word in goal_lower for word in ['analyze', 'evaluate', 'assess']):
            capabilities.append('analysis')
            
        # Add based on complexity
        if complexity in [ProblemComplexity.COMPLEX, ProblemComplexity.RESEARCH]:
            capabilities.append('planning')
            
        # Default capability
        if not capabilities:
            capabilities.append("general_processing")
            
        return list(set(capabilities))  # Remove duplicates
    
    async def _define_success_criteria(self, goal: str, context: Dict[str, Any]) -> List[str]:
        """Define success criteria for the problem"""
        criteria = []
        
        # User-defined criteria
        if 'success_criteria' in context:
            criteria.extend(context['success_criteria'])
            
        # Goal-based criteria
        if 'implement' in goal.lower() or 'build' in goal.lower():
            criteria.extend(['Working implementation', 'Passes tests', 'Documented'])
            
        if 'research' in goal.lower():
            criteria.extend(['Comprehensive findings', 'Credible sources', 'Clear conclusions'])
            
        # Default criteria
        if not criteria:
            criteria = ['Goal achieved', 'Quality output', 'Within constraints']
            
        return criteria
    
    async def _estimate_steps(self, complexity: ProblemComplexity, capabilities: List[str]) -> int:
        """Estimate number of steps needed"""
        base_steps = {
            ProblemComplexity.SIMPLE: 1,
            ProblemComplexity.MODERATE: 3,
            ProblemComplexity.COMPLEX: 5,
            ProblemComplexity.RESEARCH: 4,
            ProblemComplexity.CREATIVE: 3
        }
        
        steps = base_steps.get(complexity, 3)
        
        # Add steps for multiple capabilities
        steps += max(0, len(capabilities) - 2)
        
        return steps
    
    async def _calculate_confidence(self, goal: str, context: Dict[str, Any], capabilities: List[str]) -> float:
        """Calculate confidence in solving the problem"""
        confidence = 0.8  # Base confidence
        
        # Adjust based on context
        if 'previous_attempts' in context:
            # Lower confidence if previous attempts failed
            confidence -= 0.1 * context['previous_attempts']
            
        # Adjust based on capabilities
        if len(capabilities) > 3:
            # Complex multi-capability problems are harder
            confidence -= 0.1
            
        # Adjust based on constraints
        if 'constraints' in context and len(context['constraints']) > 2:
            confidence -= 0.1
            
        return max(0.1, min(1.0, confidence))
    
    async def _select_strategy(
        self, 
        problem_space: ProblemSpace,
        available_agents: List[str]
    ) -> ExecutionStrategy:
        """Select execution strategy based on problem analysis"""
        
        if problem_space.complexity == ProblemComplexity.SIMPLE:
            return ExecutionStrategy.DIRECT
            
        if problem_space.complexity == ProblemComplexity.RESEARCH:
            return ExecutionStrategy.EXPLORATORY
            
        if len(problem_space.required_capabilities) > 2:
            # Multiple capabilities suggest parallel execution
            return ExecutionStrategy.PARALLEL
            
        if problem_space.complexity == ProblemComplexity.CREATIVE:
            return ExecutionStrategy.ITERATIVE
            
        return ExecutionStrategy.SEQUENTIAL
    
    async def _generate_steps(
        self,
        problem_space: ProblemSpace,
        strategy: ExecutionStrategy,
        available_agents: List[str]
    ) -> List[ExecutionStep]:
        """Generate execution steps based on strategy"""
        steps = []
        
        # Map capabilities to agents
        capability_agent_map = {
            'code_generation': 'coding_agent',
            'web_search': 'research_agent',
            'writing': 'writing_agent',
            'analysis': 'analysis_agent',
            'testing': 'testing_agent',
            'debugging': 'debugging_agent'
        }
        
        # Generate steps for each required capability
        for i, capability in enumerate(problem_space.required_capabilities):
            agent_type = capability_agent_map.get(capability, 'general_agent')
            
            if agent_type in available_agents:
                step = ExecutionStep(
                    agent_type=agent_type,
                    action=f"handle_{capability}",
                    inputs={'goal': problem_space.goal, 'capability': capability},
                    expected_outputs=[f"{capability}_result"],
                    timeout_seconds=300 if problem_space.complexity == ProblemComplexity.SIMPLE else 600
                )
                
                # Set dependencies based on strategy
                if strategy == ExecutionStrategy.SEQUENTIAL and i > 0:
                    # Each step depends on the previous one
                    step.dependencies = [steps[-1].step_id]
                    
                steps.append(step)
                
        return steps
    
    async def _generate_contingencies(
        self,
        problem_space: ProblemSpace,
        available_agents: List[str]
    ) -> Dict[str, ExecutionPlan]:
        """Generate contingency plans for complex problems"""
        contingencies = {}
        
        # Timeout contingency - simpler approach
        if problem_space.complexity in [ProblemComplexity.COMPLEX, ProblemComplexity.RESEARCH]:
            simplified_space = ProblemSpace(
                goal=problem_space.goal,
                complexity=ProblemComplexity.MODERATE,
                constraints=problem_space.constraints[:1],  # Fewer constraints
                required_capabilities=problem_space.required_capabilities[:2],  # Fewer capabilities
                success_criteria=problem_space.success_criteria[:1]  # Lower bar
            )
            
            contingencies['timeout'] = await self.generate_execution_plan(
                simplified_space,
                available_agents
            )
            
        return contingencies
    
    async def _estimate_resources(self, steps: List[ExecutionStep]) -> Dict[str, Any]:
        """Estimate resource requirements"""
        return {
            'total_time_seconds': sum(step.timeout_seconds for step in steps),
            'agents_required': len(set(step.agent_type for step in steps)),
            'parallel_capacity': max(1, len([s for s in steps if not s.dependencies]))
        }
    
    async def _generate_recommendations(
        self,
        plan: ExecutionPlan,
        metrics: ExecutionMetrics
    ) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if metrics.success_rate < 0.5:
            recommendations.append("Consider breaking down the problem into smaller sub-problems")
            
        if metrics.bottlenecks:
            recommendations.append("Address identified bottlenecks to improve execution flow")
            
        if metrics.failed_steps > metrics.total_steps * 0.3:
            recommendations.append("Review agent capabilities and consider alternative approaches")
            
        return recommendations
    
    async def _select_alternative_strategy(
        self,
        current_strategy: ExecutionStrategy,
        problem_space: ProblemSpace
    ) -> ExecutionStrategy:
        """Select an alternative strategy when current one is failing"""
        
        # Strategy fallback chain
        fallbacks = {
            ExecutionStrategy.DIRECT: ExecutionStrategy.SEQUENTIAL,
            ExecutionStrategy.SEQUENTIAL: ExecutionStrategy.ITERATIVE,
            ExecutionStrategy.PARALLEL: ExecutionStrategy.SEQUENTIAL,
            ExecutionStrategy.ITERATIVE: ExecutionStrategy.EXPLORATORY,
            ExecutionStrategy.EXPLORATORY: ExecutionStrategy.SEQUENTIAL
        }
        
        return fallbacks.get(current_strategy, ExecutionStrategy.ITERATIVE)
    
    async def _address_bottlenecks(self, plan: ExecutionPlan, bottlenecks: List[str]) -> None:
        """Modify plan to address identified bottlenecks"""
        # For now, just log - in real implementation would modify steps
        for bottleneck in bottlenecks:
            self.logger.info(f"Addressing bottleneck: {bottleneck}")
            
    async def _add_failure_recovery(self, plan: ExecutionPlan, failure_info: Dict[str, Any]) -> None:
        """Add recovery steps for failures"""
        failed_step_id = failure_info.get('step_id')
        
        if failed_step_id:
            # Find the failed step
            for i, step in enumerate(plan.steps):
                if step.step_id == failed_step_id:
                    # Add a retry step
                    retry_step = ExecutionStep(
                        agent_type=step.agent_type,
                        action=f"retry_{step.action}",
                        inputs={**step.inputs, 'previous_error': failure_info.get('error')},
                        expected_outputs=step.expected_outputs,
                        dependencies=step.dependencies,
                        timeout_seconds=step.timeout_seconds * 1.5,  # Give more time
                        retry_count=1  # Only retry once
                    )
                    
                    # Insert after the failed step
                    plan.steps.insert(i + 1, retry_step)
                    break


# Test the implementation
async def test_adaptive_meta_reasoning():
    """Test the adaptive meta-reasoning engine"""
    engine = WitsV3MetaReasoningEngine()
    
    # Test problem analysis
    problem = "Create a web scraper to analyze product prices and generate a report"
    problem_space = await engine.analyze_problem_space(problem, {})
    print(f"Problem complexity: {problem_space.complexity.value}")
    print(f"Components: {problem_space.components}")
    print(f"Required capabilities: {problem_space.required_capabilities}")
    
    # Test execution planning
    plan = await engine.generate_execution_plan(problem_space, ["coding_agent", "research_agent"])
    print(f"\nExecution plan with {len(plan.steps)} steps:")
    for step in plan.steps:
        print(f"  - {step.step_id}: {step.action} ({step.agent_type})")
    
    # Test monitoring
    current_state = {
        "current_step": "step_2",
        "completed_steps": ["step_1"],
        "step_start_time": datetime.now().isoformat(),
        "elapsed_time": 120
    }
    
    monitoring_result = await engine.monitor_execution(plan, True)
    print(f"\nExecution progress: {monitoring_result.success_rate * 100:.1f}%")
    print(f"Bottlenecks: {len(monitoring_result.bottlenecks)}")
    
    # Test checkpointing
    checkpoint_id = await engine.checkpoint_state(current_state)
    restored_state = await engine.restore_from_checkpoint(checkpoint_id)
    print(f"\nCheckpoint created and verified: {restored_state == current_state}")


if __name__ == "__main__":
    asyncio.run(test_adaptive_meta_reasoning())