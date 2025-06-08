"""
Concrete implementation of meta-reasoning engine for WitsV3
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from .meta_reasoning import (
    MetaReasoningEngine, ProblemSpace, ExecutionPlan, 
    ExecutionStep, BottleneckInfo, StateCheckpoint
)


@dataclass
class ReasoningNode:
    """Represents a node in the reasoning graph"""
    id: str
    description: str
    status: str = "pending"  # pending, active, completed, failed
    dependencies: List[str] = field(default_factory=list)
    results: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class AdaptiveMetaReasoningEngine(MetaReasoningEngine):
    """
    Adaptive meta-reasoning engine that learns from experience
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.reasoning_graph: Dict[str, ReasoningNode] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.pattern_library: Dict[str, ExecutionPlan] = {}
        self.checkpoint_store: Dict[str, StateCheckpoint] = {}
        
    async def analyze_problem(self, problem_description: str) -> ProblemSpace:
        """
        Analyze a problem to understand its structure and requirements
        """
        self.logger.info(f"Analyzing problem: {problem_description[:100]}...")
        
        # Decompose problem into components
        components = await self._decompose_problem(problem_description)
        
        # Identify constraints
        constraints = await self._identify_constraints(problem_description, components)
        
        # Find similar problems from history
        similar_problems = await self._find_similar_problems(problem_description)
        
        # Estimate complexity
        complexity_score = await self._estimate_complexity(
            components, constraints, similar_problems
        )
        
        # Determine required capabilities
        required_capabilities = await self._identify_required_capabilities(
            components, problem_description
        )
        
        problem_space = ProblemSpace(
            description=problem_description,
            components=components,
            constraints=constraints,
            complexity_score=complexity_score,
            required_capabilities=required_capabilities,
            metadata={
                "similar_problems": similar_problems,
                "analysis_timestamp": datetime.now().isoformat()
            }
        )
        
        self.logger.info(f"Problem analysis complete. Complexity: {complexity_score}")
        return problem_space
    
    async def create_execution_plan(self, problem_space: ProblemSpace) -> ExecutionPlan:
        """
        Create an execution plan based on problem analysis
        """
        self.logger.info("Creating execution plan...")
        
        # Check if we have a similar pattern
        cached_plan = await self._check_pattern_library(problem_space)
        if cached_plan:
            self.logger.info("Found similar execution pattern in library")
            return await self._adapt_cached_plan(cached_plan, problem_space)
        
        # Create new plan
        steps = await self._generate_execution_steps(problem_space)
        
        # Identify decision points
        decision_points = await self._identify_decision_points(steps, problem_space)
        
        # Create fallback strategies
        fallback_strategies = await self._create_fallback_strategies(
            steps, problem_space.constraints
        )
        
        # Estimate resources
        estimated_resources = await self._estimate_resources(steps)
        
        plan = ExecutionPlan(
            plan_id=f"plan_{datetime.now().timestamp()}",
            steps=steps,
            decision_points=decision_points,
            fallback_strategies=fallback_strategies,
            estimated_time=estimated_resources.get("time", 0),
            estimated_resources=estimated_resources
        )
        
        # Store in pattern library for future use
        await self._store_in_pattern_library(problem_space, plan)
        
        return plan
    
    async def monitor_execution(
        self, 
        plan: ExecutionPlan, 
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Monitor ongoing execution and provide guidance
        """
        # Track execution progress
        progress = await self._calculate_progress(plan, current_state)
        
        # Detect bottlenecks
        bottlenecks = await self._detect_bottlenecks(plan, current_state)
        
        # Check if replanning needed
        needs_replanning = await self._check_replanning_triggers(
            plan, current_state, bottlenecks
        )
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            plan, current_state, bottlenecks
        )
        
        monitoring_result = {
            "progress": progress,
            "bottlenecks": bottlenecks,
            "needs_replanning": needs_replanning,
            "recommendations": recommendations,
            "current_step": current_state.get("current_step"),
            "elapsed_time": current_state.get("elapsed_time", 0)
        }
        
        # Update execution history
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "plan_id": plan.plan_id,
            "monitoring_result": monitoring_result
        })
        
        return monitoring_result
    
    async def identify_bottlenecks(
        self, 
        execution_state: Dict[str, Any]
    ) -> List[BottleneckInfo]:
        """
        Identify bottlenecks in the execution
        """
        bottlenecks = []
        
        # Check for stuck steps
        stuck_steps = await self._find_stuck_steps(execution_state)
        for step_id, duration in stuck_steps:
            bottlenecks.append(BottleneckInfo(
                bottleneck_type="stuck_step",
                severity=min(duration / 60, 1.0),  # Severity based on minutes stuck
                location=step_id,
                description=f"Step {step_id} has been running for {duration}s",
                suggested_action="Consider timeout or alternative approach"
            ))
        
        # Check for resource contention
        resource_issues = await self._check_resource_contention(execution_state)
        for resource, contention_level in resource_issues.items():
            if contention_level > 0.7:
                bottlenecks.append(BottleneckInfo(
                    bottleneck_type="resource_contention",
                    severity=contention_level,
                    location=resource,
                    description=f"High contention for {resource}",
                    suggested_action=f"Scale {resource} or optimize usage"
                ))
        
        # Check for circular dependencies
        circular_deps = await self._detect_circular_dependencies(execution_state)
        if circular_deps:
            bottlenecks.append(BottleneckInfo(
                bottleneck_type="circular_dependency",
                severity=0.9,
                location=str(circular_deps),
                description="Circular dependency detected",
                suggested_action="Break dependency cycle"
            ))
        
        return sorted(bottlenecks, key=lambda b: b.severity, reverse=True)
    
    async def checkpoint_state(self, state: Dict[str, Any]) -> str:
        """
        Create a checkpoint of the current state
        """
        checkpoint_id = f"checkpoint_{datetime.now().timestamp()}"
        
        checkpoint = StateCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            state=state.copy(),
            metadata={
                "reasoning_graph_size": len(self.reasoning_graph),
                "active_nodes": sum(1 for n in self.reasoning_graph.values() 
                                  if n.status == "active")
            }
        )
        
        self.checkpoint_store[checkpoint_id] = checkpoint
        self.logger.info(f"Created checkpoint: {checkpoint_id}")
        
        return checkpoint_id
    
    async def restore_from_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Restore state from a checkpoint
        """
        checkpoint = self.checkpoint_store.get(checkpoint_id)
        if not checkpoint:
            self.logger.error(f"Checkpoint {checkpoint_id} not found")
            return None
        
        self.logger.info(f"Restoring from checkpoint: {checkpoint_id}")
        return checkpoint.state.copy()
    
    # Helper methods
    
    async def _decompose_problem(self, description: str) -> List[str]:
        """Decompose problem into components"""
        # Simple heuristic decomposition
        components = []
        
        # Look for action verbs
        action_verbs = ["create", "analyze", "generate", "optimize", "search", "find"]
        for verb in action_verbs:
            if verb in description.lower():
                components.append(f"{verb}_task")
        
        # Look for domains
        domains = ["code", "data", "text", "image", "api", "database"]
        for domain in domains:
            if domain in description.lower():
                components.append(f"{domain}_component")
        
        # If no components found, add generic
        if not components:
            components.append("generic_task")
        
        return components
    
    async def _identify_constraints(
        self, 
        description: str, 
        components: List[str]
    ) -> List[str]:
        """Identify constraints from problem description"""
        constraints = []
        
        # Time constraints
        time_keywords = ["quickly", "fast", "immediate", "urgent", "asap"]
        if any(keyword in description.lower() for keyword in time_keywords):
            constraints.append("time_critical")
        
        # Resource constraints
        if "limited" in description.lower() or "constraint" in description.lower():
            constraints.append("resource_limited")
        
        # Quality constraints
        quality_keywords = ["accurate", "precise", "perfect", "exact"]
        if any(keyword in description.lower() for keyword in quality_keywords):
            constraints.append("high_quality_required")
        
        return constraints
    
    async def _find_similar_problems(self, description: str) -> List[Dict[str, Any]]:
        """Find similar problems from execution history"""
        similar = []
        
        # Simple keyword matching for now
        keywords = set(description.lower().split())
        
        for hist_item in self.execution_history[-100:]:  # Last 100 executions
            if "problem_description" in hist_item:
                hist_keywords = set(hist_item["problem_description"].lower().split())
                similarity = len(keywords & hist_keywords) / len(keywords | hist_keywords)
                
                if similarity > 0.3:
                    similar.append({
                        "description": hist_item["problem_description"],
                        "similarity": similarity,
                        "success": hist_item.get("success", False)
                    })
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)[:5]
    
    async def _estimate_complexity(
        self,
        components: List[str],
        constraints: List[str],
        similar_problems: List[Dict[str, Any]]
    ) -> float:
        """Estimate problem complexity on 0-1 scale"""
        base_complexity = len(components) * 0.1
        constraint_complexity = len(constraints) * 0.15
        
        # Adjust based on similar problems
        if similar_problems:
            avg_success = sum(p.get("success", False) for p in similar_problems) / len(similar_problems)
            history_complexity = 1.0 - avg_success
        else:
            history_complexity = 0.5  # Unknown
        
        total_complexity = (base_complexity + constraint_complexity + history_complexity) / 3
        
        return min(total_complexity, 1.0)
    
    async def _identify_required_capabilities(
        self,
        components: List[str],
        description: str
    ) -> List[str]:
        """Identify required agent capabilities"""
        capabilities = set()
        
        # Map components to capabilities
        capability_map = {
            "create_task": ["generation", "creativity"],
            "analyze_task": ["analysis", "reasoning"],
            "optimize_task": ["optimization", "evaluation"],
            "code_component": ["coding", "debugging"],
            "data_component": ["data_processing", "statistics"],
            "text_component": ["natural_language", "writing"]
        }
        
        for component in components:
            if component in capability_map:
                capabilities.update(capability_map[component])
        
        # Add generic capabilities
        capabilities.add("planning")
        capabilities.add("execution")
        
        return list(capabilities)
    
    async def _generate_execution_steps(
        self,
        problem_space: ProblemSpace
    ) -> List[ExecutionStep]:
        """Generate execution steps from problem space"""
        steps = []
        step_counter = 1
        
        # Initial planning step
        steps.append(ExecutionStep(
            step_id=f"step_{step_counter}",
            action="plan_approach",
            agent_type="orchestrator",
            parameters={
                "problem": problem_space.description,
                "constraints": problem_space.constraints
            },
            timeout=60
        ))
        step_counter += 1
        
        # Add steps for each component
        for component in problem_space.components:
            if "create" in component:
                steps.append(ExecutionStep(
                    step_id=f"step_{step_counter}",
                    action="generate_content",
                    agent_type="creative",
                    parameters={"component": component},
                    dependencies=[f"step_{step_counter - 1}"],
                    timeout=300
                ))
            elif "analyze" in component:
                steps.append(ExecutionStep(
                    step_id=f"step_{step_counter}",
                    action="analyze_data",
                    agent_type="analytical",
                    parameters={"component": component},
                    dependencies=[f"step_{step_counter - 1}"],
                    timeout=180
                ))
            else:
                steps.append(ExecutionStep(
                    step_id=f"step_{step_counter}",
                    action="process_component",
                    agent_type="general",
                    parameters={"component": component},
                    dependencies=[f"step_{step_counter - 1}"],
                    timeout=120
                ))
            step_counter += 1
        
        # Final validation step
        steps.append(ExecutionStep(
            step_id=f"step_{step_counter}",
            action="validate_results",
            agent_type="evaluator",
            parameters={"criteria": problem_space.constraints},
            dependencies=[f"step_{i}" for i in range(2, step_counter)],
            timeout=60
        ))
        
        return steps
    
    async def _calculate_progress(
        self,
        plan: ExecutionPlan,
        current_state: Dict[str, Any]
    ) -> float:
        """Calculate execution progress as percentage"""
        completed_steps = current_state.get("completed_steps", [])
        total_steps = len(plan.steps)
        
        if total_steps == 0:
            return 100.0
        
        return (len(completed_steps) / total_steps) * 100
    
    async def _detect_bottlenecks(
        self,
        plan: ExecutionPlan,
        current_state: Dict[str, Any]
    ) -> List[BottleneckInfo]:
        """Detect bottlenecks in execution"""
        bottlenecks = []
        
        current_step_id = current_state.get("current_step")
        if not current_step_id:
            return bottlenecks
        
        # Find current step
        current_step = None
        for step in plan.steps:
            if step.step_id == current_step_id:
                current_step = step
                break
        
        if not current_step:
            return bottlenecks
        
        # Check if step is taking too long
        step_start_time = current_state.get("step_start_time")
        if step_start_time:
            elapsed = (datetime.now() - datetime.fromisoformat(step_start_time)).seconds
            if elapsed > current_step.timeout * 0.8:  # 80% of timeout
                bottlenecks.append(BottleneckInfo(
                    bottleneck_type="timeout_risk",
                    severity=min(elapsed / current_step.timeout, 1.0),
                    location=current_step_id,
                    description=f"Step approaching timeout ({elapsed}s / {current_step.timeout}s)",
                    suggested_action="Consider interrupting or optimizing"
                ))
        
        return bottlenecks
    
    async def _check_replanning_triggers(
        self,
        plan: ExecutionPlan,
        current_state: Dict[str, Any],
        bottlenecks: List[BottleneckInfo]
    ) -> bool:
        """Check if replanning is needed"""
        # Replan if severe bottlenecks
        if any(b.severity > 0.8 for b in bottlenecks):
            return True
        
        # Replan if too many failures
        failed_steps = current_state.get("failed_steps", [])
        if len(failed_steps) > len(plan.steps) * 0.3:  # 30% failure rate
            return True
        
        # Replan if constraints violated
        constraint_violations = current_state.get("constraint_violations", [])
        if constraint_violations:
            return True
        
        return False
    
    async def _generate_recommendations(
        self,
        plan: ExecutionPlan,
        current_state: Dict[str, Any],
        bottlenecks: List[BottleneckInfo]
    ) -> List[str]:
        """Generate recommendations for execution"""
        recommendations = []
        
        # Recommendations based on bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck.suggested_action:
                recommendations.append(bottleneck.suggested_action)
        
        # Progress-based recommendations
        progress = await self._calculate_progress(plan, current_state)
        if progress < 50 and current_state.get("elapsed_time", 0) > plan.estimated_time * 0.7:
            recommendations.append("Consider parallelizing remaining steps")
        
        # Resource-based recommendations
        resource_usage = current_state.get("resource_usage", {})
        for resource, usage in resource_usage.items():
            if usage > 0.9:
                recommendations.append(f"High {resource} usage detected - consider scaling")
        
        return recommendations
    
    async def _find_stuck_steps(
        self,
        execution_state: Dict[str, Any]
    ) -> List[Tuple[str, int]]:
        """Find steps that are stuck"""
        stuck = []
        
        active_steps = execution_state.get("active_steps", {})
        for step_id, step_info in active_steps.items():
            if "start_time" in step_info:
                duration = (datetime.now() - datetime.fromisoformat(step_info["start_time"])).seconds
                if duration > step_info.get("expected_duration", 300) * 2:
                    stuck.append((step_id, duration))
        
        return stuck
    
    async def _check_resource_contention(
        self,
        execution_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Check for resource contention"""
        resource_usage = execution_state.get("resource_usage", {})
        contention = {}
        
        for resource, usage in resource_usage.items():
            if isinstance(usage, (int, float)):
                contention[resource] = min(usage, 1.0)
        
        return contention
    
    async def _detect_circular_dependencies(
        self,
        execution_state: Dict[str, Any]
    ) -> Optional[List[str]]:
        """Detect circular dependencies in execution graph"""
        # Simplified cycle detection
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str, graph: Dict[str, List[str]]) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in graph.get(node_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, graph):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        dependency_graph = execution_state.get("dependency_graph", {})
        for node in dependency_graph:
            if node not in visited:
                if has_cycle(node, dependency_graph):
                    return list(rec_stack)  # Return nodes in cycle
        
        return None
    
    async def _check_pattern_library(
        self,
        problem_space: ProblemSpace
    ) -> Optional[ExecutionPlan]:
        """Check if we have a cached plan for similar problem"""
        # Simple similarity check based on components and constraints
        for pattern_key, cached_plan in self.pattern_library.items():
            if pattern_key.startswith(f"{sorted(problem_space.components)}"):
                return cached_plan
        
        return None
    
    async def _adapt_cached_plan(
        self,
        cached_plan: ExecutionPlan,
        problem_space: ProblemSpace
    ) -> ExecutionPlan:
        """Adapt a cached plan to current problem"""
        # Create a copy and update parameters
        adapted_steps = []
        for step in cached_plan.steps:
            adapted_step = ExecutionStep(
                step_id=step.step_id,
                action=step.action,
                agent_type=step.agent_type,
                parameters={**step.parameters, "problem": problem_space.description},
                dependencies=step.dependencies,
                timeout=step.timeout
            )
            adapted_steps.append(adapted_step)
        
        return ExecutionPlan(
            plan_id=f"adapted_{datetime.now().timestamp()}",
            steps=adapted_steps,
            decision_points=cached_plan.decision_points,
            fallback_strategies=cached_plan.fallback_strategies,
            estimated_time=cached_plan.estimated_time,
            estimated_resources=cached_plan.estimated_resources
        )
    
    async def _identify_decision_points(
        self,
        steps: List[ExecutionStep],
        problem_space: ProblemSpace
    ) -> List[str]:
        """Identify decision points in execution"""
        decision_points = []
        
        # After analysis steps
        for i, step in enumerate(steps):
            if "analyze" in step.action:
                decision_points.append(f"after_{step.step_id}")
        
        # Before high-cost operations
        for step in steps:
            if step.timeout > 300:  # Long-running steps
                decision_points.append(f"before_{step.step_id}")
        
        return decision_points
    
    async def _create_fallback_strategies(
        self,
        steps: List[ExecutionStep],
        constraints: List[str]
    ) -> Dict[str, str]:
        """Create fallback strategies for steps"""
        strategies = {}
        
        for step in steps:
            if step.action == "generate_content":
                strategies[step.step_id] = "use_template_based_generation"
            elif step.action == "analyze_data":
                strategies[step.step_id] = "use_heuristic_analysis"
            elif "optimize" in step.action:
                strategies[step.step_id] = "use_greedy_approach"
            else:
                strategies[step.step_id] = "retry_with_reduced_scope"
        
        return strategies
    
    async def _estimate_resources(
        self,
        steps: List[ExecutionStep]
    ) -> Dict[str, Any]:
        """Estimate resource requirements"""
        total_time = sum(step.timeout for step in steps)
        
        # Estimate based on step types
        memory_usage = 0
        cpu_usage = 0
        
        for step in steps:
            if "analyze" in step.action:
                memory_usage += 500  # MB
                cpu_usage += 0.3
            elif "generate" in step.action:
                memory_usage += 300
                cpu_usage += 0.5
            else:
                memory_usage += 200
                cpu_usage += 0.2
        
        return {
            "time": total_time,
            "memory_mb": memory_usage,
            "cpu_cores": min(cpu_usage, 4.0),
            "parallel_capable": True
        }
    
    async def _store_in_pattern_library(
        self,
        problem_space: ProblemSpace,
        plan: ExecutionPlan
    ):
        """Store successful plans for reuse"""
        pattern_key = f"{sorted(problem_space.components)}_{sorted(problem_space.constraints)}"
        self.pattern_library[pattern_key] = plan
        
        # Keep library size manageable
        if len(self.pattern_library) > 100:
            # Remove oldest patterns
            oldest_key = list(self.pattern_library.keys())[0]
            del self.pattern_library[oldest_key]


# Test the implementation
async def test_adaptive_meta_reasoning():
    """Test the adaptive meta-reasoning engine"""
    engine = AdaptiveMetaReasoningEngine()
    
    # Test problem analysis
    problem = "Create a web scraper to analyze product prices and generate a report"
    problem_space = await engine.analyze_problem(problem)
    print(f"Problem complexity: {problem_space.complexity_score}")
    print(f"Components: {problem_space.components}")
    print(f"Required capabilities: {problem_space.required_capabilities}")
    
    # Test execution planning
    plan = await engine.create_execution_plan(problem_space)
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
    
    monitoring_result = await engine.monitor_execution(plan, current_state)
    print(f"\nExecution progress: {monitoring_result['progress']:.1f}%")
    print(f"Bottlenecks: {len(monitoring_result['bottlenecks'])}")
    
    # Test checkpointing
    checkpoint_id = await engine.checkpoint_state(current_state)
    restored_state = await engine.restore_from_checkpoint(checkpoint_id)
    print(f"\nCheckpoint created and verified: {restored_state == current_state}")


if __name__ == "__main__":
    asyncio.run(test_adaptive_meta_reasoning())