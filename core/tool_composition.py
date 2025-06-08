"""
Tool Composition Engine for WitsV3

This module implements intelligent tool composition and automatic workflow generation,
enabling complex multi-tool operations through dynamic composition.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from uuid import uuid4

from .schemas import StreamData, ToolCall, ToolResult
from .config import WitsV3Config

logger = logging.getLogger(__name__)


class CompositionStrategy(Enum):
    """Strategies for tool composition"""
    SEQUENTIAL = "sequential"      # Tools run one after another
    PARALLEL = "parallel"          # Tools run simultaneously
    CONDITIONAL = "conditional"    # Tools run based on conditions
    ITERATIVE = "iterative"        # Tools run in loops
    ADAPTIVE = "adaptive"          # Strategy adapts based on results


class DataFlow(Enum):
    """How data flows between tools"""
    PIPELINE = "pipeline"          # Output of one feeds input of next
    BROADCAST = "broadcast"        # One output feeds multiple inputs
    AGGREGATE = "aggregate"        # Multiple outputs combine into one
    TRANSFORM = "transform"        # Data transformed between tools
    FILTER = "filter"              # Data filtered between tools


@dataclass
class ToolSpec:
    """Specification of a tool's capabilities"""
    tool_name: str
    input_schema: Dict[str, Any]  # JSON schema for inputs
    output_schema: Dict[str, Any]  # JSON schema for outputs
    capabilities: Set[str] = field(default_factory=set)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    average_execution_time: float = 1.0
    success_rate: float = 0.95
    can_parallelize: bool = True
    idempotent: bool = True  # Safe to retry


@dataclass
class CompositionNode:
    """Node in a tool composition graph"""
    node_id: str = field(default_factory=lambda: str(uuid4()))
    tool_name: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Node IDs this depends on
    conditions: List[Dict[str, Any]] = field(default_factory=list)  # Conditions for execution
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 300.0
    
    def can_execute(self, completed_nodes: Set[str]) -> bool:
        """Check if node can execute based on dependencies"""
        return all(dep in completed_nodes for dep in self.dependencies)


@dataclass
class Workflow:
    """Represents a composed workflow of tools"""
    workflow_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    nodes: List[CompositionNode] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (from_node, to_node)
    strategy: CompositionStrategy = CompositionStrategy.SEQUENTIAL
    data_flow: DataFlow = DataFlow.PIPELINE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_execution_order(self) -> List[List[CompositionNode]]:
        """Get nodes grouped by execution order (parallel groups)"""
        # Topological sort with level grouping
        levels = []
        completed = set()
        remaining = self.nodes.copy()
        
        while remaining:
            level = []
            for node in remaining:
                if node.can_execute(completed):
                    level.append(node)
            
            if not level:
                # Circular dependency or disconnected nodes
                level = [remaining[0]]  # Force progress
                
            levels.append(level)
            for node in level:
                completed.add(node.node_id)
                remaining.remove(node)
                
        return levels


@dataclass
class WorkflowExecution:
    """Tracks workflow execution state"""
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    workflow_id: str = ""
    status: str = "pending"  # pending, running, completed, failed
    completed_nodes: Set[str] = field(default_factory=set)
    node_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ToolCompositionEngine(ABC):
    """
    Abstract base class for tool composition engines.
    
    Handles intelligent composition of tools into workflows.
    """
    
    def __init__(self, config: WitsV3Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.tool_registry: Dict[str, ToolSpec] = {}
        self.workflow_templates: Dict[str, Workflow] = {}
        self.execution_history: List[WorkflowExecution] = []
    
    @abstractmethod
    async def compose_workflow(
        self,
        goal: str,
        available_tools: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Workflow:
        """
        Compose a workflow to achieve the given goal.
        
        Args:
            goal: What to achieve
            available_tools: List of available tool names
            constraints: Optional constraints (time, resources, etc.)
            
        Returns:
            Composed workflow
        """
        pass
    
    @abstractmethod
    async def optimize_workflow(
        self,
        workflow: Workflow,
        optimization_goals: List[str]
    ) -> Workflow:
        """
        Optimize an existing workflow.
        
        Args:
            workflow: Workflow to optimize
            optimization_goals: Goals like "speed", "reliability", "resource_usage"
            
        Returns:
            Optimized workflow
        """
        pass
    
    @abstractmethod
    async def execute_workflow(
        self,
        workflow: Workflow,
        initial_inputs: Dict[str, Any],
        tool_executor: Callable
    ) -> WorkflowExecution:
        """
        Execute a workflow.
        
        Args:
            workflow: Workflow to execute
            initial_inputs: Initial inputs to the workflow
            tool_executor: Function to execute individual tools
            
        Returns:
            Execution results
        """
        pass
    
    def register_tool(self, tool_spec: ToolSpec) -> None:
        """Register a tool specification"""
        self.tool_registry[tool_spec.tool_name] = tool_spec
        self.logger.info(f"Registered tool: {tool_spec.tool_name}")
    
    def save_workflow_template(self, name: str, workflow: Workflow) -> None:
        """Save a workflow as a reusable template"""
        workflow.name = name
        self.workflow_templates[name] = workflow
        self.logger.info(f"Saved workflow template: {name}")
    
    async def learn_from_execution(
        self,
        execution: WorkflowExecution,
        feedback: Optional[Dict[str, Any]] = None
    ) -> None:
        """Learn from workflow execution to improve future compositions"""
        self.execution_history.append(execution)
        
        # Update tool success rates based on execution
        for node_id, result in execution.node_results.items():
            if "tool_name" in result:
                tool_name = result["tool_name"]
                if tool_name in self.tool_registry:
                    # Simple exponential moving average
                    old_rate = self.tool_registry[tool_name].success_rate
                    success = result.get("success", False)
                    new_rate = 0.9 * old_rate + 0.1 * (1.0 if success else 0.0)
                    self.tool_registry[tool_name].success_rate = new_rate


class IntelligentToolComposer(ToolCompositionEngine):
    """
    Intelligent implementation of tool composition engine.
    
    Uses heuristics and learning to compose effective workflows.
    """
    
    def __init__(self, config: WitsV3Config):
        super().__init__(config)
        self.composition_patterns: Dict[str, List[str]] = {
            # Common patterns for tool composition
            "research_and_write": ["web_search", "summarize", "write_document"],
            "code_and_test": ["generate_code", "run_tests", "fix_errors"],
            "analyze_and_report": ["read_data", "analyze", "generate_report"],
            "iterate_until_good": ["attempt", "evaluate", "improve"],
        }
    
    async def compose_workflow(
        self,
        goal: str,
        available_tools: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Workflow:
        """Compose a workflow using intelligent heuristics"""
        
        # Analyze goal to determine required capabilities
        required_capabilities = await self._analyze_goal(goal)
        
        # Find tools that provide required capabilities
        selected_tools = await self._select_tools(
            required_capabilities,
            available_tools,
            constraints
        )
        
        # Determine composition strategy
        strategy = await self._determine_strategy(
            goal,
            selected_tools,
            constraints
        )
        
        # Create workflow nodes
        nodes = await self._create_nodes(
            selected_tools,
            goal,
            strategy
        )
        
        # Establish data flow between nodes
        edges = await self._create_edges(
            nodes,
            strategy
        )
        
        workflow = Workflow(
            name=f"Workflow for: {goal[:50]}",
            description=goal,
            nodes=nodes,
            edges=edges,
            strategy=strategy,
            data_flow=DataFlow.PIPELINE if strategy == CompositionStrategy.SEQUENTIAL else DataFlow.BROADCAST
        )
        
        return workflow
    
    async def optimize_workflow(
        self,
        workflow: Workflow,
        optimization_goals: List[str]
    ) -> Workflow:
        """Optimize workflow based on goals"""
        
        optimized = Workflow(
            name=workflow.name + " (Optimized)",
            description=workflow.description,
            nodes=workflow.nodes.copy(),
            edges=workflow.edges.copy(),
            strategy=workflow.strategy,
            data_flow=workflow.data_flow
        )
        
        for goal in optimization_goals:
            if goal == "speed":
                optimized = await self._optimize_for_speed(optimized)
            elif goal == "reliability":
                optimized = await self._optimize_for_reliability(optimized)
            elif goal == "resource_usage":
                optimized = await self._optimize_for_resources(optimized)
                
        return optimized
    
    async def execute_workflow(
        self,
        workflow: Workflow,
        initial_inputs: Dict[str, Any],
        tool_executor: Callable
    ) -> WorkflowExecution:
        """Execute a workflow with intelligent scheduling"""
        
        execution = WorkflowExecution(
            workflow_id=workflow.workflow_id,
            status="running",
            started_at=datetime.now()
        )
        
        try:
            # Get execution order
            execution_levels = workflow.get_execution_order()
            
            # Execute levels
            current_context = initial_inputs.copy()
            
            for level in execution_levels:
                if workflow.strategy == CompositionStrategy.PARALLEL:
                    # Execute nodes in parallel
                    results = await asyncio.gather(*[
                        self._execute_node(node, current_context, tool_executor)
                        for node in level
                    ])
                    
                    for node, result in zip(level, results):
                        execution.node_results[node.node_id] = result
                        if result.get("success", False):
                            execution.completed_nodes.add(node.node_id)
                            # Update context with outputs
                            current_context.update(result.get("outputs", {}))
                        else:
                            execution.errors.append({
                                "node_id": node.node_id,
                                "error": result.get("error", "Unknown error")
                            })
                else:
                    # Execute nodes sequentially
                    for node in level:
                        result = await self._execute_node(
                            node,
                            current_context,
                            tool_executor
                        )
                        
                        execution.node_results[node.node_id] = result
                        if result.get("success", False):
                            execution.completed_nodes.add(node.node_id)
                            current_context.update(result.get("outputs", {}))
                        else:
                            execution.errors.append({
                                "node_id": node.node_id,
                                "error": result.get("error", "Unknown error")
                            })
                            
                            # Stop on error in sequential mode
                            if workflow.strategy == CompositionStrategy.SEQUENTIAL:
                                break
                
                # Check if we should continue
                if execution.errors and workflow.strategy != CompositionStrategy.ADAPTIVE:
                    break
                    
            execution.status = "completed" if not execution.errors else "failed"
            
        except Exception as e:
            execution.status = "failed"
            execution.errors.append({
                "error": str(e),
                "type": "execution_error"
            })
            
        finally:
            execution.completed_at = datetime.now()
            
        return execution
    
    # Helper methods
    
    async def _analyze_goal(self, goal: str) -> Set[str]:
        """Analyze goal to determine required capabilities"""
        capabilities = set()
        goal_lower = goal.lower()
        
        # Pattern matching for common tasks
        patterns = {
            "search": ["web_search", "information_retrieval"],
            "write": ["text_generation", "content_creation"],
            "code": ["code_generation", "programming"],
            "analyze": ["data_analysis", "pattern_recognition"],
            "summarize": ["summarization", "extraction"],
            "test": ["testing", "validation"],
            "create": ["generation", "creation"],
            "fix": ["debugging", "error_correction"],
            "optimize": ["optimization", "improvement"]
        }
        
        for keyword, caps in patterns.items():
            if keyword in goal_lower:
                capabilities.update(caps)
                
        # Default capability
        if not capabilities:
            capabilities.add("general_processing")
            
        return capabilities
    
    async def _select_tools(
        self,
        required_capabilities: Set[str],
        available_tools: List[str],
        constraints: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Select tools that provide required capabilities"""
        selected = []
        
        for tool_name in available_tools:
            if tool_name in self.tool_registry:
                tool_spec = self.tool_registry[tool_name]
                
                # Check if tool provides any required capability
                if tool_spec.capabilities.intersection(required_capabilities):
                    # Check constraints
                    if constraints:
                        if "max_time" in constraints:
                            if tool_spec.average_execution_time > constraints["max_time"]:
                                continue
                        if "min_reliability" in constraints:
                            if tool_spec.success_rate < constraints["min_reliability"]:
                                continue
                                
                    selected.append(tool_name)
                    
        return selected
    
    async def _determine_strategy(
        self,
        goal: str,
        selected_tools: List[str],
        constraints: Optional[Dict[str, Any]]
    ) -> CompositionStrategy:
        """Determine the best composition strategy"""
        
        # Check constraints first
        if constraints and "max_time" in constraints:
            # Prefer parallel execution for time constraints
            return CompositionStrategy.PARALLEL
            
        # Simple heuristics
        if len(selected_tools) == 1:
            return CompositionStrategy.SEQUENTIAL
            
        if "iterate" in goal.lower() or "improve" in goal.lower():
            return CompositionStrategy.ITERATIVE
            
        if "if" in goal.lower() or "when" in goal.lower():
            return CompositionStrategy.CONDITIONAL
            
        # Default to sequential for predictability
        return CompositionStrategy.SEQUENTIAL
    
    async def _create_nodes(
        self,
        selected_tools: List[str],
        goal: str,
        strategy: CompositionStrategy
    ) -> List[CompositionNode]:
        """Create workflow nodes from selected tools"""
        nodes = []
        
        for i, tool_name in enumerate(selected_tools):
            # Get timeout from registry or use default
            if tool_name in self.tool_registry:
                timeout = self.tool_registry[tool_name].average_execution_time * 2
            else:
                timeout = 300.0  # Default timeout
                
            node = CompositionNode(
                tool_name=tool_name,
                inputs={"goal": goal, "step": i + 1},
                timeout=timeout
            )
            
            # Set dependencies based on strategy
            if strategy == CompositionStrategy.SEQUENTIAL and i > 0:
                node.dependencies = [nodes[i-1].node_id]
                
            nodes.append(node)
            
        return nodes
    
    async def _create_edges(
        self,
        nodes: List[CompositionNode],
        strategy: CompositionStrategy
    ) -> List[Tuple[str, str]]:
        """Create edges between nodes"""
        edges = []
        
        if strategy == CompositionStrategy.SEQUENTIAL:
            for i in range(len(nodes) - 1):
                edges.append((nodes[i].node_id, nodes[i+1].node_id))
        elif strategy == CompositionStrategy.PARALLEL:
            # No edges for pure parallel execution
            pass
        elif strategy == CompositionStrategy.ITERATIVE:
            # Create cycle
            for i in range(len(nodes) - 1):
                edges.append((nodes[i].node_id, nodes[i+1].node_id))
            if len(nodes) > 1:
                edges.append((nodes[-1].node_id, nodes[0].node_id))  # Loop back
                
        return edges
    
    async def _execute_node(
        self,
        node: CompositionNode,
        context: Dict[str, Any],
        tool_executor: Callable
    ) -> Dict[str, Any]:
        """Execute a single node"""
        try:
            # Prepare inputs
            tool_inputs = {**context, **node.inputs}
            
            # Execute tool
            result = await tool_executor(
                node.tool_name,
                tool_inputs,
                timeout=node.timeout
            )
            
            return {
                "success": True,
                "tool_name": node.tool_name,
                "outputs": result,
                "execution_time": result.get("execution_time", 0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "tool_name": node.tool_name,
                "error": str(e)
            }
    
    async def _optimize_for_speed(self, workflow: Workflow) -> Workflow:
        """Optimize workflow for speed"""
        # Convert sequential to parallel where possible
        if workflow.strategy == CompositionStrategy.SEQUENTIAL:
            # Check which nodes can run in parallel
            can_parallelize = True
            for node in workflow.nodes:
                if node.tool_name in self.tool_registry:
                    if not self.tool_registry[node.tool_name].can_parallelize:
                        can_parallelize = False
                        break
                        
            if can_parallelize:
                workflow.strategy = CompositionStrategy.PARALLEL
                # Remove sequential dependencies
                for node in workflow.nodes:
                    node.dependencies = []
                workflow.edges = []
                
        return workflow
    
    async def _optimize_for_reliability(self, workflow: Workflow) -> Workflow:
        """Optimize workflow for reliability"""
        # Add retry policies to unreliable tools
        for node in workflow.nodes:
            if node.tool_name in self.tool_registry:
                tool_spec = self.tool_registry[node.tool_name]
                if tool_spec.success_rate < 0.9:
                    node.retry_policy = {
                        "max_retries": 3,
                        "backoff": "exponential",
                        "initial_delay": 1.0
                    }
                    
        return workflow
    
    async def _optimize_for_resources(self, workflow: Workflow) -> Workflow:
        """Optimize workflow for resource usage"""
        # This would implement resource-aware scheduling
        # For now, just a placeholder
        return workflow


# Test function
async def test_tool_composition():
    """Test the tool composition engine"""
    from .config import WitsV3Config
    
    print("Testing Tool Composition Engine...")
    
    config = WitsV3Config()
    composer = IntelligentToolComposer(config)
    
    # Register some test tools
    composer.register_tool(ToolSpec(
        tool_name="web_search",
        input_schema={"query": "string"},
        output_schema={"results": "array"},
        capabilities={"web_search", "information_retrieval"},
        average_execution_time=2.0
    ))
    
    composer.register_tool(ToolSpec(
        tool_name="summarize",
        input_schema={"text": "string"},
        output_schema={"summary": "string"},
        capabilities={"summarization", "extraction"},
        average_execution_time=1.5
    ))
    
    composer.register_tool(ToolSpec(
        tool_name="write_document",
        input_schema={"content": "string", "format": "string"},
        output_schema={"document": "string"},
        capabilities={"text_generation", "content_creation"},
        average_execution_time=3.0
    ))
    
    print("✓ Registered 3 test tools")
    
    # Compose a workflow
    workflow = await composer.compose_workflow(
        goal="Search for information about quantum computing and write a summary report",
        available_tools=["web_search", "summarize", "write_document"]
    )
    
    print(f"✓ Composed workflow: {workflow.name}")
    print(f"  Strategy: {workflow.strategy.value}")
    print(f"  Nodes: {len(workflow.nodes)}")
    for i, node in enumerate(workflow.nodes):
        print(f"    {i+1}. {node.tool_name}")
    
    # Optimize the workflow
    optimized = await composer.optimize_workflow(
        workflow,
        ["speed", "reliability"]
    )
    
    print(f"✓ Optimized workflow")
    print(f"  New strategy: {optimized.strategy.value}")
    
    # Simulate execution
    async def mock_tool_executor(tool_name: str, inputs: Dict[str, Any], timeout: float):
        """Mock tool executor for testing"""
        await asyncio.sleep(0.1)  # Simulate execution time
        
        if tool_name == "web_search":
            return {"results": ["Quantum computing uses qubits", "Superposition principle"]}
        elif tool_name == "summarize":
            return {"summary": "Quantum computing leverages quantum mechanics"}
        elif tool_name == "write_document":
            return {"document": "# Quantum Computing Report\n\nQuantum computing..."}
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    execution = await composer.execute_workflow(
        optimized,
        {"topic": "quantum computing"},
        mock_tool_executor
    )
    
    print(f"✓ Executed workflow")
    print(f"  Status: {execution.status}")
    print(f"  Completed nodes: {len(execution.completed_nodes)}")
    print(f"  Duration: {execution.duration:.2f}s" if execution.duration else "  Duration: N/A")
    
    print("\nTool Composition Engine tests completed! 🎉")


if __name__ == "__main__":
    asyncio.run(test_tool_composition())