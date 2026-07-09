"""Data models for the tool composition engine (split from tool_composition.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class CompositionStrategy(Enum):
    """Strategies for tool composition"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    ADAPTIVE = "adaptive"


class DataFlow(Enum):
    """How data flows between tools"""

    PIPELINE = "pipeline"
    BROADCAST = "broadcast"
    AGGREGATE = "aggregate"
    TRANSFORM = "transform"
    FILTER = "filter"


@dataclass
class ToolSpec:
    """Specification of a tool's capabilities"""

    tool_name: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    capabilities: set[str] = field(default_factory=set)
    resource_requirements: dict[str, float] = field(default_factory=dict)
    average_execution_time: float = 1.0
    success_rate: float = 0.95
    can_parallelize: bool = True
    idempotent: bool = True


@dataclass
class CompositionNode:
    """Node in a tool composition graph"""

    node_id: str = field(default_factory=lambda: str(uuid4()))
    tool_name: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    conditions: list[dict[str, Any]] = field(default_factory=list)
    retry_policy: dict[str, Any] = field(default_factory=dict)
    timeout: float = 300.0

    def can_execute(self, completed_nodes: set[str]) -> bool:
        return all(dep in completed_nodes for dep in self.dependencies)


@dataclass
class Workflow:
    """Represents a composed workflow of tools"""

    workflow_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    nodes: list[CompositionNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    strategy: CompositionStrategy = CompositionStrategy.SEQUENTIAL
    data_flow: DataFlow = DataFlow.PIPELINE
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def get_execution_order(self) -> list[list[CompositionNode]]:
        levels: list[list[CompositionNode]] = []
        completed: set[str] = set()
        remaining = self.nodes.copy()
        while remaining:
            level = [node for node in remaining if node.can_execute(completed)]
            if not level:
                level = [remaining[0]]
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
    status: str = "pending"
    completed_nodes: set[str] = field(default_factory=set)
    node_results: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def duration(self) -> float | None:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
