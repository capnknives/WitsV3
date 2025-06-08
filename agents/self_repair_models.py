"""
Data models for the Self-Repair Agent
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


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
