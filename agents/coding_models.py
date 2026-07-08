# agents/coding_models.py
"""Data models for the advanced coding agent."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class CodeProject:
    """Represents a coding project"""

    id: str
    name: str
    description: str
    language: str
    project_type: str  # web_app, cli_tool, library, api, etc.
    structure: Dict[str, Any]
    dependencies: List[str]
    files: Dict[str, str]  # filename -> content
    tests: Dict[str, str]  # test_filename -> content
    documentation: str
    status: str = "planning"  # planning, development, testing, complete
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class CodeAnalysis:
    """Represents code analysis results"""

    complexity_score: float
    maintainability_index: float
    test_coverage: float
    security_issues: List[str]
    performance_issues: List[str]
    style_violations: List[str]
    suggestions: List[str]
