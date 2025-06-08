"""
Data models for the advanced coding module
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any


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
    
    def add_file(self, filename: str, content: str) -> None:
        """Add a file to the project"""
        self.files[filename] = content
    
    def add_test(self, test_filename: str, content: str) -> None:
        """Add a test file to the project"""
        self.tests[test_filename] = content
    
    def get_file_count(self) -> int:
        """Get total number of files in project"""
        return len(self.files) + len(self.tests)
    
    def get_lines_of_code(self) -> int:
        """Calculate total lines of code in project"""
        loc = 0
        for content in self.files.values():
            loc += len(content.split('\n'))
        for content in self.tests.values():
            loc += len(content.split('\n'))
        return loc


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
    
    def get_overall_score(self) -> float:
        """Calculate overall code quality score"""
        scores = [
            self.maintainability_index,
            100 - (self.complexity_score * 5),  # Convert complexity to quality
            self.test_coverage
        ]
        
        # Deduct points for issues
        issue_penalty = (
            len(self.security_issues) * 10 +
            len(self.performance_issues) * 5 +
            len(self.style_violations) * 2
        )
        
        avg_score = sum(scores) / len(scores)
        final_score = max(0, avg_score - issue_penalty)
        
        return min(100, final_score)
    
    def get_priority_issues(self) -> List[str]:
        """Get high priority issues that need fixing"""
        priority_issues = []
        
        # Security issues are highest priority
        for issue in self.security_issues:
            priority_issues.append(f"SECURITY: {issue}")
        
        # Performance issues are next
        for issue in self.performance_issues[:3]:  # Top 3
            priority_issues.append(f"PERFORMANCE: {issue}")
        
        return priority_issues


# Language configuration
SUPPORTED_LANGUAGES = {
    'python': {
        'extensions': ['.py'],
        'frameworks': ['django', 'flask', 'fastapi', 'pytorch', 'tensorflow'],
        'testing': ['pytest', 'unittest'],
        'linting': ['pylint', 'flake8', 'black']
    },
    'javascript': {
        'extensions': ['.js', '.ts', '.jsx', '.tsx'],
        'frameworks': ['react', 'vue', 'angular', 'node', 'express'],
        'testing': ['jest', 'mocha', 'cypress'],
        'linting': ['eslint', 'prettier']
    },
    'java': {
        'extensions': ['.java'],
        'frameworks': ['spring', 'springboot', 'hibernate'],
        'testing': ['junit', 'testng'],
        'linting': ['checkstyle', 'spotbugs']
    },
    'rust': {
        'extensions': ['.rs'],
        'frameworks': ['tokio', 'actix', 'rocket'],
        'testing': ['cargo test'],
        'linting': ['clippy', 'rustfmt']
    },
    'go': {
        'extensions': ['.go'],
        'frameworks': ['gin', 'echo', 'fiber'],
        'testing': ['go test'],
        'linting': ['golint', 'gofmt']
    }
}

# Project templates
PROJECT_TEMPLATES = {
    'web_app': {
        'structure': ['src/', 'tests/', 'docs/', 'config/'],
        'files': ['main.py', 'requirements.txt', 'README.md', '.gitignore']
    },
    'api': {
        'structure': ['api/', 'models/', 'tests/', 'docs/'],
        'files': ['app.py', 'requirements.txt', 'README.md', 'docker-compose.yml']
    },
    'cli_tool': {
        'structure': ['src/', 'tests/', 'docs/'],
        'files': ['cli.py', 'setup.py', 'README.md', 'requirements.txt']
    },
    'library': {
        'structure': ['src/', 'tests/', 'docs/', 'examples/'],
        'files': ['__init__.py', 'setup.py', 'README.md', 'requirements.txt']
    }
}