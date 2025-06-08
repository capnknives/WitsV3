"""
Code analysis functionality
"""

import ast
import logging
from typing import Optional, List, Dict, Any

from .models import CodeAnalysis


class CodeAnalyzer:
    """Analyzes code quality, complexity, and issues"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def analyze_code(self, code: str, language: str) -> Optional[CodeAnalysis]:
        """
        Analyze code quality based on language
        
        Args:
            code: Source code to analyze
            language: Programming language
            
        Returns:
            CodeAnalysis object or None
        """
        try:
            if language == 'python':
                return await self.analyze_python_code(code)
            elif language == 'javascript':
                return await self.analyze_js_code(code)
            else:
                return await self.analyze_generic_code(code, language)
        except Exception as e:
            self.logger.error(f"Error analyzing code: {e}")
            return self.create_default_analysis()
    
    async def analyze_python_code(self, code: str) -> CodeAnalysis:
        """Analyze Python code quality"""
        try:
            tree = ast.parse(code)
            
            # Calculate metrics
            complexity = self.calculate_cyclomatic_complexity(tree)
            line_count = len(code.split('\n'))
            
            # Basic maintainability index calculation
            # MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * (Cyclomatic Complexity) - 16.2 * ln(Lines of Code)
            # Simplified version:
            maintainability = max(0, min(100, 171 - 0.23 * complexity - 16.2 * (line_count / 100)))
            
            # Detect issues
            security_issues = self.detect_python_security_issues(code)
            performance_issues = self.detect_python_performance_issues(tree)
            style_violations = self.detect_python_style_violations(code)
            
            # Generate suggestions
            suggestions = []
            if complexity > 10:
                suggestions.append("Consider breaking down complex functions")
            if line_count > 500:
                suggestions.append("Consider splitting this module into smaller files")
            if len(security_issues) > 0:
                suggestions.append("Address security vulnerabilities immediately")
            
            return CodeAnalysis(
                complexity_score=complexity,
                maintainability_index=maintainability,
                test_coverage=0.0,  # Would need actual test coverage data
                security_issues=security_issues,
                performance_issues=performance_issues,
                style_violations=style_violations,
                suggestions=suggestions
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing Python code: {e}")
            return self.create_default_analysis()
    
    def calculate_cyclomatic_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity for Python code"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def detect_python_security_issues(self, code: str) -> List[str]:
        """Detect common security issues in Python code"""
        issues = []
        
        security_patterns = {
            'eval(': 'Use of eval() is a security risk',
            'exec(': 'Use of exec() is a security risk',
            '__import__': 'Dynamic imports can be security risks',
            'pickle.loads': 'Pickle deserialization of untrusted data is dangerous',
            'shell=True': 'Shell injection vulnerability with subprocess',
            'os.system': 'Command injection risk with os.system'
        }
        
        for pattern, issue in security_patterns.items():
            if pattern in code:
                issues.append(issue)
        
        # Check for SQL injection patterns
        if 'cursor.execute' in code and ('%s' not in code and '?' not in code):
            issues.append("Potential SQL injection - use parameterized queries")
        
        return issues
    
    def detect_python_performance_issues(self, tree: ast.AST) -> List[str]:
        """Detect performance issues in Python code"""
        issues = []
        
        for node in ast.walk(tree):
            # Detect string concatenation in loops
            if isinstance(node, ast.For):
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.AugAssign) and isinstance(subnode.op, ast.Add):
                        if isinstance(subnode.target, ast.Name) and isinstance(subnode.value, ast.Str):
                            issues.append("String concatenation in loop - use list and join()")
            
            # Detect repeated attribute access
            if isinstance(node, ast.Attribute):
                # This is simplified - real implementation would track usage
                pass
        
        return issues
    
    def detect_python_style_violations(self, code: str) -> List[str]:
        """Detect PEP 8 style violations"""
        violations = []
        
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Line length
            if len(line) > 79:
                violations.append(f"Line {i+1}: Line too long ({len(line)} > 79 characters)")
            
            # Tabs vs spaces
            if '\t' in line:
                violations.append(f"Line {i+1}: Use spaces instead of tabs")
            
            # Trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                violations.append(f"Line {i+1}: Trailing whitespace")
        
        return violations[:10]  # Limit to first 10 violations
    
    async def analyze_js_code(self, code: str) -> CodeAnalysis:
        """Analyze JavaScript code quality"""
        # Simplified JS analysis
        line_count = len(code.split('\n'))
        
        issues = []
        if 'eval(' in code:
            issues.append("Use of eval() is a security risk")
        if 'var ' in code:
            issues.append("Use 'let' or 'const' instead of 'var'")
        
        return CodeAnalysis(
            complexity_score=10.0,  # Would need proper JS AST analysis
            maintainability_index=75.0,
            test_coverage=0.0,
            security_issues=issues,
            performance_issues=[],
            style_violations=[],
            suggestions=["Consider using a linter like ESLint"]
        )
    
    async def analyze_generic_code(self, code: str, language: str) -> CodeAnalysis:
        """Generic code analysis for unsupported languages"""
        line_count = len(code.split('\n'))
        
        return CodeAnalysis(
            complexity_score=5.0,
            maintainability_index=80.0,
            test_coverage=0.0,
            security_issues=[],
            performance_issues=[],
            style_violations=[],
            suggestions=[f"Use {language}-specific analysis tools for detailed insights"]
        )
    
    def create_default_analysis(self) -> CodeAnalysis:
        """Create default analysis when analysis fails"""
        return CodeAnalysis(
            complexity_score=0.0,
            maintainability_index=50.0,
            test_coverage=0.0,
            security_issues=[],
            performance_issues=[],
            style_violations=[],
            suggestions=["Unable to analyze code - check for syntax errors"]
        )
    
    async def suggest_improvements(self, analysis: CodeAnalysis, language: str) -> List[str]:
        """Generate improvement suggestions based on analysis"""
        suggestions = analysis.suggestions.copy()
        
        # Add specific suggestions based on scores
        if analysis.complexity_score > 15:
            suggestions.append("High complexity detected - consider refactoring into smaller functions")
        
        if analysis.maintainability_index < 50:
            suggestions.append("Low maintainability - improve code structure and documentation")
        
        if len(analysis.security_issues) > 0:
            suggestions.append(f"Fix {len(analysis.security_issues)} security issues before deployment")
        
        # Language-specific suggestions
        if language == 'python':
            suggestions.append("Run 'black' for automatic code formatting")
            suggestions.append("Use 'mypy' for type checking")
        elif language == 'javascript':
            suggestions.append("Configure ESLint for consistent code style")
            suggestions.append("Consider TypeScript for better type safety")
        
        return suggestions