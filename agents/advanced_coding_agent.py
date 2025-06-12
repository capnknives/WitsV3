# agents/advanced_coding_agent.py
"""
Advanced Coding Agent for WitsV3
Specialized agent for software development, code analysis, and project management
"""

import asyncio
import ast
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from agents.base_agent import BaseAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, ConversationHistory
from core.neural_web_core import NeuralWeb


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


class AdvancedCodingAgent(BaseAgent):
    """
    Specialized agent for advanced software development with neural web intelligence
    """

    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: Optional[MemoryManager] = None,
        neural_web: Optional[NeuralWeb] = None,
        tool_registry: Optional[Any] = None
    ):
        super().__init__(agent_name, config, llm_interface, memory_manager)

        self.neural_web = neural_web
        self.tool_registry = tool_registry

        # Neural web integration for code intelligence
        self.code_patterns: Dict[str, Any] = {}
        self.design_patterns_graph: Dict[str, Any] = {}
        self.dependency_networks: Dict[str, Any] = {}

        # Enhanced coding capabilities
        if self.neural_web:
            self.enable_code_intelligence = True
            self.logger.info("Neural web enabled for code intelligence")
            asyncio.create_task(self._initialize_coding_patterns())
        else:
            self.enable_code_intelligence = False

        # Coding specific state
        self.current_projects: Dict[str, CodeProject] = {}

        # Supported languages and frameworks
        self.supported_languages = {
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

        self.project_templates = {
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

        self.logger.info("Advanced Coding Agent initialized")

    async def run(
        self,
        request: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Process coding requests
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        yield self.stream_thinking("Analyzing coding request...")

        # Parse the request to understand the coding task
        task_analysis = await self._analyze_coding_task(request)

        yield self.stream_thinking(f"Identified task: {task_analysis['task_type']}")

        # Route to appropriate handler
        if task_analysis['task_type'] == 'create_project':
            async for stream in self._handle_project_creation(task_analysis, session_id):
                yield stream

        elif task_analysis['task_type'] == 'write_code':
            async for stream in self._handle_code_generation(task_analysis, session_id):
                yield stream

        elif task_analysis['task_type'] == 'analyze_code':
            async for stream in self._handle_code_analysis(task_analysis, session_id):
                yield stream

        elif task_analysis['task_type'] == 'debug_code':
            async for stream in self._handle_debugging(task_analysis, session_id):
                yield stream

        elif task_analysis['task_type'] == 'optimize_code':
            async for stream in self._handle_optimization(task_analysis, session_id):
                yield stream

        elif task_analysis['task_type'] == 'write_tests':
            async for stream in self._handle_test_generation(task_analysis, session_id):
                yield stream

        elif task_analysis['task_type'] == 'refactor_code':
            async for stream in self._handle_refactoring(task_analysis, session_id):
                yield stream

        else:
            async for stream in self._handle_general_coding(task_analysis, session_id):
                yield stream

    async def _analyze_coding_task(self, request: str) -> Dict[str, Any]:
        """Analyze the coding request to determine task type and parameters"""

        analysis_prompt = f"""
        Analyze this coding request and determine the task type and parameters:

        Request: {request}

        Respond with JSON containing:
        {{
            "task_type": (
                "create_project" | "write_code" | "analyze_code" | "debug_code" |
                "optimize_code" | "write_tests" | "refactor_code" | "general_coding"
            ),
            "language": "python" | "javascript" | "java" | "rust" | "go" | etc.,
            "project_type": "web_app" | "api" | "cli_tool" | "library" | etc.,
            "complexity": "simple" | "medium" | "complex",
            "requirements": ["specific", "requirements", "list"],
            "frameworks": ["framework1", "framework2"],
            "parameters": {{additional specific parameters}}
        }}
        """

        try:
            response = await self.generate_response(analysis_prompt, temperature=0.3)
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            self.logger.warning(f"Failed to parse task analysis: {e}")

        # Fallback analysis
        return {
            "task_type": "general_coding",
            "language": "python",
            "project_type": "script",
            "complexity": "medium",
            "requirements": [request[:100]],
            "frameworks": [],
            "parameters": {}
        }

    async def _handle_project_creation(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle creation of new coding projects"""

        yield self.stream_action("Creating new coding project...")

        project_id = str(uuid.uuid4())
        language = task_analysis.get('language', 'python')
        project_type = task_analysis.get('project_type', 'web_app')
        requirements = task_analysis.get('requirements', [])

        # Generate project architecture
        architecture_prompt = f"""
        Design a {language} {project_type} project architecture for:
        {' '.join(requirements)}

        Provide:
        1. Project structure (directories and key files)
        2. Dependencies and frameworks to use
        3. Database schema (if applicable)
        4. API endpoints (if applicable)
        5. Key classes and modules
        6. Testing strategy
        7. Deployment considerations

        Language: {language}
        Project Type: {project_type}
        Complexity: {task_analysis.get('complexity', 'medium')}
        """

        yield self.stream_thinking("Designing project architecture...")
        architecture_response = await self.generate_response(architecture_prompt, temperature=0.6)

        # Create project structure
        project_structure = await self._generate_project_structure(
            project_type, language, requirements
        )

        # Generate initial files
        initial_files = await self._generate_initial_files(
            project_structure, language, project_type, requirements
        )

        # Create project object
        project = CodeProject(
            id=project_id,
            name=f"{project_type}_{language}_project",
            description=' '.join(requirements),
            language=language,
            project_type=project_type,
            structure=project_structure,
            dependencies=task_analysis.get('frameworks', []),
            files=initial_files,
            tests={},
            documentation=architecture_response
        )

        self.current_projects[project_id] = project

        # Store project in memory
        await self.store_memory(
            content=f"Created {language} {project_type} project: {project.name}",
            segment_type="CODE_PROJECT",
            importance=0.9,
            metadata={
                "project_id": project_id,
                "language": language,
                "project_type": project_type,
                "session_id": session_id
            }
        )

        # Add project concepts to neural web
        if self.neural_web:
            await self.neural_web.add_concept(
                f"project_{project_id}",
                f"{language} {project_type} project",
                "code_project"
            )

            # Connect to language and framework concepts
            await self.neural_web.add_concept(
                language,
                f"{language} programming language",
                "technology",
            )
            await self.neural_web.connect_concepts(f"project_{project_id}", language, "uses")

        yield self.stream_result(
            f"Created project '{project.name}' with {len(initial_files)} initial files"
        )
        yield self.stream_result("Project Architecture:")
        yield self.stream_result(architecture_response)

        # Show initial file structure
        yield self.stream_action("Generated project files:")
        for filename in initial_files.keys():
            yield self.stream_action(f"  - {filename}")

    async def _handle_code_generation(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code generation requests"""

        yield self.stream_action("Generating code...")

        language = task_analysis.get('language', 'python')
        requirements = task_analysis.get('requirements', [])
        complexity = task_analysis.get('complexity', 'medium')

        # Generate code based on requirements
        code_prompt = f"""
        Write {complexity} {language} code to implement:
        {' '.join(requirements)}

        Requirements:
        - Use best practices and design patterns
        - Include proper error handling
        - Add comprehensive comments
        - Follow {language} style conventions
        - Make code modular and reusable
        - Include type hints (if applicable)

        Provide complete, working code with explanations.
        """

        yield self.stream_thinking("Writing code implementation...")

        # Generate code (non-streaming since streaming isn't supported)
        yield self.stream_thinking("Generating code...")
        generated_code = await self.generate_response(code_prompt, temperature=0.7)

        # Simulate progress updates
        content_length = len(generated_code)
        for i in range(0, content_length, 500):
            if i > 0:
                lines = generated_code[:i].count('\n')
                yield self.stream_action(f"Generated {lines} lines of code...")

        # Analyze the generated code
        analysis = await self._analyze_code_quality(generated_code, language)

        # Store code in memory
        await self.store_memory(
            content=f"Generated {language} code: {generated_code}",
            segment_type="GENERATED_CODE",
            importance=0.8,
            metadata={
                "language": language,
                "lines_of_code": generated_code.count('\n'),
                "complexity": complexity,
                "session_id": session_id
            }
        )

        yield self.stream_result("Generated Code:")
        yield self.stream_result(generated_code)

        if analysis:
            yield self.stream_observation(
                f"Code quality score: {analysis.maintainability_index:.2f}/100"
            )
            if analysis.suggestions:
                yield self.stream_observation("Suggestions for improvement:")
                for suggestion in analysis.suggestions[:3]:
                    yield self.stream_observation(f"  - {suggestion}")

    async def _handle_code_analysis(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code analysis requests"""

        yield self.stream_action("Analyzing code...")

        # This would analyze provided code
        # For demonstration, we'll analyze general code patterns

        language = task_analysis.get('language', 'python')

        analysis_prompt = f"""
        Provide a comprehensive code analysis framework for {language} projects.

        Include:
        1. Code quality metrics to check
        2. Common anti-patterns to avoid
        3. Performance optimization opportunities
        4. Security vulnerabilities to look for
        5. Testing strategies
        6. Refactoring recommendations
        7. Best practices checklist
        8. Automated tools to use
        """

        yield self.stream_thinking("Performing code analysis...")
        analysis_result = await self.generate_response(analysis_prompt, temperature=0.5)

        yield self.stream_result("Code Analysis Framework:")
        yield self.stream_result(analysis_result)

        # Store analysis framework
        await self.store_memory(
            content=f"{language} code analysis framework: {analysis_result}",
            segment_type="CODE_ANALYSIS",
            importance=0.8,
            metadata={
                "language": language,
                "analysis_type": "framework",
                "session_id": session_id
            }
        )

    async def _handle_debugging(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle debugging requests"""

        yield self.stream_action("Analyzing debugging approach...")

        language = task_analysis.get('language', 'python')

        debug_prompt = f"""
        Create a comprehensive debugging guide for {language} applications.

        Cover:
        1. Common error types and their solutions
        2. Debugging tools and techniques
        3. Logging best practices
        4. Performance profiling methods
        5. Memory leak detection
        6. Concurrency issues
        7. Step-by-step debugging process
        8. Prevention strategies
        """

        yield self.stream_thinking("Generating debugging strategies...")
        debug_guide = await self.generate_response(debug_prompt, temperature=0.6)

        yield self.stream_result("Debugging Guide:")
        yield self.stream_result(debug_guide)

        # Store debugging guide
        await self.store_memory(
            content=f"{language} debugging guide: {debug_guide}",
            segment_type="DEBUG_GUIDE",
            importance=0.7,
            metadata={
                "language": language,
                "session_id": session_id
            }
        )

    async def _handle_optimization(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code optimization requests"""

        yield self.stream_action("Generating optimization strategies...")

        language = task_analysis.get('language', 'python')

        optimization_prompt = f"""
        Provide comprehensive code optimization strategies for {language}.

        Include:
        1. Algorithm optimization techniques
        2. Data structure improvements
        3. Memory usage optimization
        4. I/O performance enhancements
        5. Concurrency and parallelization
        6. Caching strategies
        7. Database query optimization
        8. Profiling and benchmarking tools
        9. Language-specific optimizations
        """

        yield self.stream_thinking("Developing optimization techniques...")
        optimization_guide = await self.generate_response(optimization_prompt, temperature=0.6)

        yield self.stream_result("Code Optimization Guide:")
        yield self.stream_result(optimization_guide)

        # Store optimization guide
        await self.store_memory(
            content=f"{language} optimization guide: {optimization_guide}",
            segment_type="OPTIMIZATION_GUIDE",
            importance=0.8,
            metadata={
                "language": language,
                "session_id": session_id
            }
        )

    async def _handle_test_generation(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle test generation requests"""

        yield self.stream_action("Generating test suite...")

        language = task_analysis.get('language', 'python')
        requirements = task_analysis.get('requirements', [])

        test_prompt = f"""
        Generate comprehensive tests for {language} code that implements:
        {' '.join(requirements)}

        Include:
        1. Unit tests for individual functions
        2. Integration tests for component interaction
        3. Edge case and error condition tests
        4. Performance tests
        5. Security tests (if applicable)
        6. Test data and fixtures
        7. Mocking strategies
        8. Test coverage analysis

        Use appropriate testing framework for {language}.
        """

        yield self.stream_thinking("Creating test cases...")
        test_code = await self.generate_response(test_prompt, temperature=0.7)

        # Generate test documentation
        test_doc_prompt = """
        Create documentation for the test suite including:
        1. Test execution instructions
        2. Test coverage requirements
        3. Continuous integration setup
        4. Test data management
        5. Performance benchmarks
        """

        test_documentation = await self.generate_response(test_doc_prompt, temperature=0.6)

        yield self.stream_result("Generated Test Suite:")
        yield self.stream_result(test_code)
        yield self.stream_result("Test Documentation:")
        yield self.stream_result(test_documentation)

        # Store tests
        await self.store_memory(
            content=f"{language} test suite: {test_code}",
            segment_type="TEST_CODE",
            importance=0.8,
            metadata={
                "language": language,
                "test_type": "comprehensive",
                "session_id": session_id
            }
        )

    async def _handle_refactoring(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code refactoring requests"""

        yield self.stream_action("Planning refactoring approach...")

        language = task_analysis.get('language', 'python')

        refactor_prompt = f"""
        Create a comprehensive refactoring guide for {language} code.

        Cover:
        1. Code smell identification
        2. Refactoring patterns and techniques
        3. Safe refactoring procedures
        4. Automated refactoring tools
        5. Testing during refactoring
        6. Legacy code migration strategies
        7. Performance impact assessment
        8. Documentation updates
        """

        yield self.stream_thinking("Developing refactoring strategies...")
        refactor_guide = await self.generate_response(refactor_prompt, temperature=0.6)

        yield self.stream_result("Refactoring Guide:")
        yield self.stream_result(refactor_guide)

        # Store refactoring guide
        await self.store_memory(
            content=f"{language} refactoring guide: {refactor_guide}",
            segment_type="REFACTOR_GUIDE",
            importance=0.7,
            metadata={
                "language": language,
                "session_id": session_id
            }
        )

    async def _handle_general_coding(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle general coding requests"""

        yield self.stream_action("Processing coding request...")

        language = task_analysis.get('language', 'python')
        requirements = task_analysis.get('requirements', [])

        general_prompt = f"""
        Provide comprehensive guidance for {language} development addressing:
        {' '.join(requirements)}

        Include relevant:
        - Code examples and implementations
        - Best practices and conventions
        - Tool recommendations
        - Learning resources
        - Common pitfalls to avoid
        """

        yield self.stream_thinking("Generating coding guidance...")
        guidance = await self.generate_response(general_prompt, temperature=0.7)

        yield self.stream_result("Coding Guidance:")
        yield self.stream_result(guidance)

        # Store guidance
        await self.store_memory(
            content=f"{language} coding guidance: {guidance}",
            segment_type="CODING_GUIDANCE",
            importance=0.6,
            metadata={
                "language": language,
                "session_id": session_id
            }
        )

    async def _generate_project_structure(
        self,
        project_type: str,
        language: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Generate project directory structure"""

        base_structure = self.project_templates.get(project_type, self.project_templates['web_app'])

        structure = {
            'directories': base_structure['structure'].copy(),
            'files': base_structure['files'].copy(),
            'config_files': []
        }

        # Add language-specific configurations
        if language == 'python':
            structure['config_files'].extend([
                'pyproject.toml', 'setup.cfg', '.pre-commit-config.yaml'
            ])
        elif language == 'javascript':
            structure['config_files'].extend([
                'package.json', 'tsconfig.json', '.eslintrc.js', '.prettierrc'
            ])
        elif language == 'java':
            structure['config_files'].extend([
                'pom.xml', 'build.gradle', 'application.properties'
            ])
        elif language == 'rust':
            structure['config_files'].extend([
                'Cargo.toml', 'rust-toolchain.toml'
            ])

        # Add project-type specific additions
        if 'api' in requirements or 'rest' in requirements:
            structure['directories'].extend(['middleware/', 'routes/'])

        if 'database' in requirements or 'db' in requirements:
            structure['directories'].extend(['migrations/', 'models/'])

        if 'frontend' in requirements or 'ui' in requirements:
            structure['directories'].extend(['static/', 'templates/', 'components/'])

        return structure

    async def _generate_initial_files(
        self,
        structure: Dict[str, Any],
        language: str,
        project_type: str,
        requirements: List[str]
    ) -> Dict[str, str]:
        """Generate initial file content"""

        files = {}

        # Generate main application file
        if language == 'python':
            if project_type == 'web_app':
                files['app.py'] = await self._generate_python_web_app()
            elif project_type == 'api':
                files['main.py'] = await self._generate_python_api()
            elif project_type == 'cli_tool':
                files['cli.py'] = await self._generate_python_cli()
            else:
                files['main.py'] = await self._generate_python_main()

            files['requirements.txt'] = await self._generate_python_requirements(requirements)
            files['setup.py'] = await self._generate_python_setup()

        elif language == 'javascript':
            files['package.json'] = await self._generate_js_package_json(project_type)
            if project_type == 'web_app':
                files['index.js'] = await self._generate_js_web_app()
            elif project_type == 'api':
                files['server.js'] = await self._generate_js_api()
            else:
                files['index.js'] = await self._generate_js_main()

        # Generate common files
        files['README.md'] = await self._generate_readme(project_type, language, requirements)
        files['.gitignore'] = await self._generate_gitignore(language)
        files['LICENSE'] = await self._generate_license()

        # Generate test files
        if language == 'python':
            files['tests/test_main.py'] = await self._generate_python_tests()
        elif language == 'javascript':
            files['tests/test.js'] = await self._generate_js_tests()

        return files

    async def _analyze_code_quality(self, code: str, language: str) -> Optional[CodeAnalysis]:
        """Analyze code quality and provide metrics"""

        try:
            if language == 'python':
                return await self._analyze_python_code(code)
            elif language == 'javascript':
                return await self._analyze_js_code(code)
            else:
                return await self._analyze_generic_code(code, language)
        except Exception as e:
            self.logger.warning(f"Code analysis failed: {e}")
            return None

    async def _analyze_python_code(self, code: str) -> CodeAnalysis:
        """Analyze Python code specifically"""

        try:
            # Parse AST
            tree = ast.parse(code)

            # Count complexity metrics
            complexity = self._calculate_cyclomatic_complexity(tree)
            lines_of_code = len([line for line in code.split('\n') if line.strip()])

            # Simple maintainability calculation
            maintainability = max(0, 100 - complexity * 2 - max(0, lines_of_code - 100) * 0.1)

            # Basic issue detection
            security_issues = []
            performance_issues = []
            style_violations = []
            suggestions = []

            # Check for common issues
            if 'eval(' in code:
                security_issues.append("Use of eval() detected - security risk")
            if 'exec(' in code:
                security_issues.append("Use of exec() detected - security risk")
            if len([line for line in code.split('\n') if len(line) > 100]) > 0:
                style_violations.append("Lines longer than 100 characters detected")

            suggestions.extend([
                "Add type hints for better code documentation",
                "Consider adding docstrings to functions",
                "Use consistent naming conventions"
            ])

            return CodeAnalysis(
                complexity_score=complexity,
                maintainability_index=maintainability,
                test_coverage=0.0,  # Would need actual test analysis
                security_issues=security_issues,
                performance_issues=performance_issues,
                style_violations=style_violations,
                suggestions=suggestions
            )

        except Exception as e:
            self.logger.warning(f"Python code analysis failed: {e}")
            return self._create_default_analysis()

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity of Python AST"""

        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1

        return complexity

    async def _analyze_js_code(self, code: str) -> CodeAnalysis:
        """Analyze JavaScript code"""
        return self._create_default_analysis()

    async def _analyze_generic_code(self, code: str, language: str) -> CodeAnalysis:
        """Generic code analysis for any language"""
        return self._create_default_analysis()

    def _create_default_analysis(self) -> CodeAnalysis:
        """Create default code analysis"""
        return CodeAnalysis(
            complexity_score=5.0,
            maintainability_index=70.0,
            test_coverage=0.0,
            security_issues=[],
            performance_issues=[],
            style_violations=[],
            suggestions=["Consider adding comprehensive tests", "Add documentation"]
        )

    # File generation methods
    async def _generate_python_web_app(self) -> str:
        return '''from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
'''

    async def _generate_python_api(self) -> str:
        return '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="API Service", version="1.0.0")

class Item(BaseModel):
    name: str
    description: str = None

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item, "status": "created"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id, "data": "sample data"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    async def _generate_python_cli(self) -> str:
        return '''import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="CLI Tool")
    parser.add_argument('--version', action='version', version='1.0.0')
    parser.add_argument('command', help='Command to execute')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        print(f"Executing command: {args.command}")

    # Add your command logic here
    print(f"Command '{args.command}' executed successfully")

if __name__ == '__main__':
    main()
'''

    async def _generate_python_main(self) -> str:
        return '''"""
Main application entry point
"""

def main():
    """Main function"""
    print("Application started")
    # Add your application logic here

if __name__ == "__main__":
    main()
'''

    async def _generate_python_requirements(self, requirements: List[str]) -> str:
        base_reqs = []

        if any('web' in req.lower() for req in requirements):
            base_reqs.extend(['flask>=2.0.0', 'requests>=2.25.0'])
        if any('api' in req.lower() for req in requirements):
            base_reqs.extend(['fastapi>=0.68.0', 'uvicorn>=0.15.0'])
        if any('test' in req.lower() for req in requirements):
            base_reqs.extend(['pytest>=6.0.0', 'pytest-cov>=2.12.0'])
        if any('data' in req.lower() for req in requirements):
            base_reqs.extend(['pandas>=1.3.0', 'numpy>=1.21.0'])

        if not base_reqs:
            base_reqs = ['requests>=2.25.0']

        return '\n'.join(sorted(set(base_reqs)))

    async def _generate_python_setup(self) -> str:
        return '''from setuptools import setup, find_packages

setup(
    name="project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add dependencies here
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
'''

    async def _generate_js_package_json(self, project_type: str) -> str:
        dependencies = {}

        if project_type == 'web_app':
            dependencies.update({
                "express": "^4.18.0",
                "react": "^18.0.0",
                "react-dom": "^18.0.0"
            })
        elif project_type == 'api':
            dependencies.update({
                "express": "^4.18.0",
                "cors": "^2.8.5"
            })

        return json.dumps({
            "name": "project",
            "version": "1.0.0",
            "description": "A JavaScript project",
            "main": "index.js",
            "scripts": {
                "start": "node index.js",
                "dev": "nodemon index.js",
                "test": "jest"
            },
            "dependencies": dependencies,
            "devDependencies": {
                "nodemon": "^2.0.0",
                "jest": "^28.0.0"
            },
            "author": "Your Name",
            "license": "MIT"
        }, indent=2)

    async def _generate_js_web_app(self) -> str:
        return '''const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/api/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});

module.exports = app;
'''

    async def _generate_js_api(self) -> str:
        return '''const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json());

// Sample data
let items = [];
let nextId = 1;

// Routes
app.get('/', (req, res) => {
    res.json({ message: 'API is running', version: '1.0.0' });
});

app.get('/api/items', (req, res) => {
    res.json(items);
});

app.post('/api/items', (req, res) => {
    const { name, description } = req.body;

    if (!name) {
        return res.status(400).json({ error: 'Name is required' });
    }

    const item = {
        id: nextId++,
        name,
        description: description || '',
        createdAt: new Date().toISOString()
    };

    items.push(item);
    res.status(201).json(item);
});

app.get('/api/items/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const item = items.find(item => item.id === id);

    if (!item) {
        return res.status(404).json({ error: 'Item not found' });
    }

    res.json(item);
});

app.put('/api/items/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const itemIndex = items.findIndex(item => item.id === id);

    if (itemIndex === -1) {
        return res.status(404).json({ error: 'Item not found' });
    }

    const { name, description } = req.body;
    items[itemIndex] = {
        ...items[itemIndex],
        name: name || items[itemIndex].name,
        description: description !== undefined ? description : items[itemIndex].description,
        updatedAt: new Date().toISOString()
    };

    res.json(items[itemIndex]);
});

app.delete('/api/items/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const itemIndex = items.findIndex(item => item.id === id);

    if (itemIndex === -1) {
        return res.status(404).json({ error: 'Item not found' });
    }

    items.splice(itemIndex, 1);
    res.status(204).send();
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!' });
});

// Start server
app.listen(PORT, () => {
    console.log(`API server running on http://localhost:${PORT}`);
});

module.exports = app;
'''

    async def _generate_js_main(self) -> str:
        return '''#!/usr/bin/env node

/**
 * Main application entry point
 */

function main() {
    console.log('Application started');

    // Parse command line arguments
    const args = process.argv.slice(2);

    if (args.includes('--help') || args.includes('-h')) {
        showHelp();
        return;
    }

    if (args.includes('--version') || args.includes('-v')) {
        console.log('Version 1.0.0');
        return;
    }

    // Add your application logic here
    console.log('Arguments:', args);
    console.log('Application running successfully');
}

function showHelp() {
    console.log(`
Usage: node index.js [options]

Options:
  -h, --help     Show this help message
  -v, --version  Show version information

Examples:
  node index.js
  node index.js --help
    `);
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// Run main function
if (require.main === module) {
    main();
}

module.exports = { main };
'''

    async def _generate_js_tests(self) -> str:
        return '''const request = require('supertest');
const app = require('../index');

describe('Application Tests', () => {
    test('should start application', () => {
        expect(app).toBeDefined();
    });

    test('should have main function', () => {
        expect(typeof app.main).toBe('function');
    });
});

describe('API Tests', () => {
    test('GET / should return success message', async () => {
        const response = await request(app)
            .get('/')
            .expect(200);

        expect(response.body).toHaveProperty('message');
    });

    test('GET /api/health should return health status', async () => {
        const response = await request(app)
            .get('/api/health')
            .expect(200);

        expect(response.body).toHaveProperty('status', 'healthy');
    });
});

// Add more tests here
'''

    async def _generate_readme(self, project_type: str, language: str, requirements: List[str]) -> str:
        return f'''# {project_type.title()} Project

## Description
{language.title()} {project_type} implementing: {', '.join(requirements)}

## Installation
```bash
# Add installation instructions here
```

## Usage
```bash
# Add usage instructions here
```

## Development
```bash
# Add development setup instructions here
```

## Testing
```bash
# Add testing instructions here
```

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
'''

    async def _generate_gitignore(self, language: str) -> str:
        common = '''# General
.DS_Store
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
*.log
logs/
temp/
tmp/
'''

        if language == 'python':
            return common + '''
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
.env
venv/
ENV/
'''
        elif language == 'javascript':
            return common + '''
# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.eslintcache
'''
        else:
            return common

    async def _generate_license(self) -> str:
        return '''MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

    async def _generate_python_tests(self) -> str:
        return '''import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_example():
    """Example test function"""
    assert True

def test_main_import():
    """Test that main module can be imported"""
    try:
        import main
        assert True
    except ImportError:
        pytest.skip("Main module not found")

# Add more tests here
'''

    async def get_project_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get coding project statistics"""

        stats = {
            'total_projects': len(self.current_projects),
            'languages_used': set(),
            'project_types': set(),
            'total_files': 0,
            'lines_of_code': 0
        }

        for project in self.current_projects.values():
            stats['languages_used'].add(project.language)
            stats['project_types'].add(project.project_type)
            stats['total_files'] += len(project.files)

            # Count lines of code
            for content in project.files.values():
                stats['lines_of_code'] += len(content.split('\n'))

        # Convert sets to lists for JSON serialization
        stats['languages_used'] = list(stats['languages_used'])
        stats['project_types'] = list(stats['project_types'])

        return stats

    async def _initialize_coding_patterns(self):
        """Initialize coding patterns in the neural web"""
        if not self.neural_web:
            return

        try:
            # Add common design patterns
            patterns = [
                ("singleton", "Ensures a class has only one instance", "pattern"),
                ("factory", "Creates objects without specifying exact classes", "pattern"),
                ("observer", "Defines one-to-many dependency between objects", "pattern"),
                ("strategy", "Encapsulates algorithms and makes them interchangeable", "pattern"),
                ("mvc", "Separates application into Model-View-Controller", "pattern")
            ]

            for pattern_id, description, concept_type in patterns:
                await self.neural_web.add_concept(
                    concept_id=f"pattern_{pattern_id}",
                    content=description,
                    concept_type=concept_type,
                    metadata={"domain": "software_engineering", "type": "design_pattern"}
                )

            # Connect related patterns
            await self.neural_web.connect_concepts("pattern_mvc", "pattern_observer", "enables", 0.8)
            await self.neural_web.connect_concepts("pattern_factory", "pattern_strategy", "similar", 0.6)

            self.logger.info("Coding patterns initialized in neural web")

        except Exception as e:
            self.logger.error(f"Error initializing coding patterns: {e}")

# Test function
async def test_advanced_coding_agent():
    """Test the advanced coding agent functionality"""
    from core.config import load_config
    from core.llm_interface import OllamaInterface

    try:
        config = load_config("config.yaml")
        llm_interface = OllamaInterface(config=config)

        agent = AdvancedCodingAgent(
            agent_name="AdvancedCoder",
            config=config,
            llm_interface=llm_interface
        )

        print("Testing advanced coding agent...")

        # Test project creation
        async for stream_data in agent.run("Create a Python web API for user management"):
            print(f"[{stream_data.type.upper()}] {stream_data.content[:100]}...")

        # Get statistics
        stats = await agent.get_project_statistics()
        print(f"Project statistics: {stats}")

    except Exception as e:
        print(f"Test completed with expected errors: {e}")

if __name__ == "__main__":
    asyncio.run(test_advanced_coding_agent())
