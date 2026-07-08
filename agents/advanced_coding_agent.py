# agents/advanced_coding_agent.py
"""
Advanced Coding Agent for WitsV3
Specialized agent for software development, code analysis, and project management
"""

import asyncio
import ast
import json
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from agents.base_agent import BaseAgent
from agents.coding_handlers import CodingHandlersMixin
from agents.coding_models import CodeAnalysis, CodeProject
from agents.coding_scaffolds import CodingScaffoldMixin
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, ConversationHistory
from core.neural_web_core import NeuralWeb


class AdvancedCodingAgent(CodingHandlersMixin, CodingScaffoldMixin, BaseAgent):
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
        user_input: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Process coding requests
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        # A request naming a real, existing project file is a stronger signal
        # than keyword-based task classification below — route it straight
        # through the verify-before-commit edit pipeline instead of the
        # generic (LLM-prose-only) debugging/refactoring handlers.
        from core.safe_code_editor import extract_file_mention

        mention = extract_file_mention(user_input)
        if mention:
            file_path, _line = mention
            yield self.stream_thinking(f"Request names an existing file: {file_path}")
            async for stream in self._handle_fix_existing_file(file_path, user_input, session_id):
                yield stream
            return

        yield self.stream_thinking("Analyzing coding request...")

        task_analysis = await self._analyze_coding_task(user_input)

        yield self.stream_thinking(f"Identified task: {task_analysis['task_type']}")

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
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            self.logger.warning(f"Failed to parse task analysis: {e}")

        return {
            "task_type": "general_coding",
            "language": "python",
            "project_type": "script",
            "complexity": "medium",
            "requirements": [request[:100]],
            "frameworks": [],
            "parameters": {}
        }

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
            tree = ast.parse(code)

            complexity = self._calculate_cyclomatic_complexity(tree)
            lines_of_code = len([line for line in code.split('\n') if line.strip()])

            maintainability = max(0, 100 - complexity * 2 - max(0, lines_of_code - 100) * 0.1)

            security_issues = []
            performance_issues = []
            style_violations = []
            suggestions = []

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
                test_coverage=0.0,
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

        complexity = 1

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

            for content in project.files.values():
                stats['lines_of_code'] += len(content.split('\n'))

        stats['languages_used'] = list(stats['languages_used'])
        stats['project_types'] = list(stats['project_types'])

        return stats

    async def _initialize_coding_patterns(self):
        """Initialize coding patterns in the neural web"""
        if not self.neural_web:
            return

        try:
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

        async for stream_data in agent.run("Create a Python web API for user management"):
            print(f"[{stream_data.type.upper()}] {stream_data.content[:100]}...")

        stats = await agent.get_project_statistics()
        print(f"Project statistics: {stats}")

    except Exception as e:
        print(f"Test completed with expected errors: {e}")


if __name__ == "__main__":
    asyncio.run(test_advanced_coding_agent())
