# agents/coding_handlers.py
"""Task handler mixins for the advanced coding agent."""

import uuid
from typing import Any, AsyncGenerator, Dict, List

from core.schemas import StreamData

from agents.coding_models import CodeProject


class CodingHandlersMixin:
    """Mixin providing coding task handlers and project structure generation."""

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

        project_structure = await self._generate_project_structure(
            project_type, language, requirements
        )

        initial_files = await self._generate_initial_files(
            project_structure, language, project_type, requirements
        )

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

        if self.neural_web:
            await self.neural_web.add_concept(
                f"project_{project_id}",
                f"{language} {project_type} project",
                "code_project"
            )

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
        yield self.stream_thinking("Generating code...")
        generated_code = await self.generate_response(code_prompt, temperature=0.7)

        content_length = len(generated_code)
        for i in range(0, content_length, 500):
            if i > 0:
                lines = generated_code[:i].count('\n')
                yield self.stream_action(f"Generated {lines} lines of code...")

        analysis = await self._analyze_code_quality(generated_code, language)

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

        files['README.md'] = await self._generate_readme(project_type, language, requirements)
        files['.gitignore'] = await self._generate_gitignore(language)
        files['LICENSE'] = await self._generate_license()

        if language == 'python':
            files['tests/test_main.py'] = await self._generate_python_tests()
        elif language == 'javascript':
            files['tests/test.js'] = await self._generate_js_tests()

        return files
