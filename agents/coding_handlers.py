# agents/coding_handlers.py
"""Task handler mixins for the advanced coding agent."""

import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from agents.coding_models import CodeProject
from core.runtime_paths import workspace_subpath
from core.safe_code_editor import PROJECT_ROOT, extract_code_from_response, resolve_within_project, run_py_compile
from core.schemas import StreamData


class CodingHandlersMixin:
    """Mixin providing coding task handlers and project structure generation."""

    @staticmethod
    def _user_wants_workspace_write(user_input: str) -> bool:
        lowered = user_input.lower()
        if any(
            p in lowered
            for p in (
                "allowed file area",
                "workspace",
                "write to disk",
                "save to",
                "write to file",
                "on disk",
            )
        ):
            return True
        return "script" in lowered and any(
            v in lowered for v in ("create", "write", "generate", "make", "recreate")
        )

    @staticmethod
    def _workspace_slug_and_filename(user_input: str) -> tuple[str, str]:
        lowered = user_input.lower()
        if "pong" in lowered:
            return "pong", "pong.py"
        match = re.search(r"\b(\w+)\s+script\b", lowered)
        if match:
            slug = match.group(1)
            return slug, f"{slug}.py"
        return "generated_script", "main.py"

    async def _write_generated_script(
        self,
        user_input: str,
        generated_code: str,
        session_id: str,
    ) -> str | None:
        """Extract code from LLM output and write under var/workspace/<slug>/."""
        code = extract_code_from_response(generated_code)
        if not code.strip():
            return None

        slug, filename = self._workspace_slug_and_filename(user_input)
        rel_path = f"{workspace_subpath(slug)}/{filename}"
        resolved = resolve_within_project(rel_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(code, encoding="utf-8")

        if filename.endswith(".py"):
            ok, err = await run_py_compile(resolved)
            if not ok:
                resolved.unlink(missing_ok=True)
                raise RuntimeError(f"py_compile failed: {err}")

        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")

    async def _handle_fix_existing_file(
        self,
        file_path: str,
        user_input: str,
        session_id: str,
    ) -> AsyncGenerator[StreamData, None]:
        """Fix, debug, or refactor a real file the user named, through the
        same verify-before-commit pipeline the self-repair agent uses:
        write a candidate fix, run pytest, commit on success or revert to
        the original bytes on failure. Nothing broken is ever left in place.
        """
        fix_tool = self.tool_registry.get_tool("apply_code_fix") if self.tool_registry else None
        if fix_tool is None:
            yield self.stream_result(
                "apply_code_fix tool is unavailable — cannot safely edit an existing file."
            )
            return

        full_path = PROJECT_ROOT / file_path
        yield self.stream_action(f"Reading {file_path}...")
        original = full_path.read_text(encoding="utf-8", errors="replace")

        prompt = (
            "You are editing a real file in a software project per the user's request. "
            "Output ONLY the full, corrected file content inside a single ```python "
            "code fence — no prose, no explanation, no partial snippets. Preserve "
            "everything not related to the requested change.\n\n"
            f"File: {file_path}\n"
            f"User request: {user_input}\n\n"
            f"Current file content:\n```python\n{original}\n```\n"
        )
        yield self.stream_thinking(f"Drafting a change to {file_path}...")
        proposed = await self.generate_response(prompt, temperature=0.2, max_tokens=4000)
        new_content = extract_code_from_response(proposed)

        if new_content.strip() == original.strip():
            yield self.stream_result(
                "No change was needed — the file already satisfies the request."
            )
            return

        yield self.stream_action(
            f"Applying candidate change to {file_path} and verifying with tests..."
        )
        result = await fix_tool.execute(
            file_path=file_path, new_content=new_content, reason=user_input[:120]
        )

        if result["success"]:
            note = (
                f"committed as {result['commit_sha']}"
                if result.get("committed")
                else "applied but not committed"
            )
            yield self.stream_observation(f"Change verified — tests passed, {note}.")
            yield self.stream_result(f"Updated {file_path} as requested.")
            await self.store_memory(
                content=f"Coding agent edited {file_path}: {user_input[:200]}",
                segment_type="CODE_EDIT",
                importance=0.7,
                metadata={
                    "file": file_path,
                    "commit_sha": result.get("commit_sha"),
                    "session_id": session_id,
                },
            )
        else:
            tail = result.get("test_output", "")[-500:]
            yield self.stream_observation(
                f"Candidate change failed verification — reverted.\n{tail}"
            )
            yield self.stream_result(
                f"Could not safely apply that change to {file_path} — it failed tests and was reverted."
            )

    async def _write_project_files(
        self, project: "CodeProject", files: dict[str, str]
    ) -> list[str]:
        """Write generated project files to var/workspace/<project.name>/ and
        syntax-check each .py file with py_compile, so a broken scaffold is
        reported immediately instead of silently sitting in memory as text.
        """
        from core.safe_code_editor import resolve_within_project, run_py_compile

        results = []
        workspace_rel = workspace_subpath(project.name)
        for rel_name, content in files.items():
            file_rel = f"{workspace_rel}/{rel_name}"
            try:
                resolved = resolve_within_project(file_rel)
            except PermissionError as e:
                results.append(f"  ✗ {rel_name}: {e}")
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")

            if resolved.suffix == ".py":
                ok, output = await run_py_compile(resolved)
                status = "✓" if ok else "✗ syntax error"
                results.append(
                    f"  {status} {rel_name}" + ("" if ok else f": {output.strip()[-300:]}")
                )
            else:
                results.append(f"  ✓ {rel_name}")
        return results

    async def _handle_project_creation(
        self, task_analysis: dict[str, Any], session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle creation of new coding projects"""

        yield self.stream_action("Creating new coding project...")

        project_id = str(uuid.uuid4())
        language = task_analysis.get("language", "python")
        project_type = task_analysis.get("project_type", "web_app")
        requirements = task_analysis.get("requirements", [])

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
            description=" ".join(requirements),
            language=language,
            project_type=project_type,
            structure=project_structure,
            dependencies=task_analysis.get("frameworks", []),
            files=initial_files,
            tests={},
            documentation=architecture_response,
        )

        self.current_projects[project_id] = project

        yield self.stream_action(f"Writing {len(initial_files)} files to disk...")
        write_results = await self._write_project_files(project, initial_files)
        for line in write_results:
            yield self.stream_observation(line)

        await self.store_memory(
            content=f"Created {language} {project_type} project: {project.name}",
            segment_type="CODE_PROJECT",
            importance=0.9,
            metadata={
                "project_id": project_id,
                "language": language,
                "project_type": project_type,
                "session_id": session_id,
                "workspace_path": workspace_subpath(project.name),
            },
        )

        if self.neural_web:
            await self.neural_web.add_concept(
                f"project_{project_id}", f"{language} {project_type} project", "code_project"
            )

            await self.neural_web.add_concept(
                language,
                f"{language} programming language",
                "technology",
            )
            await self.neural_web.connect_concepts(f"project_{project_id}", language, "uses")

        yield self.stream_result(
            f"Created project '{project.name}' with {len(initial_files)} files "
            f"written to {workspace_subpath(project.name)}/"
        )
        yield self.stream_result("Project Architecture:")
        yield self.stream_result(architecture_response)

        yield self.stream_action("Generated project files:")
        for filename in initial_files.keys():
            yield self.stream_action(f"  - {filename}")

    async def _handle_code_generation(
        self,
        task_analysis: dict[str, Any],
        session_id: str,
        user_input: str = "",
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code generation requests"""

        yield self.stream_action("Generating code...")

        language = task_analysis.get("language", "python")
        requirements = task_analysis.get("requirements", [])
        complexity = task_analysis.get("complexity", "medium")
        write_to_workspace = bool(user_input and self._user_wants_workspace_write(user_input))

        if write_to_workspace:
            code_prompt = f"""
Write complete runnable {language} code for this request (code only inside one ```python fence):
{user_input}

Requirements:
- Under 180 lines; use the stdlib `turtle` module if this is a game
- No prose outside the fence
- Runnable as a single script
"""
        else:
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
        code_model = self.model_router.route(
            user_input or " ".join(requirements),
            default=self.get_model_name(),
            allow_trivial=False,
        )
        try:
            generated_code = await self.generate_response(
                code_prompt,
                model_name=code_model,
                temperature=0.3 if write_to_workspace else 0.7,
                max_tokens=4096 if write_to_workspace else None,
            )
        except Exception as e:
            yield self.stream_error(f"Code generation failed: {e}")
            return

        if write_to_workspace and not extract_code_from_response(generated_code).strip():
            retry_prompt = (
                f"{code_prompt}\n\n"
                "Your previous reply had no Python code. Reply with ONLY one ```python fence "
                "containing the full script."
            )
            try:
                generated_code = await self.generate_response(
                    retry_prompt,
                    model_name=code_model,
                    temperature=0.2,
                    max_tokens=4096,
                )
            except Exception as e:
                yield self.stream_error(f"Code generation retry failed: {e}")
                return

        content_length = len(generated_code)
        for i in range(0, content_length, 500):
            if i > 0:
                lines = generated_code[:i].count("\n")
                yield self.stream_action(f"Generated {lines} lines of code...")

        written_path: str | None = None
        if write_to_workspace:
            yield self.stream_action("Writing generated code to workspace...")
            try:
                written_path = await self._write_generated_script(
                    user_input, generated_code, session_id
                )
            except Exception as e:
                yield self.stream_error(f"Failed to write script to workspace: {e}")

        analysis = await self._analyze_code_quality(generated_code, language)

        await self.store_memory(
            content=f"Generated {language} code: {generated_code}",
            segment_type="GENERATED_CODE",
            importance=0.8,
            metadata={
                "language": language,
                "lines_of_code": generated_code.count("\n"),
                "complexity": complexity,
                "session_id": session_id,
                "workspace_path": written_path,
            },
        )

        if written_path:
            yield self.stream_result(f"Wrote script to {written_path}")
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
        self, task_analysis: dict[str, Any], session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code analysis requests"""

        yield self.stream_action("Analyzing code...")

        language = task_analysis.get("language", "python")

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
            metadata={"language": language, "analysis_type": "framework", "session_id": session_id},
        )

    async def _handle_debugging(
        self, task_analysis: dict[str, Any], session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle debugging requests"""

        yield self.stream_action("Analyzing debugging approach...")

        language = task_analysis.get("language", "python")

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
            metadata={"language": language, "session_id": session_id},
        )

    async def _handle_optimization(
        self, task_analysis: dict[str, Any], session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code optimization requests"""

        yield self.stream_action("Generating optimization strategies...")

        language = task_analysis.get("language", "python")

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
            metadata={"language": language, "session_id": session_id},
        )

    async def _handle_test_generation(
        self, task_analysis: dict[str, Any], session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle test generation requests"""

        yield self.stream_action("Generating test suite...")

        language = task_analysis.get("language", "python")
        requirements = task_analysis.get("requirements", [])

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
            metadata={"language": language, "test_type": "comprehensive", "session_id": session_id},
        )

    async def _handle_refactoring(
        self, task_analysis: dict[str, Any], session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code refactoring requests"""

        yield self.stream_action("Planning refactoring approach...")

        language = task_analysis.get("language", "python")

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
            metadata={"language": language, "session_id": session_id},
        )

    async def _handle_general_coding(
        self, task_analysis: dict[str, Any], session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle general coding requests"""

        yield self.stream_action("Processing coding request...")

        language = task_analysis.get("language", "python")
        requirements = task_analysis.get("requirements", [])

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
            metadata={"language": language, "session_id": session_id},
        )

    async def _generate_project_structure(
        self, project_type: str, language: str, requirements: list[str]
    ) -> dict[str, Any]:
        """Generate project directory structure"""

        base_structure = self.project_templates.get(project_type, self.project_templates["web_app"])

        structure = {
            "directories": base_structure["structure"].copy(),
            "files": base_structure["files"].copy(),
            "config_files": [],
        }

        if language == "python":
            structure["config_files"].extend(
                ["pyproject.toml", "setup.cfg", ".pre-commit-config.yaml"]
            )
        elif language == "javascript":
            structure["config_files"].extend(
                ["package.json", "tsconfig.json", ".eslintrc.js", ".prettierrc"]
            )
        elif language == "java":
            structure["config_files"].extend(["pom.xml", "build.gradle", "application.properties"])
        elif language == "rust":
            structure["config_files"].extend(["Cargo.toml", "rust-toolchain.toml"])

        if "api" in requirements or "rest" in requirements:
            structure["directories"].extend(["middleware/", "routes/"])

        if "database" in requirements or "db" in requirements:
            structure["directories"].extend(["migrations/", "models/"])

        if "frontend" in requirements or "ui" in requirements:
            structure["directories"].extend(["static/", "templates/", "components/"])

        return structure

    async def _generate_initial_files(
        self, structure: dict[str, Any], language: str, project_type: str, requirements: list[str]
    ) -> dict[str, str]:
        """Generate initial file content"""

        files = {}

        if language == "python":
            if project_type == "web_app":
                files["app.py"] = await self._generate_python_web_app()
            elif project_type == "api":
                files["main.py"] = await self._generate_python_api()
            elif project_type == "cli_tool":
                files["cli.py"] = await self._generate_python_cli()
            else:
                files["main.py"] = await self._generate_python_main()

            files["requirements.txt"] = await self._generate_python_requirements(requirements)
            files["setup.py"] = await self._generate_python_setup()

        elif language == "javascript":
            files["package.json"] = await self._generate_js_package_json(project_type)
            if project_type == "web_app":
                files["index.js"] = await self._generate_js_web_app()
            elif project_type == "api":
                files["server.js"] = await self._generate_js_api()
            else:
                files["index.js"] = await self._generate_js_main()

        files["README.md"] = await self._generate_readme(project_type, language, requirements)
        files[".gitignore"] = await self._generate_gitignore(language)
        files["LICENSE"] = await self._generate_license()

        if language == "python":
            files["tests/test_main.py"] = await self._generate_python_tests()
        elif language == "javascript":
            files["tests/test.js"] = await self._generate_js_tests()

        return files
