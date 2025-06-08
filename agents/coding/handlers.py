"""
Task-specific handlers for the Advanced Coding Agent
"""

import re
import json
import logging
from typing import Dict, Any, Optional, AsyncGenerator

from core.schemas import StreamData
from .models import CodeProject
from .project_manager import ProjectManager
from .code_generator import CodeGenerator
from .code_analyzer import CodeAnalyzer
from .debugging_assistant import DebuggingAssistant
from .test_generator import TestGenerator


class CodingTaskHandlers:
    """Handles different types of coding tasks for the Advanced Coding Agent"""
    
    def __init__(self, agent):
        """Initialize handlers with reference to parent agent"""
        self.agent = agent
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Get references to agent's tools
        self.project_manager = agent.project_manager
        self.code_generator = agent.code_generator
        self.code_analyzer = agent.code_analyzer
        self.debugging_assistant = agent.debugging_assistant
        self.test_generator = agent.test_generator
    
    async def handle_project_creation(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle creation of new coding projects"""
        
        yield self.agent.stream_action("Creating new coding project...")
        
        language = task_analysis.get('language', 'python')
        project_type = task_analysis.get('project_type', 'web_app')
        requirements = task_analysis.get('requirements', [])
        
        # Verify language is supported
        supported_languages = ["python", "javascript", "java", "rust", "go", "ruby", "php", "c++", "c#"]
        if language not in supported_languages:
            yield self.agent.stream_error(f"Language '{language}' is not yet supported")
            return
        
        # Generate project architecture
        yield self.agent.stream_thinking("Designing project architecture...")
        
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
        """
        
        architecture_response = await self.agent.generate_response(architecture_prompt, temperature=0.6)
        
        # Create project using project manager
        project = await self.project_manager.create_project(
            language=language,
            project_type=project_type,
            requirements=requirements
        )
        
        # Store project in memory
        await self.agent.store_memory(
            content=f"Created {language} {project_type} project: {project.name}",
            segment_type="CODE_PROJECT",
            importance=0.9,
            metadata={
                "project_id": project.id,
                "language": language,
                "project_type": project_type,
                "session_id": session_id
            }
        )
        
        # Add to neural web if enabled
        if self.agent.neural_web:
            await self._add_project_to_neural_web(project)
        
        yield self.agent.stream_result(f"Created project '{project.name}' with {project.get_file_count()} files")
        yield self.agent.stream_result("Project Architecture:")
        yield self.agent.stream_result(architecture_response)
        
        # Show file structure
        yield self.agent.stream_action("Generated project files:")
        for filename in sorted(project.files.keys()):
            yield self.agent.stream_action(f"  - {filename}")
    
    async def handle_code_generation(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code generation requests"""
        
        yield self.agent.stream_action("Generating code...")
        
        language = task_analysis.get('language', 'python')
        requirements = task_analysis.get('requirements', [])
        complexity = task_analysis.get('complexity', 'medium')
        
        # Generate code using code generator
        generated_code = await self.code_generator.generate_code(
            language=language,
            requirements=requirements,
            complexity=complexity,
            context=task_analysis.get('parameters', {})
        )
        
        yield self.agent.stream_thinking("Analyzing generated code quality...")
        
        # Analyze the generated code
        analysis = await self.code_analyzer.analyze_code(generated_code, language)
        
        # Store code in memory
        await self.agent.store_memory(
            content=f"Generated {language} code: {generated_code[:500]}...",
            segment_type="GENERATED_CODE",
            importance=0.8,
            metadata={
                "language": language,
                "lines_of_code": len(generated_code.split('\n')),
                "complexity": complexity,
                "session_id": session_id
            }
        )
        
        yield self.agent.stream_result("Generated Code:")
        yield self.agent.stream_result(generated_code)
        
        if analysis:
            yield self.agent.stream_observation(f"Code quality score: {analysis.get_overall_score():.1f}/100")
            if analysis.suggestions:
                yield self.agent.stream_observation("Suggestions for improvement:")
                for suggestion in analysis.suggestions[:3]:
                    yield self.agent.stream_observation(f"  - {suggestion}")
    
    async def handle_code_analysis(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code analysis requests"""
        
        yield self.agent.stream_action("Analyzing code...")
        
        language = task_analysis.get('language', 'python')
        
        # Generate comprehensive analysis framework
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
        
        yield self.agent.stream_thinking("Performing code analysis...")
        analysis_result = await self.agent.generate_response(analysis_prompt, temperature=0.5)
        
        yield self.agent.stream_result("Code Analysis Framework:")
        yield self.agent.stream_result(analysis_result)
        
        # Store analysis framework
        await self.agent.store_memory(
            content=f"{language} code analysis framework",
            segment_type="CODE_ANALYSIS",
            importance=0.8,
            metadata={
                "language": language,
                "analysis_type": "framework",
                "session_id": session_id
            }
        )
    
    async def handle_debugging(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle debugging requests"""
        
        yield self.agent.stream_action("Creating debugging guide...")
        
        language = task_analysis.get('language', 'python')
        
        # Generate debugging guide using debugging assistant
        debug_guide = await self.debugging_assistant.create_debugging_guide(language)
        
        yield self.agent.stream_result("Debugging Guide:")
        yield self.agent.stream_result(debug_guide)
        
        # Store debugging guide
        await self.agent.store_memory(
            content=f"{language} debugging guide",
            segment_type="DEBUG_GUIDE",
            importance=0.7,
            metadata={
                "language": language,
                "session_id": session_id
            }
        )
    
    async def handle_optimization(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code optimization requests"""
        
        yield self.agent.stream_action("Generating optimization strategies...")
        
        language = task_analysis.get('language', 'python')
        
        optimization_prompt = f"""
        Create a comprehensive optimization guide for {language} applications.
        
        Cover:
        1. Performance profiling techniques
        2. Memory optimization strategies
        3. Algorithm optimization
        4. Caching strategies
        5. Parallel processing approaches
        6. Database query optimization
        7. Code-level optimizations
        8. Architecture improvements
        """
        
        yield self.agent.stream_thinking("Generating optimization guide...")
        optimization_guide = await self.agent.generate_response(optimization_prompt, temperature=0.6)
        
        yield self.agent.stream_result("Optimization Guide:")
        yield self.agent.stream_result(optimization_guide)
    
    async def handle_test_generation(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle test generation requests"""
        
        yield self.agent.stream_action("Generating test strategy...")
        
        language = task_analysis.get('language', 'python')
        requirements = task_analysis.get('requirements', [])
        
        # Check if there's code to test
        code_to_test = task_analysis.get('parameters', {}).get('code', '')
        
        if code_to_test:
            # Generate tests for specific code
            tests = await self.test_generator.generate_tests(
                code=code_to_test,
                language=language
            )
            yield self.agent.stream_result("Generated Tests:")
            yield self.agent.stream_result(tests)
        else:
            # Generate general testing guide
            testing_prompt = f"""
            Create a comprehensive testing guide for {language} applications.
            
            Cover:
            1. Unit testing best practices
            2. Integration testing strategies
            3. End-to-end testing
            4. Test-driven development (TDD)
            5. Mocking and stubbing
            6. Test coverage goals
            7. Continuous testing
            8. Performance testing
            """
            
            testing_guide = await self.agent.generate_response(testing_prompt, temperature=0.6)
            
            yield self.agent.stream_result("Testing Guide:")
            yield self.agent.stream_result(testing_guide)
    
    async def handle_refactoring(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code refactoring requests"""
        
        yield self.agent.stream_action("Analyzing refactoring opportunities...")
        
        language = task_analysis.get('language', 'python')
        
        refactoring_prompt = f"""
        Create a comprehensive refactoring guide for {language} code.
        
        Cover:
        1. Code smell detection
        2. Design pattern application
        3. SOLID principles
        4. DRY (Don't Repeat Yourself)
        5. Function/class extraction
        6. Naming improvements
        7. Complexity reduction
        8. Performance improvements
        9. Testability improvements
        10. Refactoring tools
        """
        
        yield self.agent.stream_thinking("Generating refactoring guide...")
        refactoring_guide = await self.agent.generate_response(refactoring_prompt, temperature=0.6)
        
        yield self.agent.stream_result("Refactoring Guide:")
        yield self.agent.stream_result(refactoring_guide)
    
    async def handle_general_coding(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle general coding questions"""
        
        yield self.agent.stream_action("Processing general coding request...")
        
        requirements = task_analysis.get('requirements', [])
        
        general_prompt = f"""
        Help with this coding request:
        {' '.join(requirements)}
        
        Provide comprehensive assistance including:
        1. Solution approach
        2. Code examples
        3. Best practices
        4. Common pitfalls to avoid
        5. Testing considerations
        6. Performance considerations
        """
        
        response = await self.agent.generate_response(general_prompt, temperature=0.7)
        
        yield self.agent.stream_result(response)
    
    async def _add_project_to_neural_web(self, project: CodeProject):
        """Add project to neural web for future reference"""
        if not self.agent.neural_web:
            return
        
        try:
            # Add project concept
            await self.agent.neural_web.add_concept(
                f"project_{project.id}",
                f"{project.language} {project.project_type} project: {project.name}",
                "code_project"
            )
            
            # Add language concept if not exists
            await self.agent.neural_web.add_concept(
                project.language,
                f"{project.language} programming language",
                "technology"
            )
            
            # Connect project to language
            await self.agent.neural_web.connect_concepts(
                f"project_{project.id}",
                project.language,
                "uses",
                0.9
            )
            
            # Add framework connections
            for dep in project.dependencies:
                await self.agent.neural_web.add_concept(
                    dep,
                    f"{dep} framework/library",
                    "technology"
                )
                await self.agent.neural_web.connect_concepts(
                    f"project_{project.id}",
                    dep,
                    "depends_on",
                    0.8
                )
        
        except Exception as e:
            self.logger.error(f"Error adding project to neural web: {e}")