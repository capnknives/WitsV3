"""
Advanced Coding Agent for WitsV3
Refactored version that uses modular components
"""

import asyncio
import json
import re
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator

from agents.base_agent import BaseAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, ConversationHistory
from core.neural_web_core import NeuralWeb

from .models import CodeProject, CodeAnalysis, SUPPORTED_LANGUAGES
from .project_manager import ProjectManager
from .code_generator import CodeGenerator
from .code_analyzer import CodeAnalyzer
from .debugging_assistant import DebuggingAssistant
from .test_generator import TestGenerator


class AdvancedCodingAgent(BaseAgent):
    """
    Advanced coding agent that orchestrates various coding capabilities
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
        
        # Initialize components
        self.project_manager = ProjectManager()
        self.code_generator = CodeGenerator(llm_interface)
        self.code_analyzer = CodeAnalyzer()
        self.debugging_assistant = DebuggingAssistant(llm_interface)
        self.test_generator = TestGenerator(llm_interface)
        
        # Neural web integration
        if self.neural_web:
            self.enable_code_intelligence = True
            self.logger.info("Neural web enabled for code intelligence")
            asyncio.create_task(self._initialize_coding_patterns())
        else:
            self.enable_code_intelligence = False
        
        self.logger.info("Advanced Coding Agent initialized with modular architecture")
    
    async def run(
        self,
        user_input: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """Process coding requests"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        yield self.stream_thinking("Analyzing coding request...")
        
        # Parse the request to understand the coding task
        task_analysis = await self._analyze_coding_task(user_input)
        
        yield self.stream_thinking(f"Identified task: {task_analysis['task_type']}")
        
        # Route to appropriate handler
        handlers = {
            'create_project': self._handle_project_creation,
            'write_code': self._handle_code_generation,
            'analyze_code': self._handle_code_analysis,
            'debug_code': self._handle_debugging,
            'optimize_code': self._handle_optimization,
            'write_tests': self._handle_test_generation,
            'refactor_code': self._handle_refactoring,
            'general_coding': self._handle_general_coding
        }
        
        handler = handlers.get(task_analysis['task_type'], self._handle_general_coding)
        async for stream in handler(task_analysis, session_id):
            yield stream
    
    async def _analyze_coding_task(self, user_input: str) -> Dict[str, Any]:
        """Analyze the coding request to determine task type and parameters"""
        
        analysis_prompt = f"""
        Analyze this coding request and determine the task type and parameters:
        
        Request: {user_input}
        
        Respond with JSON containing:
        {{
            "task_type": "create_project" | "write_code" | "analyze_code" | "debug_code" | "optimize_code" | "write_tests" | "refactor_code" | "general_coding",
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
            "requirements": [user_input[:100]],
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
        
        language = task_analysis.get('language', 'python')
        project_type = task_analysis.get('project_type', 'web_app')
        requirements = task_analysis.get('requirements', [])
        
        # Verify language is supported
        if language not in SUPPORTED_LANGUAGES:
            yield self.stream_error(f"Language '{language}' is not yet supported")
            return
        
        # Generate project architecture
        yield self.stream_thinking("Designing project architecture...")
        
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
        
        architecture_response = await self.generate_response(architecture_prompt, temperature=0.6)
        
        # Create project using project manager
        project = await self.project_manager.create_project(
            language=language,
            project_type=project_type,
            requirements=requirements
        )
        
        # Store project in memory
        await self.store_memory(
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
        if self.neural_web:
            await self._add_project_to_neural_web(project)
        
        yield self.stream_result(f"Created project '{project.name}' with {project.get_file_count()} files")
        yield self.stream_result("Project Architecture:")
        yield self.stream_result(architecture_response)
        
        # Show file structure
        yield self.stream_action("Generated project files:")
        for filename in sorted(project.files.keys()):
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
        
        # Generate code using code generator
        generated_code = await self.code_generator.generate_code(
            language=language,
            requirements=requirements,
            complexity=complexity,
            context=task_analysis.get('parameters', {})
        )
        
        yield self.stream_thinking("Analyzing generated code quality...")
        
        # Analyze the generated code
        analysis = await self.code_analyzer.analyze_code(generated_code, language)
        
        # Store code in memory
        await self.store_memory(
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
        
        yield self.stream_result("Generated Code:")
        yield self.stream_result(generated_code)
        
        if analysis:
            yield self.stream_observation(f"Code quality score: {analysis.get_overall_score():.1f}/100")
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
        
        yield self.stream_thinking("Performing code analysis...")
        analysis_result = await self.generate_response(analysis_prompt, temperature=0.5)
        
        yield self.stream_result("Code Analysis Framework:")
        yield self.stream_result(analysis_result)
        
        # Store analysis framework
        await self.store_memory(
            content=f"{language} code analysis framework",
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
        
        yield self.stream_action("Creating debugging guide...")
        
        language = task_analysis.get('language', 'python')
        
        # Generate debugging guide using debugging assistant
        debug_guide = await self.debugging_assistant.create_debugging_guide(language)
        
        yield self.stream_result("Debugging Guide:")
        yield self.stream_result(debug_guide)
        
        # Store debugging guide
        await self.store_memory(
            content=f"{language} debugging guide",
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
        
        yield self.stream_thinking("Generating optimization guide...")
        optimization_guide = await self.generate_response(optimization_prompt, temperature=0.6)
        
        yield self.stream_result("Optimization Guide:")
        yield self.stream_result(optimization_guide)
    
    async def _handle_test_generation(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle test generation requests"""
        
        yield self.stream_action("Generating test strategy...")
        
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
            yield self.stream_result("Generated Tests:")
            yield self.stream_result(tests)
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
            
            testing_guide = await self.generate_response(testing_prompt, temperature=0.6)
            
            yield self.stream_result("Testing Guide:")
            yield self.stream_result(testing_guide)
    
    async def _handle_refactoring(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle code refactoring requests"""
        
        yield self.stream_action("Analyzing refactoring opportunities...")
        
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
        
        yield self.stream_thinking("Generating refactoring guide...")
        refactoring_guide = await self.generate_response(refactoring_prompt, temperature=0.6)
        
        yield self.stream_result("Refactoring Guide:")
        yield self.stream_result(refactoring_guide)
    
    async def _handle_general_coding(
        self,
        task_analysis: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """Handle general coding questions"""
        
        yield self.stream_action("Processing general coding request...")
        
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
        
        response = await self.generate_response(general_prompt, temperature=0.7)
        
        yield self.stream_result(response)
    
    async def _add_project_to_neural_web(self, project: CodeProject):
        """Add project to neural web for future reference"""
        if not self.neural_web:
            return
        
        try:
            # Add project concept
            await self.neural_web.add_concept(
                f"project_{project.id}",
                f"{project.language} {project.project_type} project: {project.name}",
                "code_project"
            )
            
            # Add language concept if not exists
            await self.neural_web.add_concept(
                project.language,
                f"{project.language} programming language",
                "technology"
            )
            
            # Connect project to language
            await self.neural_web.connect_concepts(
                f"project_{project.id}",
                project.language,
                "uses",
                0.9
            )
            
            # Add framework connections
            for dep in project.dependencies:
                await self.neural_web.add_concept(
                    dep,
                    f"{dep} framework/library",
                    "technology"
                )
                await self.neural_web.connect_concepts(
                    f"project_{project.id}",
                    dep,
                    "depends_on",
                    0.8
                )
        
        except Exception as e:
            self.logger.error(f"Error adding project to neural web: {e}")
    
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
                ("mvc", "Separates application into Model-View-Controller", "pattern"),
                ("repository", "Encapsulates data access logic", "pattern"),
                ("decorator", "Adds new functionality to objects dynamically", "pattern")
            ]
            
            for pattern_id, description, concept_type in patterns:
                await self.neural_web.add_concept(
                    concept_id=f"pattern_{pattern_id}",
                    content=description,
                    concept_type=concept_type,
                    metadata={"domain": "software_engineering", "type": "design_pattern"}
                )
            
            # Connect related patterns
            await self.neural_web.connect_concepts("pattern_mvc", "pattern_observer", "often_used_with", 0.8)
            await self.neural_web.connect_concepts("pattern_repository", "pattern_factory", "complements", 0.7)
            await self.neural_web.connect_concepts("pattern_strategy", "pattern_factory", "similar_purpose", 0.6)
            
            self.logger.info("Coding patterns initialized in neural web")
            
        except Exception as e:
            self.logger.error(f"Error initializing coding patterns: {e}")
    
    async def get_project_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get coding project statistics"""
        return self.project_manager.get_project_statistics()


# Test function
async def test_advanced_coding_agent():
    """Test the refactored advanced coding agent"""
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
        
        print("Testing refactored advanced coding agent...")
        
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