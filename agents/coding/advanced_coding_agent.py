"""
Advanced Coding Agent that generates complete projects, analyzes code, and provides expert programming assistance
"""

import asyncio
import re
import json
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List
from datetime import datetime

from agents.base_agent import BaseAgent, ConversationHistory
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData
from core.neural_web import NeuralWeb

from .models import CodeProject, CodeAnalysis
from .project_manager import ProjectManager
from .code_generator import CodeGenerator
from .code_analyzer import CodeAnalyzer
from .template_generator import TemplateGenerator
from .language_handlers import get_language_handler
from .debugging_assistant import DebuggingAssistant
from .test_generator import TestGenerator
from .handlers import CodingTaskHandlers


class AdvancedCodingAgent(BaseAgent):
    """Advanced agent specialized in coding, project generation, and code analysis"""
    
    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: Optional[MemoryManager] = None,
        neural_web: Optional[NeuralWeb] = None,
        tool_registry: Optional[Any] = None
    ):
        super().__init__(
            agent_name=agent_name,
            config=config,
            llm_interface=llm_interface,
            memory_manager=memory_manager,
            tool_registry=tool_registry
        )
        
        self.neural_web = neural_web
        
        # Initialize specialized components
        self.project_manager = ProjectManager()
        self.code_generator = CodeGenerator(self)
        self.code_analyzer = CodeAnalyzer(self)
        self.template_generator = TemplateGenerator()
        self.debugging_assistant = DebuggingAssistant(self)
        self.test_generator = TestGenerator(self)
        
        # Initialize task handlers
        self.task_handlers = CodingTaskHandlers(self)
        
        # Initialize coding patterns in neural web if available
        if self.neural_web:
            asyncio.create_task(self._initialize_coding_patterns())
    
    async def run(
        self,
        user_input: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """Process coding-related requests"""
        
        yield self.stream_thinking("Analyzing coding request...")
        
        # Analyze the task type
        task_analysis = await self._analyze_coding_task(user_input)
        
        yield self.stream_observation(
            f"Task type: {task_analysis['task_type']}, "
            f"Language: {task_analysis['language']}"
        )
        
        # Route to appropriate handler
        handlers = {
            'create_project': self.task_handlers.handle_project_creation,
            'write_code': self.task_handlers.handle_code_generation,
            'analyze_code': self.task_handlers.handle_code_analysis,
            'debug_code': self.task_handlers.handle_debugging,
            'optimize_code': self.task_handlers.handle_optimization,
            'write_tests': self.task_handlers.handle_test_generation,
            'refactor_code': self.task_handlers.handle_refactoring,
            'general_coding': self.task_handlers.handle_general_coding
        }
        
        handler = handlers.get(task_analysis['task_type'], self.task_handlers.handle_general_coding)
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