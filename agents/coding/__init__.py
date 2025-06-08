"""
WitsV3 Advanced Coding Module

This module provides advanced coding capabilities including project creation,
code generation, analysis, debugging, and testing across multiple languages.
"""

from .models import CodeProject, CodeAnalysis
from .project_manager import ProjectManager
from .code_generator import CodeGenerator
from .code_analyzer import CodeAnalyzer
from .template_generator import TemplateGenerator
from .language_handlers import LanguageHandlers
from .debugging_assistant import DebuggingAssistant
from .test_generator import TestGenerator
from .advanced_coding_agent import AdvancedCodingAgent

__all__ = [
    # Models
    'CodeProject',
    'CodeAnalysis',
    
    # Managers
    'ProjectManager',
    'CodeGenerator',
    'CodeAnalyzer',
    'TemplateGenerator',
    'LanguageHandlers',
    'DebuggingAssistant',
    'TestGenerator',
    
    # Main Agent
    'AdvancedCodingAgent',
]