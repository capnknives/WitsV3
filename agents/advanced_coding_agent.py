"""
Advanced Coding Agent Compatibility Wrapper

This file provides backward compatibility by importing from the refactored
modular version in agents.coding module.
"""

# Import everything from the refactored module to maintain compatibility
from agents.coding import (
    AdvancedCodingAgent,
    CodeProject,
    CodeAnalysis,
    test_advanced_coding_agent
)

# Re-export for backward compatibility
__all__ = [
    'AdvancedCodingAgent',
    'CodeProject', 
    'CodeAnalysis',
    'test_advanced_coding_agent'
]

# Add compatibility note
def _compatibility_note():
    """
    Note: This file has been refactored into multiple modules for better maintainability.
    
    The functionality is now organized as follows:
    - agents/coding/models.py - Data models (CodeProject, CodeAnalysis)
    - agents/coding/project_manager.py - Project creation and management
    - agents/coding/code_generator.py - Code generation logic
    - agents/coding/code_analyzer.py - Code analysis functionality
    - agents/coding/template_generator.py - File template generation
    - agents/coding/language_handlers.py - Language-specific code generation
    - agents/coding/debugging_assistant.py - Debugging and optimization
    - agents/coding/test_generator.py - Test generation
    - agents/coding/advanced_coding_agent.py - Main agent coordinating all modules
    
    This file now serves as a compatibility wrapper to maintain existing imports.
    """
    pass


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_advanced_coding_agent())
