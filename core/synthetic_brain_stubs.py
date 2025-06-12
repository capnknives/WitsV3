"""
Stub implementations for Synthetic Brain components to enable testing.

This module provides simplified stub implementations of core classes and interfaces
needed for testing the synthetic brain components without depending on the
actual implementations.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

# Stub LLM Interface
class StubLLMInterface:
    """Stub implementation of LLM interface for testing"""
    async def generate_text(self, prompt: str) -> str:
        """Generate text from a prompt"""
        return f"Stub response for: {prompt[:30]}..."

    async def generate_with_context(self, prompt: str, context: List[Dict[str, Any]]) -> str:
        """Generate text with context"""
        return f"Stub response with context for: {prompt[:30]}..."

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {"model": "stub-model", "parameters": 1000000}


# Stub Knowledge Graph
class StubKnowledgeGraph:
    """Stub implementation of Knowledge Graph for testing"""
    def __init__(self, **kwargs):
        self.concepts = {
            "concept1": {"description": "A test concept", "connections": ["concept2"]},
            "concept2": {"description": "Another test concept", "connections": ["concept1"]},
            "ai": {"description": "Artificial Intelligence", "connections": ["ml"]},
            "ml": {"description": "Machine Learning", "connections": ["ai"]},
        }

    def get_active_concepts(self) -> List[str]:
        """Get list of active concepts"""
        return ["concept1", "concept2"]

    def search_concepts(self, query: str, limit: int = 5):
        """Search for concepts related to a query"""
        return [
            type('Concept', (), {
                'id': 'concept1',
                'description': 'A test concept',
                'relevance': 0.9
            })
        ]

    def add_concept(self, concept: str, description: str) -> None:
        """Add a new concept"""
        self.concepts[concept] = {"description": description, "connections": []}


# Stub Memory Manager
class StubMemoryManager:
    """Stub implementation of Memory Manager for testing"""
    async def store(self, key: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a memory"""
        pass

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for memories"""
        return [
            {
                "key": "episodic:123",
                "content": "Test memory content",
                "metadata": {"source": "test"},
                "relevance": 0.9
            }
        ]

    async def delete(self, key: str) -> None:
        """Delete a memory"""
        pass

    async def update(self, key: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update a memory"""
        pass


# Stub Working Memory
class StubWorkingMemory:
    """Stub implementation of Working Memory for testing"""
    def __init__(self):
        self.data = {"key": "value", "focus": "test topic"}

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current working memory state"""
        return self.data.copy()

    def update(self, key: str, value: Any) -> None:
        """Update a value in working memory"""
        self.data[key] = value

    def clear(self) -> None:
        """Clear working memory"""
        self.data.clear()


# Stub functions
async def stub_export_memory(path: str) -> None:
    """Stub implementation of memory export"""
    pass

async def stub_summarize_memory_segment(text: str) -> str:
    """Stub implementation of memory summarization"""
    return f"Summary of: {text[:50]}..."

# Function to get stub LLM interface
def get_stub_llm_interface(*args, **kwargs) -> StubLLMInterface:
    """Get a stub LLM interface instance"""
    return StubLLMInterface()
