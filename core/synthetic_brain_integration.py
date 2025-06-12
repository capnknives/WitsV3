"""
Synthetic Brain Integration Module for WITS.

This module provides integration points between the new Synthetic Brain components
and the existing WITS system. It handles compatibility issues and provides
adapters for smooth integration.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union

from core.synthetic_brain_stubs import (
    StubLLMInterface,
    StubMemoryManager,
    StubWorkingMemory,
    StubKnowledgeGraph
)


def get_compatible_llm_interface(config=None):
    """
    Get an LLM interface that works with both old and new code.

    This function will attempt to load the enhanced LLM interface,
    but will fall back to a stub implementation if not available.
    """
    try:
        from core.enhanced_llm_interface import get_enhanced_llm_interface
        return get_enhanced_llm_interface(config)
    except (ImportError, TypeError):
        logging.warning("Could not load enhanced LLM interface, using stub")
        return StubLLMInterface()


def get_compatible_memory_manager(config=None):
    """
    Get a memory manager that works with both old and new code.
    """
    try:
        from core.memory_manager import MemoryManager
        if config:
            return MemoryManager(config=config)
        else:
            return MemoryManager()
    except (ImportError, TypeError):
        logging.warning("Could not load memory manager, using stub")
        return StubMemoryManager()


def get_compatible_knowledge_graph(config=None, llm_interface=None):
    """
    Get a knowledge graph that works with both old and new code.
    """
    try:
        from core.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph(config=config, llm_interface=llm_interface)
    except (ImportError, TypeError):
        try:
            from core.knowledge_graph import KnowledgeGraph
            return KnowledgeGraph()
        except (ImportError, TypeError):
            logging.warning("Could not load knowledge graph, using stub")
            return StubKnowledgeGraph()


def get_compatible_working_memory(config=None):
    """
    Get a working memory instance that works with both old and new code.
    """
    try:
        from core.working_memory import WorkingMemory
        if config:
            return WorkingMemory(config=config)
        else:
            return WorkingMemory()
    except (ImportError, TypeError):
        logging.warning("Could not load working memory, using stub")
        return StubWorkingMemory()


async def export_memory_safely(export_path: str) -> None:
    """
    Export memory safely, handling different possible implementations.
    """
    try:
        from core.memory_export import export_memory
        await export_memory(export_path)
    except ImportError:
        logging.warning(f"Memory export not available, nothing exported to {export_path}")


async def summarize_memory_safely(text: str) -> str:
    """
    Summarize memory safely, handling different possible implementations.
    """
    try:
        from core.memory_summarization import summarize_memory_segment
        return await summarize_memory_segment(text)
    except ImportError:
        logging.warning("Memory summarization not available, using simple truncation")
        return text[:200] + "..." if len(text) > 200 else text
