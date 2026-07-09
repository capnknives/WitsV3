"""Factory for MemoryManager backends (extracted from memory_manager.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface

if TYPE_CHECKING:
    from core.memory_manager import BaseMemoryBackend


def create_memory_backend(
    config: WitsV3Config, llm_interface: BaseLLMInterface
) -> BaseMemoryBackend:
    """Instantiate the configured memory backend."""
    backend_name = config.memory_manager.backend

    if backend_name == "basic":
        from core.memory_manager import BasicMemoryBackend

        return BasicMemoryBackend(config, llm_interface)
    if backend_name == "faiss_cpu":
        from core.faiss_memory_backend import FaissCPUMemoryBackend

        return FaissCPUMemoryBackend(config, llm_interface)
    if backend_name == "faiss_gpu":
        from core.faiss_memory_backend import FaissGPUMemoryBackend

        return FaissGPUMemoryBackend(config, llm_interface)
    if backend_name == "neural":
        from core.neural_memory_backend import NeuralMemoryBackend

        return NeuralMemoryBackend(config, llm_interface)
    if backend_name in ("supabase", "supabase_neural"):
        from core.supabase_backend import SupabaseMemoryBackend

        return SupabaseMemoryBackend(config, llm_interface)

    raise ValueError(f"Unsupported memory backend: {backend_name}")
