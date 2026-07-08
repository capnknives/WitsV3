"""Types and helpers for enhanced reasoning (Neural Web)."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.neural_web_core import NeuralWeb

logger = logging.getLogger(__name__)


def _resolve_neural_web(memory_manager) -> Optional[NeuralWeb]:
    """Pull the live NeuralWeb out of memory_manager when the neural backend is active."""
    if memory_manager is None:
        return None
    try:
        from core.neural_memory_backend import NeuralMemoryBackend

        backend = getattr(memory_manager, "backend", None)
        if isinstance(backend, NeuralMemoryBackend):
            return backend.neural_web
    except ImportError:
        pass
    return None


class ReasoningType(Enum):
    """Types of reasoning patterns available."""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    ETHICAL = "ethical"
    CREATIVE = "creative"
    SYSTEMATIC = "systematic"


@dataclass
class ReasoningContext:
    """Context for reasoning operations."""

    domain: str
    goal: str
    constraints: List[str]
    available_concepts: List[str]
    confidence_threshold: float = 0.5
    reasoning_depth: int = 3
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""

    reasoning_type: ReasoningType
    conclusion: str
    confidence: float
    reasoning_path: List[str]
    supporting_evidence: List[str]
    assumptions: List[str]
    domain: str
    metadata: Optional[Dict[str, Any]] = None


class BaseReasoningPattern(ABC):
    """Abstract base class for reasoning patterns."""

    def __init__(
        self,
        config: WitsV3Config,
        neural_web: NeuralWeb,
        llm_interface: BaseLLMInterface,
    ):
        self.config = config
        self.neural_web = neural_web
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Execute the reasoning pattern."""
        pass

    @abstractmethod
    def get_reasoning_type(self) -> ReasoningType:
        """Get the type of reasoning this pattern implements."""
        pass

    async def _get_relevant_concepts(self, query: str, limit: int = 10) -> List[str]:
        """Get concepts relevant to the reasoning query."""
        try:
            relevant_concepts = await self.neural_web._find_relevant_concepts(query)
            return relevant_concepts[:limit] if relevant_concepts else []
        except Exception as e:
            self.logger.error(f"Error finding relevant concepts: {e}")
            return []

    async def _activate_concept_network(self, concept_ids: List[str]) -> Dict[str, float]:
        """Activate a network of concepts and return activation levels."""
        activations = {}
        for concept_id in concept_ids:
            try:
                activated = await self.neural_web.activate_concept(concept_id, 0.8)
                for activated_id in activated:
                    if activated_id in self.neural_web.concepts:
                        activations[activated_id] = self.neural_web.concepts[
                            activated_id
                        ].activation_level
            except Exception as e:
                self.logger.error(f"Error activating concept {concept_id}: {e}")

        return activations
