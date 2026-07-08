"""
Enhanced reasoning tool and engine (Neural Web).

Public entry: EnhancedReasoningTool (auto-discovered by tool_registry).
"""

import logging
from typing import Dict, List, Optional, Any

from core.base_tool import BaseTool, ToolResult
from core.config import WitsV3Config
from core.neural_web_core import NeuralWeb
from core.llm_interface import BaseLLMInterface, LLMMessage
from core.cross_domain_learning import CrossDomainLearning

from tools.enhanced_reasoning_models import (
    ReasoningType,
    ReasoningContext,
    ReasoningResult,
    _resolve_neural_web,
)
from tools.enhanced_reasoning_patterns import (
    DeductiveReasoning,
    InductiveReasoning,
    AnalogicalReasoning,
)

logger = logging.getLogger(__name__)

class EnhancedReasoningEngine:
    """Main engine for enhanced reasoning patterns."""

    def __init__(self, config: WitsV3Config, neural_web: NeuralWeb,
                 llm_interface: BaseLLMInterface, cross_domain_learning: Optional[CrossDomainLearning] = None):
        self.config = config
        self.neural_web = neural_web
        self.llm_interface = llm_interface
        self.cross_domain_learning = cross_domain_learning
        self.logger = logging.getLogger(__name__)

        # Initialize reasoning patterns
        self.reasoning_patterns = {
            ReasoningType.DEDUCTIVE: DeductiveReasoning(config, neural_web, llm_interface),
            ReasoningType.INDUCTIVE: InductiveReasoning(config, neural_web, llm_interface),
            ReasoningType.ANALOGICAL: AnalogicalReasoning(config, neural_web, llm_interface),
        }

    async def reason(self, goal: str, domain: str,
                   reasoning_types: Optional[List[ReasoningType]] = None,
                   **kwargs) -> List[ReasoningResult]:
        """
        Execute enhanced reasoning using multiple patterns.

        Args:
            goal: The reasoning goal or question
            domain: The knowledge domain
            reasoning_types: Specific reasoning types to use (default: all available)
            **kwargs: Additional reasoning parameters

        Returns:
            List of reasoning results from different patterns
        """
        try:
            # Create reasoning context
            context = ReasoningContext(
                domain=domain,
                goal=goal,
                constraints=kwargs.get('constraints', []),
                available_concepts=kwargs.get('available_concepts', []),
                confidence_threshold=kwargs.get('confidence_threshold', 0.5),
                reasoning_depth=kwargs.get('reasoning_depth', 3)
            )

            # Determine which reasoning patterns to use
            if reasoning_types is None:
                reasoning_types = list(self.reasoning_patterns.keys())

            # Execute reasoning patterns
            results = []
            for reasoning_type in reasoning_types:
                if reasoning_type in self.reasoning_patterns:
                    try:
                        result = await self.reasoning_patterns[reasoning_type].reason(context)
                        if result.confidence >= context.confidence_threshold:
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error in {reasoning_type} reasoning: {e}")

            # Sort results by confidence
            results.sort(key=lambda x: x.confidence, reverse=True)

            self.logger.info(f"Completed reasoning with {len(results)} valid results")
            return results

        except Exception as e:
            self.logger.error(f"Error in enhanced reasoning: {e}")
            return []

    async def synthesize_reasoning_results(self, results: List[ReasoningResult], goal: str) -> str:
        """Synthesize multiple reasoning results into a unified conclusion."""
        if not results:
            return f"No valid reasoning results found for: {goal}"

        if len(results) == 1:
            return f"{results[0].reasoning_type.value.title()} reasoning: {results[0].conclusion}"

        try:
            prompt = f"""
            Synthesize these reasoning results into a unified conclusion:

            {chr(10).join(f"- {r.reasoning_type.value.title()}: {r.conclusion} (confidence: {r.confidence:.2f})" for r in results)}

            Goal: {goal}

            Provide a coherent synthesis that integrates the different reasoning approaches.
            """

            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_interface.generate_response(messages=messages)
            return response.content.strip()

        except Exception as e:
            self.logger.error(f"Error synthesizing reasoning results: {e}")
            return f"Multiple reasoning approaches suggest: {results[0].conclusion}"


class EnhancedReasoningTool(BaseTool):
    """Tool for enhanced reasoning with domain-specific patterns.

    Dependencies (config, llm_interface, and — when the neural memory
    backend is active — a live NeuralWeb) are injected lazily via
    set_dependencies(), the same pattern document/web-search tools use.
    This lets the tool_registry auto-discover it (zero required
    constructor args) and wire it up after the real system is built.
    """

    def __init__(self):
        super().__init__(
            name="enhanced_reasoning",
            description="Apply enhanced reasoning patterns with domain-specific knowledge",
        )
        self.config: Optional[WitsV3Config] = None
        self.llm_interface: Optional[BaseLLMInterface] = None
        self._neural_web: Optional[NeuralWeb] = None

    def set_dependencies(self, config: WitsV3Config, llm_interface=None, memory_manager=None, **kwargs) -> None:
        """Wire in shared system dependencies (called by WitsV3System startup)."""
        self.config = config
        self.llm_interface = llm_interface
        self._neural_web = _resolve_neural_web(memory_manager)

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The reasoning goal or question to address"
                    },
                    "domain": {
                        "type": "string",
                        "description": "The knowledge domain (e.g., 'technology', 'science', 'business')"
                    },
                    "reasoning_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["deductive", "inductive", "analogical", "causal", "creative"]
                        },
                        "description": "Specific reasoning types to apply"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence threshold for results (0.0-1.0)"
                    },
                    "synthesize_results": {
                        "type": "boolean",
                        "description": "Whether to synthesize multiple reasoning results"
                    }
                },
                "required": ["goal", "domain"]
            }
        }

    async def execute(self, **kwargs) -> ToolResult:
        try:
            if self.config is None or self.llm_interface is None:
                return ToolResult(
                    success=False,
                    result=None,
                    error="enhanced_reasoning tool has no dependencies wired (set_dependencies was never called)"
                )

            goal = kwargs.get("goal", "")
            domain = kwargs.get("domain", "general")
            reasoning_type_names = kwargs.get("reasoning_types", [])
            confidence_threshold = kwargs.get("confidence_threshold", 0.5)
            synthesize_results = kwargs.get("synthesize_results", True)

            if not goal.strip():
                return ToolResult(
                    success=False,
                    error="No reasoning goal provided"
                )

            # Convert reasoning type names to enums
            reasoning_types = []
            for name in reasoning_type_names:
                try:
                    reasoning_types.append(ReasoningType(name))
                except ValueError:
                    self.logger.warning(f"Unknown reasoning type: {name}")

            # Reason over the live Neural Web when the neural memory backend
            # is active; otherwise fall back to a scratch instance so the
            # tool still works for one-off reasoning over the given goal.
            neural_web = self._neural_web if self._neural_web is not None else NeuralWeb()
            reasoning_engine = EnhancedReasoningEngine(
                self.config, neural_web, self.llm_interface
            )

            # Execute reasoning
            results = await reasoning_engine.reason(
                goal=goal,
                domain=domain,
                reasoning_types=reasoning_types if reasoning_types else None,
                confidence_threshold=confidence_threshold
            )

            # Prepare response
            if not results:
                return ToolResult(
                    success=True,
                    result="No reasoning results met the confidence threshold",
                    metadata={"goal": goal, "domain": domain, "threshold": confidence_threshold}
                )

            # Synthesize results if requested
            if synthesize_results and len(results) > 1:
                synthesis = await reasoning_engine.synthesize_reasoning_results(results, goal)

                return ToolResult(
                    success=True,
                    result=synthesis,
                    metadata={
                        "goal": goal,
                        "domain": domain,
                        "reasoning_results": [
                            {
                                "type": r.reasoning_type.value,
                                "conclusion": r.conclusion,
                                "confidence": round(r.confidence, 3),
                                "supporting_evidence": len(r.supporting_evidence)
                            }
                            for r in results
                        ],
                        "synthesis": synthesis
                    }
                )
            else:
                # Return best single result
                best_result = results[0]
                return ToolResult(
                    success=True,
                    result=f"{best_result.reasoning_type.value.title()} reasoning: {best_result.conclusion}",
                    metadata={
                        "goal": goal,
                        "domain": domain,
                        "reasoning_type": best_result.reasoning_type.value,
                        "confidence": round(best_result.confidence, 3),
                        "reasoning_path": best_result.reasoning_path,
                        "supporting_evidence": best_result.supporting_evidence
                    }
                )

        except Exception as e:
            logger.error(f"Error in enhanced reasoning tool: {e}")
            return ToolResult(
                success=False,
                error=f"Error during enhanced reasoning: {str(e)}"
            )
