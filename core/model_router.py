"""
Smart model routing for WitsV3.

Classifies the user's raw message (or orchestration goal) and picks the
Ollama model best suited to it:

- trivial  -> small fast model (greetings, small talk, one-liner questions)
- code     -> coder model (anything programming-shaped)
- complex  -> the default model (everything else)

The classifier is deliberately heuristic and cheap — no LLM call, no
embedding. It must only ever be fed the raw user text, never a full prompt
template: templates carry history and document inventories that would make
every request look "complex".
"""

import logging
import re
from typing import Literal, Optional

from .config import WitsV3Config

RouteClass = Literal["trivial", "code", "complex"]

# Programming-shaped requests. Word-boundary matched (the WCCA casual-chat
# heuristic was bitten by substring matching — see agents/wits_control_center_agent.py).
_CODE_PATTERN = re.compile(
    r"\b("
    r"python|javascript|typescript|java|rust|c\+\+|c#|sql|html|css|bash|powershell"
    r"|code|coding|program|programming|script|function|class|method|variable"
    r"|debug|debugging|bug|error|exception|traceback|stack ?trace|compile|syntax"
    r"|refactor|regex|algorithm|api|json|yaml|xml|unit ?tests?|pytest"
    r"|git|repo|repository|library|module|import|install|pip|npm"
    r")\b",
    re.IGNORECASE,
)
_CODE_FENCE = re.compile(r"```|^\s{4,}\S", re.MULTILINE)

# Signals that a short message still needs real work (tools, documents,
# multi-step reasoning) and should not go to the trivial model.
_TASK_PATTERN = re.compile(
    r"\b("
    r"search|find|look ?up|read|open|write|create|make|build|generate"
    r"|analyz\w*|summariz\w*|compare|explain|calculate|convert|translate"
    r"|plan\w*|organiz\w*|research|review|check|update|fix"
    r"|document|file|report|pdf|upload|ingest|remember|schedule|list"
    r")\b",
    re.IGNORECASE,
)


class ModelRouter:
    """Picks an Ollama model per request from config.model_routing."""

    def __init__(self, config: WitsV3Config):
        self.settings = config.model_routing
        self.logger = logging.getLogger("WitsV3.ModelRouter")

    def classify(self, text: str) -> RouteClass:
        """Classify raw user text as trivial, code, or complex."""
        text = (text or "").strip()
        if not text:
            return "trivial"

        if _CODE_FENCE.search(text) or _CODE_PATTERN.search(text):
            return "code"

        if len(text) <= self.settings.trivial_max_chars and not _TASK_PATTERN.search(text):
            return "trivial"

        return "complex"

    def route(
        self,
        text: str,
        default: Optional[str] = None,
        allow_trivial: bool = True,
    ) -> Optional[str]:
        """
        Pick the model for the given raw user text.

        Args:
            text: The raw user message or goal (never a full prompt template)
            default: Model to use for the "complex" class and when routing is
                disabled (falls back to complex_model if not given)
            allow_trivial: Set False for callers that need structured output
                (e.g. the ReAct loop) where the small model would do worse

        Returns:
            The model name to pass to the LLM interface
        """
        fallback = default or self.settings.complex_model
        if not self.settings.enabled:
            return fallback

        route_class = self.classify(text)
        if route_class == "code":
            model = self.settings.code_model
        elif route_class == "trivial" and allow_trivial:
            model = self.settings.trivial_model
        else:
            model = fallback

        if model != fallback:
            self.logger.info(f"Routed {route_class!r} request to model {model} (default was {fallback})")
        return model
