"""
Enhanced LLM Interface with Model Reliability Support.
This module provides enhanced model selection and reliability tracking.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from .config import WitsV3Config
from .llm_interface import BaseLLMInterface, OllamaInterface
from .model_reliability import ModelReliabilityManager, get_model_reliability_manager

logger = logging.getLogger(__name__)


class ReliableOllamaInterface(OllamaInterface):
    """
    Enhanced OllamaInterface with model reliability and fallback support.
    """

    def __init__(self, config: WitsV3Config):
        super().__init__(config)
        self.reliability_manager: ModelReliabilityManager | None = None

        # Initialize model reliability manager
        try:
            self.reliability_manager = get_model_reliability_manager(config)
            # Start health monitoring in background
            if self.reliability_manager.ollama_settings.enable_health_monitoring:
                asyncio.create_task(self._start_health_monitoring())
        except Exception as e:
            logger.warning(f"Failed to initialize model reliability manager: {e}")

    async def _start_health_monitoring(self):
        """Start health monitoring with proper error handling."""
        try:
            if self.reliability_manager:
                await self.reliability_manager.start_health_monitoring()
        except Exception as e:
            logger.warning(f"Failed to start health monitoring: {e}")

    def _get_best_model(
        self, preferred_model: str | None = None, agent_type: str | None = None
    ) -> str:
        """
        Get the best available model considering health and fallbacks.

        Args:
            preferred_model: The preferred model to use
            agent_type: Type of agent requesting the model

        Returns:
            str: Best available model name
        """
        base_model = preferred_model or self.ollama_settings.default_model

        if self.reliability_manager:
            try:
                return self.reliability_manager.get_best_model(base_model, agent_type)
            except Exception as e:
                logger.warning(f"Failed to get best model from reliability manager: {e}")

        return base_model

    def _record_success(self, model_name: str, response_time: float):
        """Record successful operation."""
        if self.reliability_manager:
            try:
                self.reliability_manager.record_success(model_name, response_time)
            except Exception as e:
                logger.debug(f"Failed to record success: {e}")

    def _record_failure(
        self, model_name: str, error: Exception, context: dict[str, Any] | None = None
    ):
        """Record failed operation."""
        if self.reliability_manager:
            try:
                self.reliability_manager.record_failure(model_name, error, context)
            except Exception as e:
                logger.debug(f"Failed to record failure: {e}")

    async def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Generate text with model reliability tracking.

        Extra keyword args (e.g. `format="json"` for Ollama structured output)
        are forwarded to the base interface — this wrapper must not swallow
        params the base class understands, or callers like the orchestrator's
        JSON reasoning break with 'unexpected keyword argument'.
        """
        # Get the best available model
        effective_model = self._get_best_model(model, "generate")
        start_time = time.time()

        try:
            result = await super().generate_text(
                prompt=prompt,
                model=effective_model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                **kwargs,
            )

            # Record success
            response_time = time.time() - start_time
            self._record_success(effective_model, response_time)

            return result

        except Exception as e:
            # Record failure
            self._record_failure(
                effective_model,
                e,
                {"operation": "generate_text", "prompt_length": len(prompt) if prompt else 0},
            )
            raise

    async def stream_text(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream text with model reliability tracking. Extra kwargs forwarded."""
        # Get the best available model
        effective_model = self._get_best_model(model, "stream")
        start_time = time.time()
        success = False

        try:
            async for chunk in super().stream_text(
                prompt=prompt,
                model=effective_model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                **kwargs,
            ):
                yield chunk
                success = True  # At least one chunk was yielded

            # Record success if we got at least one chunk
            if success:
                response_time = time.time() - start_time
                self._record_success(effective_model, response_time)

        except Exception as e:
            # Record failure
            self._record_failure(
                effective_model,
                e,
                {
                    "operation": "stream_text",
                    "prompt_length": len(prompt) if prompt else 0,
                    "partial_success": success,
                },
            )
            raise

    async def get_embedding(self, text: str, model: str | None = None) -> list[float]:
        """Get embedding with model reliability tracking."""
        # Get the best available model
        effective_model = self._get_best_model(model, "embedding")
        start_time = time.time()

        try:
            result = await super().get_embedding(text=text, model=effective_model)

            # Record success
            response_time = time.time() - start_time
            self._record_success(effective_model, response_time)

            return result

        except Exception as e:
            # Record failure
            self._record_failure(
                effective_model,
                e,
                {"operation": "get_embedding", "text_length": len(text) if text else 0},
            )
            raise

    async def get_health_summary(self) -> dict[str, Any]:
        """Get model health summary."""
        if self.reliability_manager:
            return self.reliability_manager.get_health_summary()
        return {"error": "Model reliability manager not available"}

    async def reset_model_health(self, model_name: str):
        """Reset health status for a specific model."""
        if self.reliability_manager:
            self.reliability_manager.reset_model_health(model_name)

    async def shutdown(self) -> None:
        """Shutdown the interface and stop health monitoring."""
        if self.reliability_manager:
            try:
                await self.reliability_manager.stop_health_monitoring()
            except Exception as e:
                logger.warning(f"Failed to stop health monitoring: {e}")

        await super().shutdown()


def get_enhanced_llm_interface(config: WitsV3Config) -> BaseLLMInterface:
    """
    Get an enhanced LLM interface with model reliability support.

    Args:
        config: WitsV3 configuration

    Returns:
        Enhanced LLM interface
    """
    if config.llm_interface.default_provider == "adaptive":
        logger.warning(
            "llm_interface.default_provider 'adaptive' is deprecated — "
            "using ReliableOllamaInterface. See docs/archive/adaptive_llm/README.md."
        )
        return ReliableOllamaInterface(config)
    elif config.llm_interface.default_provider == "ollama":
        return ReliableOllamaInterface(config)
    else:
        # Fallback to standard interface
        from .llm_interface import get_llm_interface

        return get_llm_interface(config)
