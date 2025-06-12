"""
Model Reliability and Fallback System for WitsV3.
Handles model failures, automatic fallbacks, and health monitoring.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .config import WitsV3Config, OllamaSettings

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"
    FAILED = "failed"
    UNKNOWN = "unknown"

class FailureType(Enum):
    """Types of model failures."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    MODEL_NOT_FOUND = "model_not_found"
    MEMORY_ERROR = "memory_error"
    GENERATION_ERROR = "generation_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ModelFailure:
    """Record of a model failure."""
    model_name: str
    failure_type: FailureType
    error_message: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelHealth:
    """Health status of a model."""
    model_name: str
    status: ModelStatus
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    total_failures: int = 0
    total_requests: int = 0
    average_response_time: float = 0.0
    quarantine_until: Optional[datetime] = None
    failure_history: List[ModelFailure] = field(default_factory=list)

class ModelReliabilityManager:
    """
    Manages model reliability, health monitoring, and automatic fallbacks.
    """

    def __init__(self, config: WitsV3Config):
        """
        Initialize the model reliability manager.

        Args:
            config: WitsV3 configuration
        """
        self.config = config
        self.ollama_settings: OllamaSettings = config.ollama_settings
        self.logger = logging.getLogger("WitsV3.ModelReliability")

        # Model health tracking
        self.model_health: Dict[str, ModelHealth] = {}
        self.quarantined_models: Set[str] = set()

        # Model selection cache
        self._model_selection_cache: Dict[str, str] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: Dict[str, datetime] = {}

        # Health monitoring task
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

        # Initialize model health for configured models
        self._initialize_model_health()

        self.logger.info("Model reliability manager initialized")

    def _initialize_model_health(self):
        """Initialize health tracking for all configured models."""
        models_to_track = {
            self.ollama_settings.default_model,
            self.ollama_settings.control_center_model,
            self.ollama_settings.orchestrator_model,
            self.ollama_settings.embedding_model,
        }

        # Add fallback models
        models_to_track.update(self.ollama_settings.fallback_models)

        for model in models_to_track:
            if model not in self.model_health:
                self.model_health[model] = ModelHealth(
                    model_name=model,
                    status=ModelStatus.UNKNOWN
                )
                self.logger.debug(f"Initialized health tracking for model: {model}")

    async def start_health_monitoring(self):
        """Start continuous health monitoring."""
        if not self.ollama_settings.enable_health_monitoring:
            self.logger.info("Health monitoring disabled in configuration")
            return

        if self._is_monitoring:
            self.logger.warning("Health monitoring already running")
            return

        self._is_monitoring = True
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self.logger.info("Started model health monitoring")

    async def stop_health_monitoring(self):
        """Stop health monitoring."""
        self._is_monitoring = False
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped model health monitoring")

    async def _health_monitor_loop(self):
        """Main health monitoring loop."""
        while self._is_monitoring:
            try:
                await self._check_all_models_health()
                await self._update_quarantine_status()
                await asyncio.sleep(self.ollama_settings.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retrying

    async def _check_all_models_health(self):
        """Check health of all tracked models."""
        for model_name in self.model_health.keys():
            try:
                await self._check_model_health(model_name)
            except Exception as e:
                self.logger.error(f"Error checking health for model {model_name}: {e}")

    async def _check_model_health(self, model_name: str):
        """
        Check health of a specific model.

        Args:
            model_name: Name of the model to check
        """
        if model_name in self.quarantined_models:
            return  # Skip quarantined models

        # TODO: Implement actual health check by sending a small test request
        # For now, we'll just update the status based on recent failures
        health = self.model_health.get(model_name)
        if not health:
            return

        # Simple health assessment based on recent failures
        recent_failures = [
            f for f in health.failure_history
            if f.timestamp > datetime.now() - timedelta(minutes=10)
        ]

        if len(recent_failures) >= 3:
            health.status = ModelStatus.DEGRADED
        elif health.consecutive_failures >= self.ollama_settings.model_failure_threshold:
            health.status = ModelStatus.QUARANTINED
            self.quarantined_models.add(model_name)
            health.quarantine_until = datetime.now() + timedelta(
                seconds=self.ollama_settings.quarantine_duration
            )
            self.logger.warning(f"Model {model_name} quarantined due to repeated failures")
        elif health.consecutive_failures == 0 and len(recent_failures) == 0:
            health.status = ModelStatus.HEALTHY

    async def _update_quarantine_status(self):
        """Update quarantine status for models."""
        current_time = datetime.now()
        models_to_release = []

        for model_name in list(self.quarantined_models):
            health = self.model_health.get(model_name)
            if health and health.quarantine_until and current_time >= health.quarantine_until:
                models_to_release.append(model_name)

        for model_name in models_to_release:
            self.quarantined_models.remove(model_name)
            health = self.model_health[model_name]
            health.status = ModelStatus.UNKNOWN
            health.quarantine_until = None
            health.consecutive_failures = 0
            self.logger.info(f"Model {model_name} released from quarantine")

    def record_success(self, model_name: str, response_time: float):
        """
        Record a successful model operation.

        Args:
            model_name: Name of the model
            response_time: Response time in seconds
        """
        health = self._ensure_model_health(model_name)

        health.last_success = datetime.now()
        health.consecutive_failures = 0
        health.total_requests += 1

        # Update average response time
        if health.average_response_time == 0:
            health.average_response_time = response_time
        else:
            # Simple moving average
            health.average_response_time = (health.average_response_time + response_time) / 2

        # Update status if it was degraded
        if health.status in [ModelStatus.DEGRADED, ModelStatus.UNKNOWN]:
            health.status = ModelStatus.HEALTHY

        self.logger.debug(f"Recorded success for model {model_name} (response_time: {response_time:.2f}s)")

    def record_failure(
        self,
        model_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record a model failure.

        Args:
            model_name: Name of the model
            error: The error that occurred
            context: Additional context about the failure
        """
        health = self._ensure_model_health(model_name)

        # Classify the failure type
        failure_type = self._classify_failure(error)

        # Create failure record
        failure = ModelFailure(
            model_name=model_name,
            failure_type=failure_type,
            error_message=str(error),
            timestamp=datetime.now(),
            context=context or {}
        )

        # Update health metrics
        health.last_failure = datetime.now()
        health.consecutive_failures += 1
        health.total_failures += 1
        health.total_requests += 1
        health.failure_history.append(failure)

        # Trim failure history to last 100 entries
        if len(health.failure_history) > 100:
            health.failure_history = health.failure_history[-100:]

        # Update status
        if health.consecutive_failures >= self.ollama_settings.model_failure_threshold:
            health.status = ModelStatus.QUARANTINED
            self.quarantined_models.add(model_name)
            health.quarantine_until = datetime.now() + timedelta(
                seconds=self.ollama_settings.quarantine_duration
            )
        else:
            health.status = ModelStatus.DEGRADED

        self.logger.warning(
            f"Recorded failure for model {model_name}: {failure_type.value} - {str(error)[:100]}"
        )

    def _classify_failure(self, error: Exception) -> FailureType:
        """
        Classify the type of failure based on the error.

        Args:
            error: The error that occurred

        Returns:
            FailureType: Classified failure type
        """
        error_str = str(error).lower()

        if "timeout" in error_str or "timed out" in error_str:
            return FailureType.TIMEOUT
        elif "connection" in error_str or "connect" in error_str:
            return FailureType.CONNECTION_ERROR
        elif "not found" in error_str or "404" in error_str:
            return FailureType.MODEL_NOT_FOUND
        elif "memory" in error_str or "out of memory" in error_str:
            return FailureType.MEMORY_ERROR
        elif "generation" in error_str or "generate" in error_str:
            return FailureType.GENERATION_ERROR
        else:
            return FailureType.UNKNOWN_ERROR

    def _ensure_model_health(self, model_name: str) -> ModelHealth:
        """
        Ensure model health tracking exists for a model.

        Args:
            model_name: Name of the model

        Returns:
            ModelHealth: Health tracking object
        """
        if model_name not in self.model_health:
            self.model_health[model_name] = ModelHealth(
                model_name=model_name,
                status=ModelStatus.UNKNOWN
            )
        return self.model_health[model_name]

    def get_best_model(self, preferred_model: str, agent_type: Optional[str] = None) -> str:
        """
        Get the best available model, considering health and fallbacks.

        Args:
            preferred_model: The preferred model to use
            agent_type: Type of agent requesting the model (for caching)

        Returns:
            str: Best available model name
        """
        # Check cache first
        cache_key = f"{preferred_model}:{agent_type or 'default'}"
        if cache_key in self._model_selection_cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < timedelta(seconds=self._cache_ttl):
                return self._model_selection_cache[cache_key]

        # Check if preferred model is available
        if self._is_model_available(preferred_model):
            selected_model = preferred_model
        else:
            # Find fallback model
            selected_model = self._find_fallback_model(preferred_model)

        # Cache the selection
        self._model_selection_cache[cache_key] = selected_model
        self._cache_timestamps[cache_key] = datetime.now()

        if selected_model != preferred_model:
            self.logger.info(f"Using fallback model {selected_model} instead of {preferred_model}")

        return selected_model

    def _is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available for use.

        Args:
            model_name: Name of the model to check

        Returns:
            bool: True if model is available
        """
        if not self.ollama_settings.enable_model_fallback:
            return True  # If fallback is disabled, assume all models are available

        if model_name in self.quarantined_models:
            return False

        health = self.model_health.get(model_name)
        if not health:
            return True  # Unknown models are assumed available

        return health.status in [ModelStatus.HEALTHY, ModelStatus.UNKNOWN]

    def _find_fallback_model(self, preferred_model: str) -> str:
        """
        Find the best fallback model.

        Args:
            preferred_model: The preferred model that's not available

        Returns:
            str: Best fallback model
        """
        # Try configured fallback models in order
        for fallback in self.ollama_settings.fallback_models:
            if self._is_model_available(fallback):
                return fallback

        # If no configured fallbacks are available, try the default model
        if (preferred_model != self.ollama_settings.default_model and
            self._is_model_available(self.ollama_settings.default_model)):
            return self.ollama_settings.default_model

        # Last resort: return the preferred model anyway
        self.logger.error(f"No fallback models available for {preferred_model}, using anyway")
        return preferred_model

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of model health status.

        Returns:
            Dict[str, Any]: Health summary
        """
        summary = {
            "total_models": len(self.model_health),
            "healthy_models": 0,
            "degraded_models": 0,
            "quarantined_models": len(self.quarantined_models),
            "models": {}
        }

        for model_name, health in self.model_health.items():
            if health.status == ModelStatus.HEALTHY:
                summary["healthy_models"] += 1
            elif health.status == ModelStatus.DEGRADED:
                summary["degraded_models"] += 1

            summary["models"][model_name] = {
                "status": health.status.value,
                "consecutive_failures": health.consecutive_failures,
                "total_failures": health.total_failures,
                "total_requests": health.total_requests,
                "success_rate": (
                    (health.total_requests - health.total_failures) / health.total_requests
                    if health.total_requests > 0 else 0
                ),
                "average_response_time": health.average_response_time,
                "last_success": health.last_success.isoformat() if health.last_success else None,
                "last_failure": health.last_failure.isoformat() if health.last_failure else None,
                "quarantine_until": health.quarantine_until.isoformat() if health.quarantine_until else None
            }

        return summary

    def reset_model_health(self, model_name: str):
        """
        Reset health status for a specific model.

        Args:
            model_name: Name of the model to reset
        """
        if model_name in self.model_health:
            health = self.model_health[model_name]
            health.status = ModelStatus.UNKNOWN
            health.consecutive_failures = 0
            health.failure_history.clear()
            health.quarantine_until = None

            if model_name in self.quarantined_models:
                self.quarantined_models.remove(model_name)

            self.logger.info(f"Reset health status for model: {model_name}")

    def clear_cache(self):
        """Clear the model selection cache."""
        self._model_selection_cache.clear()
        self._cache_timestamps.clear()
        self.logger.debug("Cleared model selection cache")

# Global model reliability manager instance
_model_reliability_manager: Optional[ModelReliabilityManager] = None

def get_model_reliability_manager(config: Optional[WitsV3Config] = None) -> ModelReliabilityManager:
    """
    Get the global model reliability manager instance.

    Args:
        config: Configuration (only used for first initialization)

    Returns:
        ModelReliabilityManager: The global instance
    """
    global _model_reliability_manager

    if _model_reliability_manager is None:
        if config is None:
            raise ValueError("Configuration required for first initialization")
        _model_reliability_manager = ModelReliabilityManager(config)

    return _model_reliability_manager
