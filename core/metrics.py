from typing import Dict, Any, List
from datetime import datetime
import logging

class MetricsManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, List[Dict[str, Any]]] = {
            "system": [],
            "llm": [],
            "errors": []
        }

    def record_metric(self, metric_type: str, data: Dict[str, Any]) -> None:
        """Record a metric with timestamp."""
        if metric_type not in self.metrics:
            self.metrics[metric_type] = []

        self.metrics[metric_type].append({
            **data,
            "timestamp": datetime.now().isoformat()
        })
        self.logger.debug(f"Recorded {metric_type} metric: {data}")

    def record_error(self, error_type: str) -> None:
        """Record an error with timestamp."""
        self.metrics["errors"].append({
            "type": error_type,
            "timestamp": datetime.now().isoformat()
        })
        self.logger.error(f"Recorded error: {error_type}")

    def get_metrics(self, metric_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics of a specific type."""
        return self.metrics.get(metric_type, [])[-limit:]

    def clear_metrics(self, metric_type: str) -> None:
        """Clear metrics of a specific type."""
        if metric_type in self.metrics:
            self.metrics[metric_type] = []
            self.logger.info(f"Cleared {metric_type} metrics")
