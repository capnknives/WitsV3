"""
Self-repair agent handlers module.

This module provides handlers for various self-repair tasks including
health checks, system repairs, optimization, and failure learning.
"""

from .health_check_handlers import (
    handle_health_check,
    handle_tool_status
)

from .repair_handlers import (
    handle_issue_diagnosis,
    handle_system_repair,
    handle_tool_repair
)

from .optimization_handlers import (
    handle_performance_optimization,
    handle_capability_evolution,
    handle_resource_optimization
)

from .learning_handlers import (
    handle_failure_learning,
    handle_maintenance_task,
    handle_general_maintenance
)

__all__ = [
    # Health check
    'handle_health_check',
    'handle_tool_status',
    
    # Repair
    'handle_issue_diagnosis',
    'handle_system_repair',
    'handle_tool_repair',
    
    # Optimization
    'handle_performance_optimization',
    'handle_capability_evolution',
    'handle_resource_optimization',
    
    # Learning
    'handle_failure_learning',
    'handle_maintenance_task',
    'handle_general_maintenance'
]