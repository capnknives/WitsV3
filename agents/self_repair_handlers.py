"""
Task handlers for the Self-Repair Agent
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator

from core.schemas import StreamData
from agents.self_repair_models import SystemIssue, SystemMetrics, EvolutionSuggestion
from agents.self_repair_utils import (
    create_issue_record,
    determine_repair_strategy,
    fix_memory_leak,
    fix_high_cpu,
    fix_disk_space,
    fix_configuration_error,
    fix_tool_failure,
    fix_performance_issue,
    repair_tool,
    analyze_performance_trends,
    extract_fix_suggestions,
    provide_manual_repair_guidance,
    apply_automatic_optimizations,
    analyze_usage_patterns,
    identify_capability_gaps,
    parse_evolution_suggestions,
    apply_automatic_evolutions,
    extract_resilience_improvements,
    perform_maintenance_task
)


async def handle_health_check(
    agent,
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Perform comprehensive system health check.
    
    Args:
        agent: The SelfRepairAgent instance
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the health check process
    """
    
    yield agent.stream_action("Performing system health check...")
    
    # Collect current metrics
    current_metrics = await agent._collect_system_metrics()
    agent.system_metrics.append(current_metrics)
    
    # Keep only recent metrics
    if len(agent.system_metrics) > agent.metrics_retention:
        agent.system_metrics = agent.system_metrics[-agent.metrics_retention:]
    
    yield agent.stream_observation(f"CPU: {current_metrics.cpu_usage:.1f}%, Memory: {current_metrics.memory_usage:.1f}%")
    yield agent.stream_observation(f"Disk: {current_metrics.disk_usage:.1f}%, Response Time: {current_metrics.response_time:.2f}s")
    
    # Check for threshold violations
    issues_detected = []
    
    if current_metrics.cpu_usage > agent.thresholds['cpu_usage']:
        issues_detected.append(f"High CPU usage: {current_metrics.cpu_usage:.1f}%")
    
    if current_metrics.memory_usage > agent.thresholds['memory_usage']:
        issues_detected.append(f"High memory usage: {current_metrics.memory_usage:.1f}%")
    
    if current_metrics.disk_usage > agent.thresholds['disk_usage']:
        issues_detected.append(f"High disk usage: {current_metrics.disk_usage:.1f}%")
    
    if current_metrics.response_time > agent.thresholds['response_time']:
        issues_detected.append(f"Slow response time: {current_metrics.response_time:.2f}s")
    
    if current_metrics.error_rate > agent.thresholds['error_rate']:
        issues_detected.append(f"High error rate: {current_metrics.error_rate:.2%}")
    
    # Check for tool failures if tool registry is available
    if agent.tool_registry and current_metrics.tool_failures > 0:
        issues_detected.append(f"Tool failures detected: {current_metrics.tool_failures}")
    
    if issues_detected:
        yield agent.stream_observation("Issues detected:")
        for issue in issues_detected:
            yield agent.stream_observation(f"  - {issue}")
        
        # Create issue records
        for issue_desc in issues_detected:
            category = "performance"
            if "tool" in issue_desc.lower():
                category = "tool_failure"
            elif "memory" in issue_desc.lower():
                category = "memory"
            elif "cpu" in issue_desc.lower():
                category = "cpu"
            elif "disk" in issue_desc.lower():
                category = "disk_space"
            
            issue = await create_issue_record(issue_desc, category, "medium")
            agent.detected_issues[issue.id] = issue
    
    else:
        yield agent.stream_result("System health check passed - all metrics within normal ranges")
    
    # Store health check results
    await agent.store_memory(
        content=f"Health check completed: {len(issues_detected)} issues detected",
        segment_type="HEALTH_CHECK",
        importance=0.7,
        metadata={
            "cpu_usage": current_metrics.cpu_usage,
            "memory_usage": current_metrics.memory_usage,
            "issues_count": len(issues_detected),
            "session_id": session_id
        }
    )
    
    # Trend analysis
    if len(agent.system_metrics) > 10:
        yield agent.stream_thinking("Analyzing performance trends...")
        trends = await analyze_performance_trends()
        if trends['concerns']:
            yield agent.stream_observation("Performance trends indicate:")
            for concern in trends['concerns']:
                yield agent.stream_observation(f"  - {concern}")


async def handle_issue_diagnosis(
    agent,
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Diagnose existing system issues.
    
    Args:
        agent: The SelfRepairAgent instance
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the issue diagnosis process
    """
    
    yield agent.stream_action("Diagnosing system issues...")
    
    if not agent.detected_issues:
        yield agent.stream_result("No outstanding issues detected")
        return
    
    for issue_id, issue in agent.detected_issues.items():
        yield agent.stream_thinking(f"Analyzing issue: {issue.description}")
        
        # Generate detailed diagnosis
        diagnosis_prompt = f"""
        Provide detailed diagnosis for this system issue:
        
        Category: {issue.category}
        Description: {issue.description}
        Location: {issue.location}
        Severity: {issue.severity}
        
        Provide:
        1. Root cause analysis
        2. Impact assessment
        3. Potential solutions
        4. Prevention strategies
        5. Urgency level
        """
        
        diagnosis = await agent.generate_response(diagnosis_prompt, temperature=0.5)
        
        # Extract actionable suggestions
        suggestions = await extract_fix_suggestions(diagnosis)
        issue.fix_suggestions = suggestions
        
        yield agent.stream_observation(f"Issue: {issue.description}")
        yield agent.stream_observation(f"Diagnosis: {diagnosis[:200]}...")
        
        if suggestions:
            yield agent.stream_observation("Suggested fixes:")
            for suggestion in suggestions[:3]:
                yield agent.stream_observation(f"  - {suggestion}")
        
        # Store diagnosis
        await agent.store_memory(
            content=f"Issue diagnosis: {diagnosis}",
            segment_type="ISSUE_DIAGNOSIS",
            importance=0.8,
            metadata={
                "issue_id": issue_id,
                "category": issue.category,
                "severity": issue.severity,
                "session_id": session_id
            }
        )


async def handle_system_repair(
    agent,
    task_analysis: Dict[str, Any], 
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Handle automated system repair.
    
    Args:
        agent: The SelfRepairAgent instance
        task_analysis: Analysis of the repair task
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the system repair process
    """
    
    yield agent.stream_action("Initiating system repair procedures...")
    
    if not agent.auto_fix_enabled:
        yield agent.stream_observation("Auto-repair is disabled - providing manual repair guidance")
        guidance = await provide_manual_repair_guidance(session_id)
        yield agent.stream_result(guidance)
        return
    
    repairs_attempted = 0
    repairs_successful = 0
    
    for issue_id, issue in list(agent.detected_issues.items()):  # Use list() to avoid modification during iteration
        if issue.status != "open" or not issue.auto_fixable:
            continue
        
        yield agent.stream_action(f"Attempting to fix: {issue.description}")
        
        try:
            # Determine repair strategy
            strategy = await determine_repair_strategy(issue)
            
            success = False
            if strategy == "memory_leak":
                success = await fix_memory_leak(issue)
            elif strategy == "high_cpu":
                success = await fix_high_cpu(issue)
            elif strategy == "disk_space":
                success = await fix_disk_space(issue)
            elif strategy == "configuration_error":
                success = await fix_configuration_error(issue)
            elif strategy == "tool_failure":
                success = await fix_tool_failure(issue)
            elif strategy == "performance_degradation":
                success = await fix_performance_issue(issue)
            
            repairs_attempted += 1
            
            if success:
                issue.status = "resolved"
                repairs_successful += 1
                yield agent.stream_result(f"Successfully fixed: {issue.description}")
                
                # Remove from active issues
                del agent.detected_issues[issue_id]
            else:
                issue.status = "failed"
                issue.resolution_attempts += 1
                yield agent.stream_observation(f"Failed to fix: {issue.description}")
        
        except Exception as e:
            yield agent.stream_observation(f"Error during repair: {str(e)}")
            issue.status = "error"
            issue.resolution_attempts += 1
    
    yield agent.stream_result(f"Repair summary: {repairs_successful}/{repairs_attempted} fixes successful")
    
    # Store repair results
    await agent.store_memory(
        content=f"System repair completed: {repairs_successful}/{repairs_attempted} successful",
        segment_type="SYSTEM_REPAIR",
        importance=0.9,
        metadata={
            "repairs_attempted": repairs_attempted,
            "repairs_successful": repairs_successful,
            "session_id": session_id
        }
    )


async def handle_performance_optimization(
    agent,
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Handle performance optimization.
    
    Args:
        agent: The SelfRepairAgent instance
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the performance optimization process
    """
    
    yield agent.stream_action("Analyzing performance optimization opportunities...")
    
    # Analyze recent performance data
    if len(agent.system_metrics) < 10:
        yield agent.stream_observation("Insufficient performance data for optimization analysis")
        return
    
    optimization_prompt = """
    Analyze these system performance metrics and suggest optimizations:
    
    Recent metrics:
    - Average CPU: {cpu}%
    - Average Memory: {memory}%
    - Average Response Time: {response}s
    - Error Rate: {error}%
    
    Provide specific optimization recommendations:
    1. Performance bottlenecks to address
    2. Resource utilization improvements
    3. Configuration optimizations
    4. Code-level optimizations
    5. Infrastructure improvements
    """.format(
        cpu=sum(m.cpu_usage for m in agent.system_metrics[-10:]) / 10,
        memory=sum(m.memory_usage for m in agent.system_metrics[-10:]) / 10,
        response=sum(m.response_time for m in agent.system_metrics[-10:]) / 10,
        error=sum(m.error_rate for m in agent.system_metrics[-10:]) / 10 * 100
    )
    
    yield agent.stream_thinking("Generating optimization recommendations...")
    optimization_suggestions = await agent.generate_response(optimization_prompt, temperature=0.6)
    
    yield agent.stream_result("Performance Optimization Recommendations:")
    yield agent.stream_result(optimization_suggestions)
    
    # Implement automatic optimizations
    auto_optimizations = await apply_automatic_optimizations()
    
    if auto_optimizations:
        yield agent.stream_action("Applied automatic optimizations:")
        for opt in auto_optimizations:
            yield agent.stream_action(f"  - {opt}")
    
    # Store optimization analysis
    await agent.store_memory(
        content=f"Performance optimization analysis: {optimization_suggestions}",
        segment_type="PERFORMANCE_OPTIMIZATION",
        importance=0.8,
        metadata={
            "optimizations_applied": len(auto_optimizations),
            "session_id": session_id
        }
    )


async def handle_capability_evolution(
    agent,
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Handle capability evolution and enhancement.
    
    Args:
        agent: The SelfRepairAgent instance
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the capability evolution process
    """
    
    yield agent.stream_action("Analyzing capability evolution opportunities...")
    
    # Analyze usage patterns and gaps
    usage_analysis = await analyze_usage_patterns()
    capability_gaps = await identify_capability_gaps()
    
    evolution_prompt = f"""
    Based on system usage analysis, suggest capability evolution:
    
    Usage Patterns: {usage_analysis}
    Capability Gaps: {capability_gaps}
    
    Suggest:
    1. New features to develop
    2. Existing capabilities to enhance
    3. Integration opportunities
    4. Automation improvements
    5. User experience enhancements
    
    Prioritize by impact and feasibility.
    """
    
    yield agent.stream_thinking("Generating evolution suggestions...")
    evolution_suggestions = await agent.generate_response(evolution_prompt, temperature=0.7)
    
    # Parse suggestions into structured format
    suggestions = await parse_evolution_suggestions(evolution_suggestions)
    
    for suggestion in suggestions:
        agent.evolution_suggestions[suggestion.id] = suggestion
    
    yield agent.stream_result("Capability Evolution Suggestions:")
    yield agent.stream_result(evolution_suggestions)
    
    # Implement simple evolutionary improvements
    auto_evolutions = await apply_automatic_evolutions()
    
    if auto_evolutions:
        yield agent.stream_action("Applied automatic improvements:")
        for evolution in auto_evolutions:
            yield agent.stream_action(f"  - {evolution}")
    
    # Store evolution analysis
    await agent.store_memory(
        content=f"Capability evolution analysis: {evolution_suggestions}",
        segment_type="CAPABILITY_EVOLUTION",
        importance=0.8,
        metadata={
            "suggestions_count": len(suggestions),
            "auto_evolutions": len(auto_evolutions),
            "session_id": session_id
        }
    )


async def handle_failure_learning(
    agent,
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Learn from system failures and improve resilience.
    
    Args:
        agent: The SelfRepairAgent instance
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the failure learning process
    """
    
    yield agent.stream_action("Analyzing failure patterns for learning...")
    
    # Search for failure-related memories
    failure_memories = []
    if agent.memory_manager:
        for failure_type in ['ERROR', 'SYSTEM_FAILURE', 'TOOL_FAILURE']:
            memories = await agent.memory_manager.search_memory(failure_type, limit=20)
            failure_memories.extend(memories)
    
    if not failure_memories:
        yield agent.stream_result("No failure patterns found for analysis")
        return
    
    # Analyze failure patterns
    learning_prompt = f"""
    Analyze these system failures to identify patterns and learning opportunities:
    
    Failure Count: {len(failure_memories)}
    
    Extract:
    1. Common failure patterns
    2. Root causes
    3. Prevention strategies
    4. System resilience improvements
    5. Monitoring enhancements
    6. Recovery procedures
    
    Focus on actionable improvements to prevent future failures.
    """
    
    yield agent.stream_thinking("Extracting failure insights...")
    failure_analysis = await agent.generate_response(learning_prompt, temperature=0.5)
    
    # Extract actionable improvements
    improvements = await extract_resilience_improvements(failure_analysis)
    
    yield agent.stream_result("Failure Analysis and Learning:")
    yield agent.stream_result(failure_analysis)
    
    # Update neural web with failure patterns
    if agent.neural_web:
        for improvement in improvements:
            await agent.neural_web.add_concept(
                f"resilience_{improvement.replace(' ', '_')}",
                f"Resilience improvement: {improvement}",
                "resilience_pattern"
            )
    
    # Store learning insights
    await agent.store_memory(
        content=f"Failure learning analysis: {failure_analysis}",
        segment_type="FAILURE_LEARNING",
        importance=0.9,
        metadata={
            "failures_analyzed": len(failure_memories),
            "improvements_identified": len(improvements),
            "session_id": session_id
        }
    )


async def handle_tool_monitoring(
    agent,
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Handle tool monitoring and repair.
    
    Args:
        agent: The SelfRepairAgent instance
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the tool monitoring process
    """
    
    yield agent.stream_action("Analyzing tool performance and failures...")
    
    if not agent.tool_registry:
        yield agent.stream_observation("Tool registry not available - cannot monitor tools")
        return
    
    # Get tool failure counts
    failing_tools = {name: count for name, count in agent.tool_failure_counts.items() if count > 0}
    
    if not failing_tools:
        yield agent.stream_result("No tool failures detected")
        return
    
    yield agent.stream_observation(f"Detected {len(failing_tools)} tools with failures:")
    for tool_name, failure_count in failing_tools.items():
        yield agent.stream_observation(f"  - {tool_name}: {failure_count} failures")
        
        # Create issue record for each failing tool
        issue_desc = f"Tool failure: {tool_name} ({failure_count} failures)"
        issue = await create_issue_record(issue_desc, "tool_failure", "high" if failure_count > 5 else "medium")
        agent.detected_issues[issue.id] = issue
    
    # Attempt to repair failing tools
    repairs_attempted = 0
    repairs_successful = 0
    
    for tool_name in failing_tools:
        yield agent.stream_action(f"Attempting to repair tool: {tool_name}")
        
        try:
            success = await repair_tool(tool_name)
            
            repairs_attempted += 1
            
            if success:
                repairs_successful += 1
                agent.tool_failure_counts[tool_name] = 0  # Reset failure count
                yield agent.stream_result(f"Successfully repaired tool: {tool_name}")
            else:
                yield agent.stream_observation(f"Failed to repair tool: {tool_name}")
        
        except Exception as e:
            yield agent.stream_observation(f"Error repairing tool {tool_name}: {str(e)}")
    
    yield agent.stream_result(f"Tool repair summary: {repairs_successful}/{repairs_attempted} repairs successful")
    
    # Store tool monitoring results
    await agent.store_memory(
        content=f"Tool monitoring completed: {len(failing_tools)} tools with failures, {repairs_successful}/{repairs_attempted} repairs successful",
        segment_type="TOOL_MONITORING",
        importance=0.8,
        metadata={
            "failing_tools": len(failing_tools),
            "repairs_successful": repairs_successful,
            "session_id": session_id
        }
    )


async def handle_general_maintenance(
    agent,
    task_analysis: Dict[str, Any], 
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Handle general maintenance tasks.
    
    Args:
        agent: The SelfRepairAgent instance
        task_analysis: Analysis of the maintenance task
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the general maintenance process
    """
    
    yield agent.stream_action("Performing general system maintenance...")
    
    maintenance_tasks = [
        "Cleaning temporary files",
        "Updating configuration",
        "Optimizing memory usage",
        "Checking component health",
        "Updating tool registry",
        "Pruning old memories"
    ]
    
    completed_tasks = []
    
    for task in maintenance_tasks:
        yield agent.stream_action(f"Task: {task}")
        
        try:
            # Perform maintenance task
            success = await perform_maintenance_task(task)
            
            if success:
                completed_tasks.append(task)
                yield agent.stream_observation(f"✓ {task}")
            else:
                yield agent.stream_observation(f"✗ {task} - failed")
        
        except Exception as e:
            yield agent.stream_observation(f"✗ {task} - error: {str(e)}")
    
    yield agent.stream_result(f"Maintenance completed: {len(completed_tasks)}/{len(maintenance_tasks)} tasks successful")
    
    # Store maintenance results
    await agent.store_memory(
        content=f"General maintenance: {len(completed_tasks)} tasks completed",
        segment_type="GENERAL_MAINTENANCE",
        importance=0.6,
        metadata={
            "tasks_completed": completed_tasks,
            "session_id": session_id
        }
    )
