"""
Health check handlers for the Self-Repair Agent
"""

from datetime import datetime
from typing import Dict, AsyncGenerator

from core.schemas import StreamData
from agents.self_repair_models import SystemMetrics
from agents.self_repair_utils import create_issue_record, analyze_performance_trends


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


async def handle_tool_status(
    agent,
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Check status of all registered tools.
    
    Args:
        agent: The SelfRepairAgent instance
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the tool status check
    """
    
    yield agent.stream_action("Checking tool statuses...")
    
    if not agent.tool_registry:
        yield agent.stream_observation("No tool registry available")
        return
    
    tools = agent.tool_registry.list_tools()
    tools_checked = 0
    tools_healthy = 0
    tools_failed = []
    
    for tool_name in tools:
        try:
            tool = agent.tool_registry.get_tool(tool_name)
            if tool:
                # Test basic functionality
                test_result = await tool.test_connectivity() if hasattr(tool, 'test_connectivity') else True
                
                tools_checked += 1
                if test_result:
                    tools_healthy += 1
                else:
                    tools_failed.append(tool_name)
                    
        except Exception as e:
            tools_failed.append(tool_name)
            yield agent.stream_observation(f"Tool {tool_name} check failed: {str(e)}")
    
    yield agent.stream_observation(f"Tools checked: {tools_checked}")
    yield agent.stream_observation(f"Healthy tools: {tools_healthy}")
    
    if tools_failed:
        yield agent.stream_observation(f"Failed tools: {', '.join(tools_failed)}")
        
        # Create issues for failed tools
        for tool_name in tools_failed:
            issue = await create_issue_record(
                f"Tool failure: {tool_name}",
                "tool_failure",
                "high"
            )
            issue.location = f"tool:{tool_name}"
            agent.detected_issues[issue.id] = issue
    
    # Store tool status
    await agent.store_memory(
        content=f"Tool status check: {tools_healthy}/{tools_checked} healthy",
        segment_type="TOOL_STATUS",
        importance=0.7,
        metadata={
            "tools_checked": tools_checked,
            "tools_healthy": tools_healthy,
            "tools_failed": len(tools_failed),
            "session_id": session_id
        }
    )