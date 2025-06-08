"""
System repair and diagnosis handlers for the Self-Repair Agent
"""

from typing import Dict, Any, AsyncGenerator

from core.schemas import StreamData
from agents.self_repair_utils import (
    extract_fix_suggestions,
    determine_repair_strategy,
    fix_memory_leak,
    fix_high_cpu,
    fix_disk_space,
    fix_configuration_error,
    fix_tool_failure,
    fix_performance_issue,
    repair_tool,
    provide_manual_repair_guidance
)


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


async def handle_tool_repair(
    agent,
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Repair malfunctioning tools.
    
    Args:
        agent: The SelfRepairAgent instance
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the tool repair process
    """
    
    yield agent.stream_action("Initiating tool repair procedures...")
    
    # Find tool-related issues
    tool_issues = [
        (issue_id, issue) for issue_id, issue in agent.detected_issues.items()
        if issue.category == "tool_failure" and issue.status == "open"
    ]
    
    if not tool_issues:
        yield agent.stream_result("No tool failures detected")
        return
    
    repairs_attempted = 0
    repairs_successful = 0
    
    for issue_id, issue in tool_issues:
        if not issue.location or not issue.location.startswith("tool:"):
            continue
        
        tool_name = issue.location.replace("tool:", "")
        yield agent.stream_action(f"Attempting to repair tool: {tool_name}")
        
        try:
            success = await repair_tool(tool_name, issue)
            repairs_attempted += 1
            
            if success:
                issue.status = "resolved"
                repairs_successful += 1
                yield agent.stream_result(f"Successfully repaired: {tool_name}")
                del agent.detected_issues[issue_id]
            else:
                issue.status = "failed"
                issue.resolution_attempts += 1
                yield agent.stream_observation(f"Failed to repair: {tool_name}")
                
        except Exception as e:
            yield agent.stream_observation(f"Error repairing {tool_name}: {str(e)}")
            issue.status = "error"
            issue.resolution_attempts += 1
    
    yield agent.stream_result(f"Tool repair summary: {repairs_successful}/{repairs_attempted} successful")
    
    # Store repair results
    await agent.store_memory(
        content=f"Tool repair completed: {repairs_successful}/{repairs_attempted} successful",
        segment_type="TOOL_REPAIR",
        importance=0.8,
        metadata={
            "repairs_attempted": repairs_attempted,
            "repairs_successful": repairs_successful,
            "session_id": session_id
        }
    )