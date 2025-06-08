"""
Learning and failure analysis handlers for the Self-Repair Agent
"""

from typing import Dict, Any, AsyncGenerator

from core.schemas import StreamData
from agents.self_repair_utils import (
    extract_resilience_improvements,
    perform_maintenance_task
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


async def handle_maintenance_task(
    agent,
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Perform routine maintenance tasks.
    
    Args:
        agent: The SelfRepairAgent instance
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the maintenance process
    """
    
    yield agent.stream_action("Performing system maintenance...")
    
    # Define maintenance tasks
    maintenance_tasks = [
        "Clean temporary files",
        "Optimize database indices",
        "Update system dependencies",
        "Rotate log files",
        "Verify backup integrity",
        "Clear cache files",
        "Update configuration settings",
        "Check security updates"
    ]
    
    completed_tasks = []
    failed_tasks = []
    
    for task in maintenance_tasks:
        yield agent.stream_action(f"Executing: {task}")
        
        try:
            success = await perform_maintenance_task(task)
            
            if success:
                completed_tasks.append(task)
                yield agent.stream_observation(f"✓ {task} completed")
            else:
                failed_tasks.append(task)
                yield agent.stream_observation(f"✗ {task} failed")
                
        except Exception as e:
            failed_tasks.append(task)
            yield agent.stream_observation(f"Error during {task}: {str(e)}")
    
    # Generate maintenance report
    yield agent.stream_result(f"Maintenance Summary:")
    yield agent.stream_result(f"- Completed: {len(completed_tasks)}/{len(maintenance_tasks)} tasks")
    
    if failed_tasks:
        yield agent.stream_observation(f"Failed tasks:")
        for task in failed_tasks:
            yield agent.stream_observation(f"  - {task}")
    
    # Store maintenance results
    await agent.store_memory(
        content=f"Maintenance completed: {len(completed_tasks)}/{len(maintenance_tasks)} successful",
        segment_type="SYSTEM_MAINTENANCE",
        importance=0.6,
        metadata={
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "session_id": session_id
        }
    )


async def handle_general_maintenance(
    agent,
    task_analysis: Dict[str, Any], 
    session_id: str
) -> AsyncGenerator[StreamData, None]:
    """
    Handle general maintenance requests based on task analysis.
    
    Args:
        agent: The SelfRepairAgent instance
        task_analysis: Analysis of the maintenance task
        session_id: The current session ID
        
    Yields:
        StreamData: Stream data for the general maintenance process
    """
    
    yield agent.stream_thinking("Determining appropriate maintenance action...")
    
    # Extract specific requirements from task analysis
    requirements = task_analysis.get('requirements', [])
    
    if any('health' in req.lower() for req in requirements):
        # Delegate to health check
        from .health_check_handlers import handle_health_check
        async for stream in handle_health_check(agent, session_id):
            yield stream
    
    elif any('repair' in req.lower() for req in requirements):
        # Delegate to system repair
        from .repair_handlers import handle_system_repair
        async for stream in handle_system_repair(agent, task_analysis, session_id):
            yield stream
    
    elif any('optimize' in req.lower() or 'performance' in req.lower() for req in requirements):
        # Delegate to performance optimization
        from .optimization_handlers import handle_performance_optimization
        async for stream in handle_performance_optimization(agent, session_id):
            yield stream
    
    elif any('learn' in req.lower() or 'failure' in req.lower() for req in requirements):
        # Delegate to failure learning
        async for stream in handle_failure_learning(agent, session_id):
            yield stream
    
    else:
        # General maintenance
        yield agent.stream_result("Performing general system maintenance...")
        
        # Run a basic health check
        from .health_check_handlers import handle_health_check
        async for stream in handle_health_check(agent, session_id):
            yield stream
        
        # Run routine maintenance
        async for stream in handle_maintenance_task(agent, session_id):
            yield stream
        
        # Provide summary
        yield agent.stream_result("General maintenance completed successfully")