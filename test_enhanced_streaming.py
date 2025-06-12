#!/usr/bin/env python3
"""
Test script for enhanced streaming error context in WitsV3.
This demonstrates the new error context and tracing capabilities.
"""

import asyncio
import logging
import traceback
import uuid
from typing import Dict, Any, Optional, List, AsyncGenerator
from core.schemas import StreamData
from agents.base_agent import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStreamingAgent(BaseAgent):
    """Test agent to demonstrate enhanced streaming error context."""

    def __init__(self):
        super().__init__(
            agent_name="TestStreamingAgent",
            agent_description="Agent for testing enhanced streaming error context"
        )    async def process_user_input(self, user_input: str) -> AsyncGenerator[StreamData, None]:
        """Process user input with enhanced error tracking."""
        # Generate a trace ID for this request
        trace_id = str(uuid.uuid4())

        try:
            # Stream thinking with context
            yield self.stream_thinking(
                "Starting to process user input...",
                context={"input_length": len(user_input), "input_preview": user_input[:50]},
                trace_id=trace_id
            )

            # Simulate some processing with potential errors
            if "error" in user_input.lower():
                # Demonstrate enhanced error streaming
                yield self.create_enhanced_error(
                    error="Simulated processing error",
                    error_code="TEST_ERROR_001",
                    category="validation",
                    severity="high",
                    details=f"Error triggered by user input: {user_input}",
                    suggested_actions=["Try a different input", "Check input format"],
                    context={"original_input": user_input, "error_location": "input_validation"},
                    trace_id=trace_id
                )
                return

            elif "timeout" in user_input.lower():
                # Demonstrate timeout error with retry info
                yield self.create_enhanced_error(
                    error="Processing timeout",
                    error_code="TIMEOUT_001",
                    category="timeout",
                    severity="medium",
                    details="Processing took longer than expected",
                    suggested_actions=["Retry with shorter input", "Try again later"],
                    retry_info={"max_retries": 3, "current_attempt": 1, "next_retry_in": "30s"},
                    trace_id=trace_id
                )
                return

            elif "warning" in user_input.lower():
                # Demonstrate warning with suggestions
                yield self.create_warning(
                    warning="Input may contain sensitive information",
                    severity="medium",
                    suggested_actions=["Review input for sensitive data", "Use anonymized examples"],
                    context={"detected_patterns": ["email", "phone"], "confidence": 0.8},
                    trace_id=trace_id
                )

            # Stream action
            yield self.stream_action(
                "Processing input successfully",
                context={"processing_stage": "main", "input_type": "text"},
                trace_id=trace_id
            )

            # Stream result
            yield self.stream_result(
                f"Successfully processed: {user_input}",
                context={"output_length": len(user_input) * 2, "processing_time": "0.1s"},
                trace_id=trace_id
            )

        except Exception as e:
            # Demonstrate exception handling with full context
            yield self.create_enhanced_error(
                error=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                category="execution",
                severity="critical",
                details=f"Exception occurred while processing: {user_input}",
                stack_trace=traceback.format_exc(),
                suggested_actions=["Report this error", "Try with simpler input"],
                context={"input": user_input, "exception_type": type(e).__name__},
                trace_id=trace_id
            )

    def create_enhanced_error(self, error: str, error_code: str, category: str, severity: str = "medium",
                             details: Optional[str] = None, stack_trace: Optional[str] = None,
                             suggested_actions: Optional[List[str]] = None,
                             context: Optional[Dict[str, Any]] = None,
                             retry_info: Optional[Dict[str, Any]] = None,
                             trace_id: Optional[str] = None) -> StreamData:
        """Create an enhanced error stream with comprehensive context."""
        stream_data = StreamData(
            type="error",
            content=error,
            source=self.agent_name,
            error_details=details,
            error_code=error_code,
            error_category=category,
            severity=severity,
            stack_trace=stack_trace,
            suggested_actions=suggested_actions,
            context=context,
            retry_info=retry_info,
            trace_id=trace_id,
            correlation_id=f"corr_{trace_id[:8]}" if trace_id else None
        )
        return stream_data

    def create_warning(self, warning: str, severity: str = "low",
                      suggested_actions: Optional[List[str]] = None,
                      context: Optional[Dict[str, Any]] = None,
                      trace_id: Optional[str] = None) -> StreamData:
        """Create a warning stream with context."""
        stream_data = StreamData(
            type="warning",
            content=warning,
            source=self.agent_name,
            severity=severity,
            suggested_actions=suggested_actions,
            context=context,
            trace_id=trace_id,
            correlation_id=f"corr_{trace_id[:8]}" if trace_id else None
        )
        return stream_data

    def stream_thinking(self, thought: str, context: Optional[Dict[str, Any]] = None,
                       trace_id: Optional[str] = None) -> StreamData:
        """Create an enhanced thinking stream with context."""
        stream_data = StreamData(
            type="thinking",
            content=thought,
            source=self.agent_name,
            context=context,
            trace_id=trace_id
        )
        return stream_data

    def stream_action(self, action: str, context: Optional[Dict[str, Any]] = None,
                     trace_id: Optional[str] = None) -> StreamData:
        """Create an enhanced action stream with context."""
        stream_data = StreamData(
            type="action",
            content=action,
            source=self.agent_name,
            context=context,
            trace_id=trace_id
        )
        return stream_data

    def stream_result(self, result: str, context: Optional[Dict[str, Any]] = None,
                     trace_id: Optional[str] = None) -> StreamData:
        """Create an enhanced result stream with context."""
        stream_data = StreamData(
            type="result",
            content=result,
            source=self.agent_name,
            context=context,
            trace_id=trace_id        )
        return stream_data

    async def run(self, user_input: str) -> str:
        """Required run method implementation."""
        # For this test, we'll just return a simple result
        return f"Processed: {user_input}"


def format_stream_data(stream_data: StreamData) -> str:
    """Format stream data for display with enhanced error context."""
    timestamp = stream_data.timestamp.strftime("%H:%M:%S.%f")[:-3]
    base_msg = f"[{timestamp}] [{stream_data.type.upper()}] {stream_data.source}: {stream_data.content}"

    additional_info = []

    # Add trace information
    if stream_data.trace_id:
        additional_info.append(f"Trace: {stream_data.trace_id[:8]}...")

    # Add error context
    if stream_data.is_error():
        if stream_data.error_code:
            additional_info.append(f"Code: {stream_data.error_code}")
        if stream_data.error_category:
            additional_info.append(f"Category: {stream_data.error_category}")
        if stream_data.severity:
            additional_info.append(f"Severity: {stream_data.severity}")

    # Add warnings
    if stream_data.is_warning() and stream_data.severity:
        additional_info.append(f"Severity: {stream_data.severity}")    # Add context information
    if stream_data.has_context() and stream_data.context:
        context_summary = ", ".join([f"{k}={v}" for k, v in list(stream_data.context.items())[:3]])
        additional_info.append(f"Context: {context_summary}")

    # Add suggested actions
    if stream_data.suggested_actions:
        actions = ", ".join(stream_data.suggested_actions[:2])
        additional_info.append(f"Suggestions: {actions}")

    # Add retry info
    if stream_data.retry_info:
        retry_info = ", ".join([f"{k}={v}" for k, v in list(stream_data.retry_info.items())[:2]])
        additional_info.append(f"Retry: {retry_info}")

    if additional_info:
        base_msg += f" | {' | '.join(additional_info)}"

    return base_msg


async def test_enhanced_streaming():
    """Test the enhanced streaming error context system."""
    print("🎬 Testing Enhanced Streaming Error Context")
    print("=" * 60)

    agent = TestStreamingAgent()

    test_inputs = [
        "Hello, how are you?",  # Normal processing
        "This input contains error keyword",  # Trigger error
        "This will cause a timeout",  # Trigger timeout
        "This input has warning content",  # Trigger warning
    ]

    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n🧪 Test {i}: {test_input}")
        print("-" * 40)

        try:
            async for stream_data in agent.process_user_input(test_input):
                print(format_stream_data(stream_data))

                # Demonstrate error summary
                if stream_data.is_error():
                    error_summary = stream_data.get_error_summary()
                    if error_summary:
                        print(f"    📋 Error Summary: {error_summary}")

                # Show stack trace for critical errors
                if stream_data.is_error() and stream_data.severity == "critical" and stream_data.stack_trace:
                    print(f"    📚 Stack Trace: {stream_data.stack_trace[:100]}...")

        except Exception as e:
            print(f"    ❌ Unexpected error: {e}")

    print("\n✅ Enhanced streaming error context testing completed!")


async def test_error_tracing():
    """Test error tracing and correlation."""
    print("\n🔍 Testing Error Tracing and Correlation")
    print("=" * 60)

    # Create some related stream events with trace correlation
    trace_id = str(uuid.uuid4())
    parent_id = str(uuid.uuid4())

    events = [
        StreamData(
            type="thinking",
            content="Starting complex operation",
            source="ParentAgent",
            trace_id=trace_id,
            context={"operation": "complex_task", "step": 1}
        ),
        StreamData(
            type="action",
            content="Calling sub-process",
            source="ParentAgent",
            trace_id=trace_id,
            parent_id=parent_id,
            context={"subprocess": "data_processing", "step": 2}
        ),
        StreamData(
            type="error",
            content="Sub-process failed",
            source="SubAgent",
            trace_id=trace_id,
            parent_id=parent_id,
            error_code="SUB_PROC_001",
            error_category="execution",
            severity="high",
            suggested_actions=["Retry with different parameters", "Check input data"],
            context={"failed_step": "data_validation", "error_count": 1}
        )
    ]

    print("Event correlation chain:")
    for event in events:
        print(f"  {format_stream_data(event)}")
        if event.parent_id:
            print(f"    └─ Parent: {event.parent_id[:8]}...")

    print("\n✅ Error tracing testing completed!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_streaming())
    asyncio.run(test_error_tracing())
