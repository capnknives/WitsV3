# core/schemas.py
"""
Core schemas and data models for WitsV3.
These define the structure for all data flowing through the system.
"""

from typing import Any, Dict, List, Optional, Union, Literal, Type
from pydantic import BaseModel, Field, ValidationError, validator
from datetime import datetime


class StreamData(BaseModel):
    """
    Data structure for streaming responses from agents.
    This allows real-time feedback to users during agent processing.
    Enhanced with comprehensive error context and tracing.
    """
    type: Literal[
        "thinking", "action", "observation", "result", "error",
        "clarification", "goal_defined", "memory_search", "tool_call",
        "warning", "debug", "trace"
    ] = Field(description="Type of stream data")

    content: str = Field(description="The main content/message")

    source: str = Field(description="Which agent/component generated this")

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about this stream"
    )

    timestamp: datetime = Field(default_factory=datetime.now)

    # Enhanced error context fields
    error_details: Optional[str] = Field(
        default=None,
        description="Error details if type is 'error'"
    )

    error_code: Optional[str] = Field(
        default=None,
        description="Error code for programmatic handling"
    )

    error_category: Optional[Literal[
        "validation", "execution", "communication", "configuration",
        "authentication", "permission", "resource", "timeout", "network"
    ]] = Field(
        default=None,
        description="Category of error for better classification"
    )

    stack_trace: Optional[str] = Field(
        default=None,
        description="Stack trace for debugging errors"
    )

    # Context and tracing fields
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contextual information relevant to this stream"
    )

    correlation_id: Optional[str] = Field(
        default=None,
        description="ID to correlate related stream events"
    )

    parent_id: Optional[str] = Field(
        default=None,
        description="ID of parent stream event for tracing"
    )

    trace_id: Optional[str] = Field(
        default=None,
        description="Unique trace ID for end-to-end request tracing"
    )

    severity: Optional[Literal["low", "medium", "high", "critical"]] = Field(
        default=None,
        description="Severity level for errors and warnings"
    )

    # Recovery and suggestions
    suggested_actions: Optional[List[str]] = Field(
        default=None,
        description="Suggested actions to resolve errors or improve results"
    )

    retry_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Information about retry attempts and strategies"
    )

    def is_error(self) -> bool:
        """Check if this stream data represents an error."""
        return self.type == "error"

    def is_warning(self) -> bool:
        """Check if this stream data represents a warning."""
        return self.type == "warning"

    def has_context(self) -> bool:
        """Check if this stream data has context information."""
        return self.context is not None and len(self.context) > 0

    def get_error_summary(self) -> Optional[str]:
        """Get a concise error summary for logging."""
        if not self.is_error():
            return None

        parts = [f"Error from {self.source}: {self.content}"]

        if self.error_code:
            parts.append(f"Code: {self.error_code}")

        if self.error_category:
            parts.append(f"Category: {self.error_category}")

        if self.severity:
            parts.append(f"Severity: {self.severity}")

        return " | ".join(parts)

    def add_trace_context(self, trace_id: str, correlation_id: Optional[str] = None, parent_id: Optional[str] = None) -> 'StreamData':
        """Add tracing context to this stream data."""
        self.trace_id = trace_id
        if correlation_id:
            self.correlation_id = correlation_id
        if parent_id:
            self.parent_id = parent_id
        return self

    def add_error_context(self, error_code: str, category: str, severity: str = "medium",
                         stack_trace: Optional[str] = None, suggested_actions: Optional[List[str]] = None) -> 'StreamData':
        """Add comprehensive error context to this stream data."""
        self.error_code = error_code
        self.error_category = category
        self.severity = severity
        if stack_trace:
            self.stack_trace = stack_trace
        if suggested_actions:
            self.suggested_actions = suggested_actions
        return self


class ToolParameter(BaseModel):
    """Schema for individual tool parameters."""
    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type")
    description: str = Field(description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not required")
    enum_values: Optional[List[Any]] = Field(default=None, description="Allowed values for enum parameters")
    min_value: Optional[Union[int, float]] = Field(default=None, description="Minimum value for numeric parameters")
    max_value: Optional[Union[int, float]] = Field(default=None, description="Maximum value for numeric parameters")
    min_length: Optional[int] = Field(default=None, description="Minimum length for string parameters")
    max_length: Optional[int] = Field(default=None, description="Maximum length for string parameters")
    pattern: Optional[str] = Field(default=None, description="Regex pattern for string validation")


class ToolSchema(BaseModel):
    """Enhanced tool schema with validation support."""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")

    def validate_arguments(self, arguments: Dict[str, Any]) -> "ToolValidationResult":
        """Validate tool arguments against this schema."""
        errors = []
        warnings = []
        validated_args = {}

        # Check required parameters
        required_params = {p.name for p in self.parameters if p.required}
        provided_params = set(arguments.keys())
        missing_params = required_params - provided_params

        if missing_params:
            errors.append(f"Missing required parameters: {', '.join(missing_params)}")

        # Check unknown parameters
        known_params = {p.name for p in self.parameters}
        unknown_params = provided_params - known_params

        if unknown_params:
            warnings.append(f"Unknown parameters will be ignored: {', '.join(unknown_params)}")

        # Validate each provided parameter
        for param in self.parameters:
            if param.name in arguments:
                value = arguments[param.name]
                validation_result = self._validate_parameter(param, value)

                if validation_result["valid"]:
                    validated_args[param.name] = validation_result["value"]
                else:
                    errors.extend(validation_result["errors"])
            elif not param.required and param.default is not None:
                validated_args[param.name] = param.default

        return ToolValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validated_arguments=validated_args
        )

    def _validate_parameter(self, param: ToolParameter, value: Any) -> Dict[str, Any]:
        """Validate a single parameter value."""
        errors = []

        # Type validation
        if param.type == "string" and not isinstance(value, str):
            try:
                value = str(value)
            except Exception:
                errors.append(f"Parameter '{param.name}' must be a string")
                return {"valid": False, "errors": errors}

        elif param.type == "integer" and not isinstance(value, int):
            try:
                value = int(value)
            except Exception:
                errors.append(f"Parameter '{param.name}' must be an integer")
                return {"valid": False, "errors": errors}

        elif param.type == "number" and not isinstance(value, (int, float)):
            try:
                value = float(value)
            except Exception:
                errors.append(f"Parameter '{param.name}' must be a number")
                return {"valid": False, "errors": errors}

        elif param.type == "boolean" and not isinstance(value, bool):
            if isinstance(value, str):
                if value.lower() in ("true", "1", "yes", "on"):
                    value = True
                elif value.lower() in ("false", "0", "no", "off"):
                    value = False
                else:
                    errors.append(f"Parameter '{param.name}' must be a boolean")
                    return {"valid": False, "errors": errors}
            else:
                errors.append(f"Parameter '{param.name}' must be a boolean")
                return {"valid": False, "errors": errors}

        elif param.type == "array" and not isinstance(value, list):
            errors.append(f"Parameter '{param.name}' must be an array")
            return {"valid": False, "errors": errors}

        elif param.type == "object" and not isinstance(value, dict):
            errors.append(f"Parameter '{param.name}' must be an object")
            return {"valid": False, "errors": errors}

        # Enum validation
        if param.enum_values and value not in param.enum_values:
            errors.append(f"Parameter '{param.name}' must be one of: {param.enum_values}")
          # Range validation for numbers
        if param.type in ("integer", "number") and isinstance(value, (int, float)):
            if param.min_value is not None and value < param.min_value:
                errors.append(f"Parameter '{param.name}' must be >= {param.min_value}")
            if param.max_value is not None and value > param.max_value:
                errors.append(f"Parameter '{param.name}' must be <= {param.max_value}")

        # Length validation for strings
        if param.type == "string" and isinstance(value, str):
            if param.min_length is not None and len(value) < param.min_length:
                errors.append(f"Parameter '{param.name}' must be at least {param.min_length} characters")
            if param.max_length is not None and len(value) > param.max_length:
                errors.append(f"Parameter '{param.name}' must be at most {param.max_length} characters")

        # Pattern validation for strings
        if param.type == "string" and param.pattern and isinstance(value, str):
            import re
            if not re.match(param.pattern, value):
                errors.append(f"Parameter '{param.name}' does not match required pattern")

        return {"valid": len(errors) == 0, "errors": errors, "value": value}


class ToolValidationResult(BaseModel):
    """Result of tool argument validation."""
    valid: bool = Field(description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    validated_arguments: Dict[str, Any] = Field(default_factory=dict, description="Validated and coerced arguments")


class ToolExecutionContext(BaseModel):
    """Context for tool execution with validation and error handling."""
    tool_name: str = Field(description="Name of the tool being executed")
    arguments: Dict[str, Any] = Field(description="Tool arguments")
    call_id: Optional[str] = Field(default=None, description="Unique call ID")
    timeout: Optional[int] = Field(default=30, description="Execution timeout in seconds")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    max_retries: int = Field(default=2, description="Maximum number of retries")
    validation_result: Optional[ToolValidationResult] = Field(default=None, description="Validation result")


class EnhancedToolResult(BaseModel):
    """Enhanced tool result with execution context and detailed error information."""
    call_id: Optional[str] = Field(default=None, description="ID of the tool call")
    success: bool = Field(description="Whether the tool execution was successful")
    result: Any = Field(description="The result data from the tool")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    execution_context: Optional[ToolExecutionContext] = Field(default=None, description="Execution context")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Execution warnings")


class ToolCall(BaseModel):
    """Represents a tool being called by an agent."""
    tool_name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(description="Arguments to pass to the tool")
    call_id: Optional[str] = Field(default=None, description="Unique ID for this tool call")


class ToolResult(BaseModel):
    """Result from a tool execution."""
    call_id: Optional[str] = Field(default=None, description="ID of the tool call")
    success: bool = Field(description="Whether the tool execution was successful")
    result: Any = Field(description="The result data from the tool")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class AgentResponse(BaseModel):
    """
    Structured response from an agent's decision-making process.
    This is what agents return after processing user input.
    """
    type: Literal[
        "goal_defined", "clarification_question", "direct_response",
        "tool_call", "final_answer", "delegate_to_agent"
    ] = Field(description="Type of response")

    content: str = Field(description="Main response content")

    # For goal definition
    goal_statement: Optional[str] = Field(
        default=None,
        description="Clear goal statement for orchestrator"
    )

    # For clarification
    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to ask user for clarification"
    )

    # For tool calls
    tool_call: Optional[ToolCall] = Field(
        default=None,
        description="Tool to be executed"
    )

    # For agent delegation
    target_agent: Optional[str] = Field(
        default=None,
        description="Name of agent to delegate to"
    )

    # Reasoning process
    thought_process: Optional[str] = Field(
        default=None,
        description="Agent's reasoning about the decision"
    )

    confidence: Optional[float] = Field(
        default=None,
        description="Confidence level (0.0 to 1.0)"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional response metadata"
    )


class ConversationMessage(BaseModel):
    """Single message in a conversation."""
    role: Literal["user", "assistant", "system"] = Field(description="Message role")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class ConversationHistory(BaseModel):
    """Complete conversation history."""
    session_id: str = Field(description="Unique session identifier")
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_message(self, role: Literal["user", "assistant", "system"], content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a new message to the conversation."""
        message = ConversationMessage(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_recent_messages(self, limit: int = 10) -> List[ConversationMessage]:
        """Get the most recent messages."""
        return self.messages[-limit:] if len(self.messages) > limit else self.messages

    def to_llm_format(self, limit: int = 10) -> List[Dict[str, str]]:
        """Convert to format expected by LLM interface."""
        recent = self.get_recent_messages(limit)
        return [{"role": msg.role, "content": msg.content} for msg in recent]


# Test function
async def test_schemas():
    """Test the schema models."""
    print("Testing WitsV3 schemas...")

    # Test StreamData
    stream = StreamData(
        type="thinking",
        content="I'm analyzing your request...",
        source="WitsControlCenter"
    )
    print(f"✓ StreamData: {stream.type} - {stream.content}")

    # Test AgentResponse
    response = AgentResponse(
        type="goal_defined",
        content="I understand you want to analyze the data.",
        goal_statement="Analyze the provided dataset and generate insights",
        confidence=0.9
    )
    print(f"✓ AgentResponse: {response.type} - {response.goal_statement}")

    # Test ConversationHistory
    history = ConversationHistory(session_id="test-123")
    history.add_message("user", "Hello, can you help me?")
    history.add_message("assistant", "Of course! What do you need help with?")

    llm_format = history.to_llm_format()
    print(f"✓ ConversationHistory: {len(history.messages)} messages")
    print(f"  LLM format: {llm_format}")

    print("All schema tests passed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_schemas())
