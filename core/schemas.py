# core/schemas.py
"""
Core schemas and data models for WitsV3.
These define the structure for all data flowing through the system.
"""

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class StreamData(BaseModel):
    """
    Data structure for streaming responses from agents.
    This allows real-time feedback to users during agent processing.
    """
    type: Literal[
        "thinking", "action", "observation", "result", "error", 
        "clarification", "goal_defined", "memory_search", "tool_call"
    ] = Field(description="Type of stream data")
    
    content: str = Field(description="The main content/message")
    
    source: str = Field(description="Which agent/component generated this")
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional metadata about this stream"
    )
    
    timestamp: datetime = Field(default_factory=datetime.now)
    
    error_details: Optional[str] = Field(
        default=None, 
        description="Error details if type is 'error'"
    )


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
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
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
