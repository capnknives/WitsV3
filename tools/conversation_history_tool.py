# tools/conversation_history_tool.py
"""
Conversation History Tool for WitsV3.
Provides tools for reading and analyzing conversation history.
"""

import json
from typing import Any, Dict, List, Optional

from core.base_tool import BaseTool
from core.schemas import ConversationHistory


class ReadConversationHistoryTool(BaseTool):
    """Tool for reading conversation history."""
    
    def __init__(self):
        super().__init__(
            name="read_conversation_history",
            description="Read and format conversation history for analysis"
        )
    
    async def execute(
        self, 
        conversation_history: Optional[ConversationHistory] = None,
        max_messages: int = 10,
        include_metadata: bool = False
    ) -> str:
        """
        Read and format conversation history.
        
        Args:
            conversation_history: Conversation history object
            max_messages: Maximum number of messages to include
            include_metadata: Whether to include message metadata
            
        Returns:
            Formatted conversation history
        """
        try:
            if not conversation_history:
                return "No conversation history provided."
            
            if not hasattr(conversation_history, 'messages') or not conversation_history.messages:
                return "Conversation history is empty."
            
            # Get recent messages
            recent_messages = conversation_history.messages[-max_messages:] if max_messages > 0 else conversation_history.messages
            
            # Format messages
            formatted_messages = []
            for msg in recent_messages:
                if include_metadata:
                    formatted_messages.append(
                        f"{msg.role.upper()} [{msg.timestamp}]: {msg.content}"
                    )
                else:
                    formatted_messages.append(
                        f"{msg.role.upper()}: {msg.content}"
                    )
            
            result = "\n\n".join(formatted_messages)
            self.logger.info(f"Read conversation history: {len(recent_messages)} messages")
            return result
            
        except Exception as e:
            self.logger.error(f"Error reading conversation history: {e}")
            return f"Error reading conversation history: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "conversation_history": {
                    "type": "object",
                    "description": "Conversation history object"
                },
                "max_messages": {
                    "type": "integer",
                    "description": "Maximum number of messages to include (0 for all)",
                    "default": 10
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Whether to include message metadata",
                    "default": False
                }
            },
            "required": ["conversation_history"]
        }


class AnalyzeConversationTool(BaseTool):
    """Tool for analyzing conversation patterns."""
    
    def __init__(self):
        super().__init__(
            name="analyze_conversation",
            description="Analyze conversation patterns and extract key information"
        )
    
    async def execute(
        self, 
        conversation_history: Optional[ConversationHistory] = None,
        analysis_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        Analyze conversation patterns.
        
        Args:
            conversation_history: Conversation history object
            analysis_type: Type of analysis to perform (summary, sentiment, topics)
            
        Returns:
            Analysis results
        """
        try:
            if not conversation_history:
                return {"error": "No conversation history provided."}
            
            if not hasattr(conversation_history, 'messages') or not conversation_history.messages:
                return {"error": "Conversation history is empty."}
            
            # Basic analysis
            message_count = len(conversation_history.messages)
            user_messages = sum(1 for msg in conversation_history.messages if msg.role == "user")
            assistant_messages = sum(1 for msg in conversation_history.messages if msg.role == "assistant")
            
            # Perform requested analysis
            if analysis_type == "summary":
                result = {
                    "message_count": message_count,
                    "user_messages": user_messages,
                    "assistant_messages": assistant_messages,
                    "conversation_turns": min(user_messages, assistant_messages),
                    "last_speaker": conversation_history.messages[-1].role if message_count > 0 else None,
                    "summary": "Conversation summary would be generated here with more advanced NLP"
                }
            elif analysis_type == "sentiment":
                # Simplified sentiment analysis
                result = {
                    "message_count": message_count,
                    "overall_sentiment": "neutral",  # Placeholder for real sentiment analysis
                    "sentiment_trend": "stable"      # Placeholder for real trend analysis
                }
            elif analysis_type == "topics":
                # Simplified topic extraction
                result = {
                    "message_count": message_count,
                    "main_topics": ["topic1", "topic2"],  # Placeholder for real topic extraction
                    "topic_shifts": []                    # Placeholder for real topic shift detection
                }
            else:
                result = {
                    "error": f"Unknown analysis type: {analysis_type}",
                    "supported_types": ["summary", "sentiment", "topics"]
                }
            
            self.logger.info(f"Analyzed conversation: {analysis_type} analysis on {message_count} messages")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation: {e}")
            return {"error": str(e)}
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "conversation_history": {
                    "type": "object",
                    "description": "Conversation history object"
                },
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis to perform",
                    "enum": ["summary", "sentiment", "topics"],
                    "default": "summary"
                }
            },
            "required": ["conversation_history"]
        }


# Test function
async def test_conversation_tools():
    """Test the conversation history tools."""
    print("Testing conversation history tools...")
    
    # Create mock conversation history
    from core.schemas import ConversationHistory, ConversationMessage
    from datetime import datetime
    
    # Create a mock conversation
    conversation = ConversationHistory(session_id="test_session")
    conversation.add_message("user", "Hello, can you help me write a story?")
    conversation.add_message("assistant", "Of course! What kind of story would you like?")
    conversation.add_message("user", "A science fiction story about time travel.")
    conversation.add_message("assistant", "Great choice! Here's a short sci-fi story about time travel...")
    
    # Test read conversation history tool
    read_tool = ReadConversationHistoryTool()
    analyze_tool = AnalyzeConversationTool()
    
    try:
        # Test reading history
        history_result = await read_tool.execute(conversation)
        print("Conversation History:")
        print(history_result)
        print("-" * 40)
        
        # Test analyzing history
        for analysis_type in ["summary", "sentiment", "topics"]:
            analysis_result = await analyze_tool.execute(conversation, analysis_type)
            print(f"{analysis_type.capitalize()} Analysis:")
            print(json.dumps(analysis_result, indent=2))
            print("-" * 40)
        
    except Exception as e:
        print(f"Error testing conversation tools: {e}")
    
    print("Conversation history tools tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_conversation_tools())
