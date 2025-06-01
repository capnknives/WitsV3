# tools/intent_analysis_tool.py
"""
Intent Analysis Tool for WitsV3.
Analyzes user input to determine intent and appropriate response strategy.
"""

import json
from typing import Any, Dict, Optional

from core.base_tool import BaseTool


class IntentAnalysisTool(BaseTool):
    """Tool for analyzing user intent."""
    
    def __init__(self):
        super().__init__(
            name="intent_analysis",
            description="Analyze user input to determine intent and response strategy"
        )
    
    async def execute(self, input_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze user intent.
        
        Args:
            input_text: User input to analyze
            context: Optional conversation context
            
        Returns:
            Intent analysis result
        """
        try:
            # This is a simplified implementation
            # In a real system, this would use more sophisticated NLP
            
            input_lower = input_text.lower()
            
            # Simple intent detection based on keywords
            intent_type = "direct_response"  # Default
            confidence = 0.5
            goal_statement = input_text
            
            # Check for task/goal patterns
            task_keywords = ["create", "make", "build", "write", "generate", 
                            "develop", "implement", "design", "fix", "solve"]
            
            if any(keyword in input_lower for keyword in task_keywords):
                intent_type = "goal_defined"
                confidence = 0.7
            
            # Check for question patterns
            question_keywords = ["what", "how", "why", "when", "where", "who", "can you", "could you"]
            
            if any(keyword in input_lower for keyword in question_keywords):
                if "?" in input_text:
                    intent_type = "direct_response"
                    confidence = 0.8
            
            # Check for clarification needs
            ambiguous_keywords = ["it", "that", "this", "they", "them", "those", "something"]
            
            if any(keyword in input_lower.split() for keyword in ambiguous_keywords) and len(input_text.split()) < 5:
                intent_type = "clarification_question"
                confidence = 0.6
            
            # Prepare response
            result = {
                "type": intent_type,
                "confidence": confidence,
                "goal_statement": goal_statement if intent_type == "goal_defined" else None,
                "direct_response": None,
                "clarification_question": "Could you please provide more details?" if intent_type == "clarification_question" else None,
                "reasoning": f"Detected intent type: {intent_type} based on keyword analysis"
            }
            
            self.logger.info(f"Analyzed intent: {intent_type} (confidence: {confidence})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing intent: {e}")
            return {
                "type": "error",
                "error": str(e),
                "confidence": 0.0
            }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "User input to analyze"
                },
                "context": {
                    "type": "string",
                    "description": "Optional conversation context",
                    "default": ""
                }
            },
            "required": ["input_text"]
        }


# Test function
async def test_intent_analysis_tool():
    """Test the intent analysis tool."""
    print("Testing IntentAnalysisTool...")
    
    tool = IntentAnalysisTool()
    
    # Test cases
    test_inputs = [
        "Write me a story about dragons",
        "What is the capital of France?",
        "Can you help me with this?",
        "Fix the bug in my code"
    ]
    
    for input_text in test_inputs:
        try:
            result = await tool.execute(input_text)
            print(f"Input: '{input_text}'")
            print(f"Result: {json.dumps(result, indent=2)}")
            print("-" * 40)
        except Exception as e:
            print(f"Error testing '{input_text}': {e}")
    
    print("IntentAnalysisTool tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_intent_analysis_tool())
