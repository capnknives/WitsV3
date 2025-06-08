"""
Response parsing utilities for WitsV3 - Compatibility wrapper

This module maintains backward compatibility while using the new modular parsing system.
The original 605-line file has been split into:
- parsing/base_parser.py (88 lines)
- parsing/json_parser.py (149 lines)
- parsing/react_parser.py (187 lines)
- parsing/format_detector.py (156 lines)
- parsing/parser_factory.py (182 lines)
- parsing/prompt_builder.py (193 lines)
"""

# Import everything from the new parsing module for backward compatibility
from .parsing import (
    # Types and classes
    ResponseType,
    ParsedResponse,
    
    # Parsers
    BaseParser,
    JSONParser,
    ReactParser,
    
    # Factory and utilities
    ParserFactory,
    FormatDetector,
    create_response_parser,
    create_structured_prompt,
)

from .parsing.parser_factory import GeneralParser


# Create the main ResponseParser class for backward compatibility
class ResponseParser(GeneralParser):
    """
    Main response parser for backward compatibility.
    
    This class maintains the same interface as the original ResponseParser
    but uses the new modular parsing system underneath.
    """
    
    def __init__(self):
        super().__init__()
        # Maintain all the original patterns for compatibility
        self.tool_call_patterns = [
            r'```json\s*(\{[^`]+\})\s*```',
            r'\{[^}]*"tool"[^}]*\}',
            r'(\w+)\s*\(\s*([^)]*)\s*\)',
            r'<tool\s+name="([^"]+)"[^>]*>(.*?)</tool>',
            r'\[tool:([^\]]+)\]\(([^)]*)\)',
        ]
        
        self.reasoning_patterns = [
            r'(?:Thought|Reasoning|Think):\s*(.+?)(?=\n(?:Action|Tool|Observation|$))',
            r'(?:Let me think|I need to|I should)\s*(.+?)(?=\n|$)',
        ]
        
        self.observation_patterns = [
            r'(?:Observation|Result|Output):\s*(.+?)(?=\n(?:Thought|Action|$)|$)',
        ]
        
        self.final_answer_patterns = [
            r'(?:Final Answer|Answer|Conclusion):\s*(.+?)(?=$)',
            r'(?:Therefore|In conclusion|To summarize):\s*(.+?)(?=$)',
        ]


# Specialized ReactParser is already available from imports
# Just need to ensure it has the parse_react_response method
# (which it does in the new implementation)


# Re-export test function for compatibility
async def test_response_parser():
    """Test the response parser functionality."""
    print("Testing Response Parser (Compatibility Mode)...")
    
    # Test basic parsing
    parser = ResponseParser()
    
    test_responses = [
        '{"tool": "search", "arguments": {"query": "test"}}',
        'Thought: I need to search\nAction: search(query="test")',
        'Final Answer: The test is complete.',
    ]
    
    for response in test_responses:
        parsed = parser.parse_response(response)
        print(f"✓ Parsed: {parsed.response_type} - {parsed.content[:50]}...")
    
    # Test format detection
    detector = FormatDetector()
    for response in test_responses:
        format_type = detector.detect_format(response)
        print(f"✓ Detected format: {format_type}")
    
    # Test prompt creation
    tools = [
        {"name": "search", "description": "Search the web"},
        {"name": "calculate", "description": "Perform calculations"},
    ]
    
    for format_type in ["json", "react", "function"]:
        prompt = create_structured_prompt(tools, format_type)
        print(f"✓ Created {format_type} prompt (length: {len(prompt)})")
    
    print("Response Parser tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_response_parser())
