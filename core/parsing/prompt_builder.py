"""
Prompt building utilities for structured LLM responses
"""

from typing import List, Dict, Any


def create_structured_prompt(tools: List[Dict[str, Any]], format_type: str = "json") -> str:
    """
    Create a structured prompt for tool usage
    
    Args:
        tools: List of available tools with their schemas
        format_type: Format for tool calls ("json", "function", "xml", "markdown", "react")
        
    Returns:
        Formatted prompt string
    """
    format_builders = {
        "json": _create_json_prompt,
        "function": _create_function_prompt,
        "xml": _create_xml_prompt,
        "markdown": _create_markdown_prompt,
        "react": _create_react_prompt,
    }
    
    builder = format_builders.get(format_type, _create_json_prompt)
    return builder(tools)


def _create_json_prompt(tools: List[Dict[str, Any]]) -> str:
    """Create JSON format prompt"""
    tool_list = "\n".join([
        f"- {tool['name']}: {tool.get('description', '')}" 
        for tool in tools
    ])
    
    return f"""You have access to the following tools:
{tool_list}

To use a tool, respond with a JSON object in this format:
```json
{{
    "tool": "tool_name",
    "arguments": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}
```

Use the Thought-Action-Observation pattern:
- Thought: Explain your reasoning
- Action: Use a tool or provide final answer
- Observation: Note the result and plan next steps

For your final answer, start with "Final Answer:"
"""


def _create_function_prompt(tools: List[Dict[str, Any]]) -> str:
    """Create function call format prompt"""
    tool_list = "\n".join([
        f"- {tool['name']}: {tool.get('description', '')}"
        for tool in tools
    ])
    
    return f"""You have access to the following tools:
{tool_list}

To use a tool, call it like a function:
tool_name(param1="value1", param2="value2")

Use the Thought-Action-Observation pattern:
- Thought: Explain your reasoning
- Action: Call a tool or provide final answer
- Observation: Note the result and plan next steps

For your final answer, start with "Final Answer:"
"""


def _create_xml_prompt(tools: List[Dict[str, Any]]) -> str:
    """Create XML format prompt"""
    tool_list = "\n".join([
        f"- {tool['name']}: {tool.get('description', '')}"
        for tool in tools
    ])
    
    return f"""You have access to the following tools:
{tool_list}

To use a tool, use XML format:
<tool name="tool_name">
    <argument name="param1">value1</argument>
    <argument name="param2">value2</argument>
</tool>

Use the Thought-Action-Observation pattern:
- Thought: Explain your reasoning
- Action: Use a tool or provide final answer
- Observation: Note the result and plan next steps

For your final answer, start with "Final Answer:"
"""


def _create_markdown_prompt(tools: List[Dict[str, Any]]) -> str:
    """Create markdown format prompt"""
    tool_list = "\n".join([
        f"- {tool['name']}: {tool.get('description', '')}"
        for tool in tools
    ])
    
    return f"""You have access to the following tools:
{tool_list}

To use a tool, use markdown format:
[tool:tool_name](param1="value1", param2="value2")

Use the Thought-Action-Observation pattern:
- Thought: Explain your reasoning
- Action: Use a tool or provide final answer
- Observation: Note the result and plan next steps

For your final answer, start with "Final Answer:"
"""


def _create_react_prompt(tools: List[Dict[str, Any]]) -> str:
    """Create ReAct format prompt"""
    tool_list = "\n".join([
        f"- {tool['name']}: {tool.get('description', '')}"
        for tool in tools
    ])
    
    # Generate parameter descriptions
    tool_details = []
    for tool in tools:
        params = tool.get('parameters', {})
        param_desc = ", ".join([
            f"{name}: {info.get('type', 'any')}"
            for name, info in params.items()
        ])
        tool_details.append(f"{tool['name']}({param_desc})")
    
    tool_signatures = "\n".join(tool_details)
    
    return f"""You are an AI agent using the ReAct (Reason-Act-Observe) framework.

Available Tools:
{tool_list}

Tool Signatures:
{tool_signatures}

Follow this format EXACTLY:

Thought: [Your reasoning about what to do next]
Action: [tool_name(param1="value1", param2="value2")]
Observation: [This will be filled by the system]

Continue with Thought-Action-Observation cycles until you can provide a final answer.

When you have the final answer:
Thought: [Final reasoning]
Final Answer: [Your complete answer to the user]

Remember:
- Always start with a Thought
- Actions must be valid tool calls with proper syntax
- Wait for Observations before proceeding
- Provide a clear Final Answer when done
"""


def create_tool_description(tool: Dict[str, Any]) -> str:
    """
    Create a detailed description of a tool for LLM understanding
    
    Args:
        tool: Tool schema dictionary
        
    Returns:
        Formatted tool description
    """
    name = tool.get('name', 'unknown')
    description = tool.get('description', 'No description available')
    parameters = tool.get('parameters', {})
    
    # Format parameters
    param_lines = []
    for param_name, param_info in parameters.items():
        param_type = param_info.get('type', 'any')
        param_desc = param_info.get('description', '')
        required = param_info.get('required', False)
        
        param_line = f"  - {param_name} ({param_type})"
        if required:
            param_line += " [REQUIRED]"
        if param_desc:
            param_line += f": {param_desc}"
        
        param_lines.append(param_line)
    
    # Build complete description
    tool_desc = f"{name}: {description}"
    if param_lines:
        tool_desc += "\n  Parameters:\n" + "\n".join(param_lines)
    
    return tool_desc