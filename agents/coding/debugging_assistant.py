"""
Debugging, optimization, and refactoring assistance
"""

import logging
from typing import Dict, List, Any, Optional

from core.llm_interface import BaseLLMInterface


class DebuggingAssistant:
    """Assists with debugging, optimization, and refactoring"""
    
    def __init__(self, llm_interface: BaseLLMInterface):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.llm_interface = llm_interface
    
    async def create_debugging_guide(self, language: str) -> str:
        """Create comprehensive debugging guide for a language"""
        prompt = f"""Create a comprehensive debugging guide for {language} applications.

Cover:
1. Common error types and their solutions
2. Debugging tools and techniques
3. Logging best practices
4. Performance profiling methods
5. Memory leak detection
6. Concurrency issues
7. Step-by-step debugging process
8. Prevention strategies
9. IDE debugging features
10. Command-line debugging tools

Provide practical examples and actionable advice."""
        
        try:
            response = await self.llm_interface.generate(
                prompt=prompt,
                temperature=0.6,
                max_tokens=2000
            )
            return response
        except Exception as e:
            self.logger.error(f"Error creating debugging guide: {e}")
            return self.get_fallback_debugging_guide(language)
    
    async def analyze_error(
        self,
        error_message: str,
        code_context: str,
        language: str
    ) -> Dict[str, Any]:
        """Analyze an error and provide solutions"""
        prompt = f"""Analyze this {language} error and provide solutions:

Error Message:
{error_message}

Code Context:
```{language}
{code_context}
```

Provide:
1. Error explanation
2. Root cause analysis
3. Step-by-step solution
4. Code fix (if applicable)
5. Prevention tips
6. Related common issues"""
        
        try:
            response = await self.llm_interface.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1500
            )
            
            return {
                "analysis": response,
                "error_type": self.classify_error(error_message),
                "severity": self.assess_severity(error_message)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing error: {e}")
            return {
                "analysis": "Unable to analyze error",
                "error_type": "unknown",
                "severity": "medium"
            }
    
    async def optimize_code(
        self,
        code: str,
        language: str,
        optimization_goals: List[str]
    ) -> str:
        """Optimize code for performance"""
        goals_text = '\n'.join(f"- {goal}" for goal in optimization_goals)
        
        prompt = f"""Optimize the following {language} code for these goals:

Optimization Goals:
{goals_text}

Original Code:
```{language}
{code}
```

Provide:
1. Optimized code
2. Explanation of optimizations made
3. Performance impact estimates
4. Trade-offs considered
5. Further optimization opportunities"""
        
        try:
            response = await self.llm_interface.generate(
                prompt=prompt,
                temperature=0.6,
                max_tokens=2000
            )
            return response
        except Exception as e:
            self.logger.error(f"Error optimizing code: {e}")
            return code
    
    async def suggest_refactoring(
        self,
        code: str,
        language: str,
        code_smells: Optional[List[str]] = None
    ) -> str:
        """Suggest refactoring improvements"""
        prompt = f"""Analyze this {language} code and suggest refactoring improvements:

Code:
```{language}
{code}
```
"""
        
        if code_smells:
            prompt += f"\nIdentified issues:\n{chr(10).join(f'- {smell}' for smell in code_smells)}\n"
        
        prompt += """
Focus on:
1. Code readability
2. Maintainability
3. Design patterns
4. SOLID principles
5. DRY principle
6. Performance
7. Error handling
8. Testing considerations

Provide refactored code with explanations."""
        
        try:
            response = await self.llm_interface.generate(
                prompt=prompt,
                temperature=0.6,
                max_tokens=2000
            )
            return response
        except Exception as e:
            self.logger.error(f"Error suggesting refactoring: {e}")
            return "Unable to analyze code for refactoring"
    
    async def generate_performance_profile(
        self,
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """Generate performance analysis"""
        prompt = f"""Analyze the performance characteristics of this {language} code:

Code:
```{language}
{code}
```

Provide analysis of:
1. Time complexity (Big O)
2. Space complexity
3. Potential bottlenecks
4. Memory usage patterns
5. I/O operations
6. CPU-intensive operations
7. Optimization opportunities
8. Scalability considerations"""
        
        try:
            response = await self.llm_interface.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1500
            )
            
            return {
                "analysis": response,
                "complexity": self.estimate_complexity(code),
                "bottlenecks": self.identify_bottlenecks(code)
            }
        except Exception as e:
            self.logger.error(f"Error generating performance profile: {e}")
            return {
                "analysis": "Performance analysis unavailable",
                "complexity": "O(n)",
                "bottlenecks": []
            }
    
    def classify_error(self, error_message: str) -> str:
        """Classify error type"""
        error_lower = error_message.lower()
        
        if any(word in error_lower for word in ['syntax', 'parse', 'unexpected']):
            return "syntax_error"
        elif any(word in error_lower for word in ['type', 'cannot read', 'undefined']):
            return "type_error"
        elif any(word in error_lower for word in ['reference', 'not defined', 'name error']):
            return "reference_error"
        elif any(word in error_lower for word in ['index', 'range', 'bounds']):
            return "index_error"
        elif any(word in error_lower for word in ['memory', 'heap', 'stack']):
            return "memory_error"
        elif any(word in error_lower for word in ['permission', 'access', 'denied']):
            return "permission_error"
        elif any(word in error_lower for word in ['network', 'connection', 'timeout']):
            return "network_error"
        else:
            return "runtime_error"
    
    def assess_severity(self, error_message: str) -> str:
        """Assess error severity"""
        error_lower = error_message.lower()
        
        if any(word in error_lower for word in ['fatal', 'crash', 'core dump', 'segfault']):
            return "critical"
        elif any(word in error_lower for word in ['error', 'exception', 'failed']):
            return "high"
        elif any(word in error_lower for word in ['warning', 'deprecated']):
            return "medium"
        else:
            return "low"
    
    def estimate_complexity(self, code: str) -> str:
        """Estimate time complexity"""
        code_lower = code.lower()
        
        # Simple heuristics
        loop_count = code_lower.count('for') + code_lower.count('while')
        
        if 'for' in code_lower and 'for' in code_lower[code_lower.find('for')+3:]:
            return "O(n²) or higher"
        elif loop_count > 0:
            return "O(n)"
        else:
            return "O(1)"
    
    def identify_bottlenecks(self, code: str) -> List[str]:
        """Identify potential bottlenecks"""
        bottlenecks = []
        code_lower = code.lower()
        
        # Common bottleneck patterns
        if 'sleep' in code_lower or 'time.sleep' in code_lower:
            bottlenecks.append("Blocking sleep operations")
        
        if any(db in code_lower for db in ['select', 'insert', 'update', 'delete']):
            bottlenecks.append("Database operations")
        
        if any(io in code_lower for io in ['open(', 'read(', 'write(']):
            bottlenecks.append("File I/O operations")
        
        if 'request' in code_lower or 'http' in code_lower:
            bottlenecks.append("Network requests")
        
        if code_lower.count('for') > 2:
            bottlenecks.append("Multiple nested loops")
        
        return bottlenecks
    
    def get_fallback_debugging_guide(self, language: str) -> str:
        """Get fallback debugging guide"""
        return f"""# {language} Debugging Guide

## Common Debugging Techniques

1. **Print Debugging**
   - Add print statements to trace execution
   - Log variable values at key points

2. **Use a Debugger**
   - Set breakpoints
   - Step through code line by line
   - Inspect variables

3. **Error Messages**
   - Read error messages carefully
   - Look at stack traces
   - Identify the exact line causing issues

4. **Divide and Conquer**
   - Comment out sections to isolate problems
   - Test individual functions separately
   - Use unit tests

5. **Check Common Issues**
   - Off-by-one errors
   - Null/undefined references
   - Type mismatches
   - Scope issues

6. **Performance Debugging**
   - Use profiling tools
   - Measure execution time
   - Monitor memory usage

7. **Best Practices**
   - Write defensive code
   - Add error handling
   - Use logging instead of print
   - Write tests first (TDD)
"""