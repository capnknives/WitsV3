"""
Code generation functionality
"""

import logging
from typing import Dict, List, Any, AsyncGenerator, Optional

from core.schemas import StreamData
from core.llm_interface import BaseLLMInterface


class CodeGenerator:
    """Generates code based on requirements using LLM"""
    
    def __init__(self, llm_interface: BaseLLMInterface):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.llm_interface = llm_interface
    
    async def generate_code(
        self,
        language: str,
        requirements: List[str],
        complexity: str = "medium",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate code based on requirements
        
        Args:
            language: Programming language
            requirements: List of requirements
            complexity: Code complexity level
            context: Additional context
            
        Returns:
            Generated code
        """
        prompt = self.build_code_prompt(language, requirements, complexity, context)
        
        try:
            response = await self.llm_interface.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            return response
        except Exception as e:
            self.logger.error(f"Error generating code: {e}")
            return self.generate_fallback_code(language)
    
    def build_code_prompt(
        self,
        language: str,
        requirements: List[str],
        complexity: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for code generation"""
        prompt = f"""Generate {complexity} {language} code to implement the following requirements:

Requirements:
{chr(10).join(f"- {req}" for req in requirements)}

Guidelines:
- Use best practices and design patterns
- Include proper error handling
- Add comprehensive comments
- Follow {language} style conventions
- Make code modular and reusable
- Include type hints/annotations where applicable
- Ensure code is production-ready

"""
        
        if context:
            if 'frameworks' in context:
                prompt += f"Frameworks to use: {', '.join(context['frameworks'])}\n"
            if 'patterns' in context:
                prompt += f"Design patterns to apply: {', '.join(context['patterns'])}\n"
            if 'constraints' in context:
                prompt += f"Constraints: {', '.join(context['constraints'])}\n"
        
        prompt += "\nProvide complete, working code with explanations."
        
        return prompt
    
    async def generate_function(
        self,
        language: str,
        function_name: str,
        description: str,
        parameters: List[Dict[str, str]],
        return_type: str
    ) -> str:
        """Generate a specific function"""
        prompt = f"""Generate a {language} function with the following specification:

Function Name: {function_name}
Description: {description}
Parameters: {', '.join(f"{p['name']}: {p['type']}" for p in parameters)}
Return Type: {return_type}

Requirements:
- Include docstring/documentation
- Add parameter validation
- Handle edge cases
- Include example usage

Generate the complete function implementation."""
        
        try:
            response = await self.llm_interface.generate(
                prompt=prompt,
                temperature=0.6,
                max_tokens=1000
            )
            return response
        except Exception as e:
            self.logger.error(f"Error generating function: {e}")
            return self.generate_fallback_function(language, function_name)
    
    async def generate_class(
        self,
        language: str,
        class_name: str,
        description: str,
        methods: List[Dict[str, Any]],
        attributes: List[Dict[str, str]]
    ) -> str:
        """Generate a class implementation"""
        prompt = f"""Generate a {language} class with the following specification:

Class Name: {class_name}
Description: {description}

Attributes:
{chr(10).join(f"- {attr['name']}: {attr['type']} - {attr.get('description', '')}" for attr in attributes)}

Methods:
{chr(10).join(f"- {method['name']}({method.get('params', '')}): {method.get('description', '')}" for method in methods)}

Requirements:
- Follow OOP best practices
- Include constructor/initializer
- Add appropriate access modifiers
- Include documentation
- Implement all specified methods

Generate the complete class implementation."""
        
        try:
            response = await self.llm_interface.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            return response
        except Exception as e:
            self.logger.error(f"Error generating class: {e}")
            return self.generate_fallback_class(language, class_name)
    
    async def refactor_code(
        self,
        code: str,
        language: str,
        improvements: List[str]
    ) -> str:
        """Refactor existing code"""
        prompt = f"""Refactor the following {language} code with these improvements:

Improvements requested:
{chr(10).join(f"- {imp}" for imp in improvements)}

Original code:
```{language}
{code}
```

Provide the refactored code with explanations of changes made."""
        
        try:
            response = await self.llm_interface.generate(
                prompt=prompt,
                temperature=0.6,
                max_tokens=2000
            )
            return response
        except Exception as e:
            self.logger.error(f"Error refactoring code: {e}")
            return code  # Return original if refactoring fails
    
    async def add_tests(
        self,
        code: str,
        language: str,
        test_framework: Optional[str] = None
    ) -> str:
        """Generate tests for given code"""
        if not test_framework:
            test_framework = self.get_default_test_framework(language)
        
        prompt = f"""Generate comprehensive tests for the following {language} code using {test_framework}:

Code to test:
```{language}
{code}
```

Requirements:
- Test all functions/methods
- Include edge cases
- Test error conditions
- Add setup/teardown if needed
- Use appropriate assertions
- Include test documentation

Generate complete test code."""
        
        try:
            response = await self.llm_interface.generate(
                prompt=prompt,
                temperature=0.6,
                max_tokens=1500
            )
            return response
        except Exception as e:
            self.logger.error(f"Error generating tests: {e}")
            return self.generate_fallback_tests(language, test_framework)
    
    async def add_documentation(
        self,
        code: str,
        language: str,
        doc_style: Optional[str] = None
    ) -> str:
        """Add documentation to code"""
        if not doc_style:
            doc_style = self.get_default_doc_style(language)
        
        prompt = f"""Add comprehensive documentation to the following {language} code using {doc_style} style:

Code:
```{language}
{code}
```

Requirements:
- Document all functions/methods
- Include parameter descriptions
- Document return values
- Add usage examples where helpful
- Include module/class level documentation

Provide the code with added documentation."""
        
        try:
            response = await self.llm_interface.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=2000
            )
            return response
        except Exception as e:
            self.logger.error(f"Error adding documentation: {e}")
            return code
    
    def get_default_test_framework(self, language: str) -> str:
        """Get default test framework for language"""
        frameworks = {
            'python': 'pytest',
            'javascript': 'jest',
            'java': 'junit',
            'csharp': 'nunit',
            'go': 'testing',
            'rust': 'cargo test'
        }
        return frameworks.get(language, 'unit tests')
    
    def get_default_doc_style(self, language: str) -> str:
        """Get default documentation style for language"""
        styles = {
            'python': 'Google docstrings',
            'javascript': 'JSDoc',
            'java': 'Javadoc',
            'csharp': 'XML documentation',
            'go': 'GoDoc',
            'rust': 'rustdoc'
        }
        return styles.get(language, 'inline comments')
    
    def generate_fallback_code(self, language: str) -> str:
        """Generate fallback code when LLM fails"""
        fallbacks = {
            'python': '''def main():
    """Main function"""
    print("Hello, World!")
    # Add your code here

if __name__ == "__main__":
    main()
''',
            'javascript': '''function main() {
    console.log("Hello, World!");
    // Add your code here
}

main();
'''
        }
        return fallbacks.get(language, '// Add your code here')
    
    def generate_fallback_function(self, language: str, function_name: str) -> str:
        """Generate fallback function"""
        if language == 'python':
            return f'''def {function_name}(*args, **kwargs):
    """
    Function: {function_name}
    
    Args:
        *args: Variable positional arguments
        **kwargs: Variable keyword arguments
        
    Returns:
        None
    """
    # Implement function logic here
    pass
'''
        else:
            return f'''function {function_name}() {{
    // Implement function logic here
}}
'''
    
    def generate_fallback_class(self, language: str, class_name: str) -> str:
        """Generate fallback class"""
        if language == 'python':
            return f'''class {class_name}:
    """Class: {class_name}"""
    
    def __init__(self):
        """Initialize {class_name}"""
        pass
    
    # Add methods here
'''
        else:
            return f'''class {class_name} {{
    constructor() {{
        // Initialize
    }}
    
    // Add methods here
}}
'''
    
    def generate_fallback_tests(self, language: str, framework: str) -> str:
        """Generate fallback tests"""
        if language == 'python':
            return f'''import pytest

def test_example():
    """Example test"""
    assert True
    
# Add more tests here
'''
        else:
            return '''describe('Tests', () => {
    test('example test', () => {
        expect(true).toBe(true);
    });
    
    // Add more tests here
});
'''