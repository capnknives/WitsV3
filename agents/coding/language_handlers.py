"""
Language-specific code generators coordinator
"""

import logging
from typing import List, Dict, Any

from .languages import PythonHandler, JavaScriptHandler


class LanguageHandlers:
    """Coordinates language-specific code generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.python_handler = PythonHandler()
        self.javascript_handler = JavaScriptHandler()
    
    # Python handlers
    async def generate_python_main(self) -> str:
        """Generate Python main.py file"""
        return await self.python_handler.generate_main()
    
    async def generate_python_web_app(self) -> str:
        """Generate Python web app with Flask"""
        return await self.python_handler.generate_web_app()
    
    async def generate_python_api(self) -> str:
        """Generate Python API with FastAPI"""
        return await self.python_handler.generate_api()
    
    async def generate_python_cli(self) -> str:
        """Generate Python CLI tool"""
        return await self.python_handler.generate_cli()
    
    async def generate_python_requirements(self, requirements: List[str]) -> str:
        """Generate Python requirements.txt"""
        return await self.python_handler.generate_requirements(requirements)
    
    async def generate_python_setup(self) -> str:
        """Generate Python setup.py"""
        return await self.python_handler.generate_setup()
    
    async def generate_python_tests(self) -> str:
        """Generate Python test file"""
        return await self.python_handler.generate_tests()
    
    # JavaScript handlers
    async def generate_js_main(self) -> str:
        """Generate JavaScript index.js"""
        return await self.javascript_handler.generate_main()
    
    async def generate_js_web_app(self) -> str:
        """Generate JavaScript web app with Express"""
        return await self.javascript_handler.generate_web_app()
    
    async def generate_js_api(self) -> str:
        """Generate JavaScript API with Express"""
        return await self.javascript_handler.generate_api()
    
    async def generate_js_package_json(self, project_type: str) -> str:
        """Generate package.json for JavaScript projects"""
        return await self.javascript_handler.generate_package_json(project_type)
    
    async def generate_js_tests(self) -> str:
        """Generate JavaScript test file"""
        return await self.javascript_handler.generate_tests()