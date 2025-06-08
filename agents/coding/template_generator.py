"""
Template generator coordinator - delegates to specialized template modules
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from .templates import DocumentationTemplates, ConfigTemplates, CITemplates


class TemplateGenerator:
    """Coordinates template generation across different template modules"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.documentation_templates = DocumentationTemplates()
        self.config_templates = ConfigTemplates()
        self.ci_templates = CITemplates()
    
    # Documentation templates delegation
    async def generate_readme(
        self,
        project_type: str,
        language: str,
        requirements: List[str],
        project_name: str = "Project"
    ) -> str:
        """Generate README.md file"""
        return await self.documentation_templates.generate_readme(
            project_type, language, requirements, project_name
        )
    
    async def generate_license(self, author: str = "Your Name") -> str:
        """Generate MIT license"""
        return await self.documentation_templates.generate_license(author)
    
    async def generate_contributing(self, project_name: str) -> str:
        """Generate CONTRIBUTING.md"""
        return await self.documentation_templates.generate_contributing(project_name)
    
    # Configuration templates delegation
    async def generate_gitignore(self, language: str) -> str:
        """Generate .gitignore file"""
        return await self.config_templates.generate_gitignore(language)
    
    async def generate_dockerfile(self, language: str, project_type: str) -> str:
        """Generate Dockerfile"""
        return await self.config_templates.generate_dockerfile(language, project_type)
    
    async def generate_makefile(self, language: str) -> str:
        """Generate Makefile"""
        return await self.config_templates.generate_makefile(language)
    
    # CI/CD templates delegation
    async def generate_github_actions(self, language: str) -> str:
        """Generate GitHub Actions workflow"""
        return await self.ci_templates.generate_github_actions(language)
    
    # Additional template methods for completeness
    async def generate_gitlab_ci(self, language: str) -> str:
        """Generate GitLab CI configuration"""
        return await self.ci_templates.generate_gitlab_ci(language)
    
    async def generate_travis_ci(self, language: str) -> str:
        """Generate Travis CI configuration"""
        return await self.ci_templates.generate_travis_ci(language)
    
    async def generate_jenkinsfile(self, language: str) -> str:
        """Generate Jenkinsfile"""
        return await self.ci_templates.generate_jenkinsfile(language)
    
    # Utility methods
    async def generate_all_config_files(
        self,
        language: str,
        project_type: str,
        project_name: str = "Project",
        author: str = "Your Name"
    ) -> Dict[str, str]:
        """Generate all configuration files for a project"""
        configs = {}
        
        # Documentation files
        configs['.gitignore'] = await self.generate_gitignore(language)
        configs['LICENSE'] = await self.generate_license(author)
        configs['CONTRIBUTING.md'] = await self.generate_contributing(project_name)
        
        # Build files
        configs['Makefile'] = await self.generate_makefile(language)
        configs['Dockerfile'] = await self.generate_dockerfile(language, project_type)
        
        # CI/CD files
        configs['.github/workflows/ci.yml'] = await self.generate_github_actions(language)
        
        return configs
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return ['python', 'javascript', 'java', 'rust', 'go']
    
    def get_supported_ci_systems(self) -> List[str]:
        """Get list of supported CI/CD systems"""
        return ['github_actions', 'gitlab_ci', 'travis_ci', 'jenkins']