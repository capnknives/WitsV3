"""
Project management functionality
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import CodeProject, PROJECT_TEMPLATES
from .template_generator import TemplateGenerator
from .language_handlers import LanguageHandlers


class ProjectManager:
    """Manages coding projects and their lifecycle"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.projects: Dict[str, CodeProject] = {}
        self.template_generator = TemplateGenerator()
        self.language_handlers = LanguageHandlers()
    
    async def create_project(
        self,
        language: str,
        project_type: str,
        requirements: List[str],
        name: Optional[str] = None
    ) -> CodeProject:
        """
        Create a new coding project
        
        Args:
            language: Programming language
            project_type: Type of project (web_app, api, cli_tool, library)
            requirements: List of requirements
            name: Optional project name
            
        Returns:
            Created CodeProject
        """
        project_id = str(uuid.uuid4())
        
        if not name:
            name = f"{project_type}_{language}_project"
        
        # Generate project structure
        structure = await self.generate_project_structure(project_type, language, requirements)
        
        # Generate initial files
        initial_files = await self.generate_initial_files(structure, language, project_type, requirements)
        
        # Create project
        project = CodeProject(
            id=project_id,
            name=name,
            description=' '.join(requirements),
            language=language,
            project_type=project_type,
            structure=structure,
            dependencies=self.extract_dependencies(requirements, language),
            files=initial_files,
            tests={},
            documentation=""
        )
        
        # Store project
        self.projects[project_id] = project
        
        self.logger.info(f"Created project '{name}' with ID {project_id}")
        return project
    
    async def generate_project_structure(
        self,
        project_type: str,
        language: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Generate project directory structure"""
        base_template = PROJECT_TEMPLATES.get(project_type, PROJECT_TEMPLATES['library'])
        
        structure = {
            'directories': base_template['structure'].copy(),
            'files': base_template['files'].copy()
        }
        
        # Language-specific adjustments
        if language == 'python':
            if project_type == 'library':
                structure['directories'].append('src/package_name/')
                structure['files'].append('src/package_name/__init__.py')
        elif language == 'javascript':
            structure['directories'].append('public/')
            structure['files'].append('.eslintrc.json')
        elif language in ['rust', 'go']:
            if 'src/' not in structure['directories']:
                structure['directories'].insert(0, 'src/')
        
        # Add CI/CD if mentioned in requirements
        if any('ci' in req.lower() or 'deploy' in req.lower() for req in requirements):
            structure['directories'].append('.github/workflows/')
            structure['files'].append('.github/workflows/ci.yml')
        
        return structure
    
    async def generate_initial_files(
        self,
        structure: Dict[str, Any],
        language: str,
        project_type: str,
        requirements: List[str]
    ) -> Dict[str, str]:
        """Generate initial project files"""
        files = {}
        
        # Generate main application file
        if language == 'python':
            if project_type == 'web_app':
                files['app.py'] = await self.language_handlers.generate_python_web_app()
            elif project_type == 'api':
                files['app.py'] = await self.language_handlers.generate_python_api()
            elif project_type == 'cli_tool':
                files['cli.py'] = await self.language_handlers.generate_python_cli()
            else:
                files['main.py'] = await self.language_handlers.generate_python_main()
            
            files['requirements.txt'] = await self.language_handlers.generate_python_requirements(requirements)
            files['setup.py'] = await self.language_handlers.generate_python_setup()
            files['tests/test_main.py'] = await self.language_handlers.generate_python_tests()
            
        elif language == 'javascript':
            if project_type == 'web_app':
                files['index.js'] = await self.language_handlers.generate_js_web_app()
            elif project_type == 'api':
                files['index.js'] = await self.language_handlers.generate_js_api()
            else:
                files['index.js'] = await self.language_handlers.generate_js_main()
            
            files['package.json'] = await self.language_handlers.generate_js_package_json(project_type)
            files['tests/index.test.js'] = await self.language_handlers.generate_js_tests()
        
        # Generate common files
        files['README.md'] = await self.template_generator.generate_readme(
            project_type, language, requirements
        )
        files['.gitignore'] = await self.template_generator.generate_gitignore(language)
        files['LICENSE'] = await self.template_generator.generate_license()
        files['CONTRIBUTING.md'] = await self.template_generator.generate_contributing(
            structure.get('name', 'Project')
        )
        
        # Generate CI/CD files if needed
        if '.github/workflows/ci.yml' in structure.get('files', []):
            files['.github/workflows/ci.yml'] = await self.template_generator.generate_github_actions(language)
        
        # Generate Dockerfile if containerization is mentioned
        if any('docker' in req.lower() or 'container' in req.lower() for req in requirements):
            files['Dockerfile'] = await self.template_generator.generate_dockerfile(language, project_type)
            files['docker-compose.yml'] = await self.generate_docker_compose(project_type)
        
        # Generate Makefile for convenient commands
        files['Makefile'] = await self.template_generator.generate_makefile(language)
        
        return files
    
    def extract_dependencies(self, requirements: List[str], language: str) -> List[str]:
        """Extract dependencies from requirements"""
        dependencies = []
        
        # Common framework keywords
        framework_map = {
            'python': {
                'web': ['flask', 'django', 'fastapi'],
                'api': ['fastapi', 'flask-restful', 'django-rest-framework'],
                'data': ['pandas', 'numpy', 'scipy'],
                'ml': ['tensorflow', 'pytorch', 'scikit-learn']
            },
            'javascript': {
                'web': ['react', 'vue', 'angular'],
                'api': ['express', 'koa', 'hapi'],
                'test': ['jest', 'mocha', 'cypress']
            }
        }
        
        if language in framework_map:
            for category, frameworks in framework_map[language].items():
                for framework in frameworks:
                    if any(framework in req.lower() for req in requirements):
                        dependencies.append(framework)
        
        return dependencies
    
    async def generate_docker_compose(self, project_type: str) -> str:
        """Generate docker-compose.yml"""
        base_compose = '''version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=development
    volumes:
      - .:/app
'''
        
        if project_type == 'api':
            base_compose += '''  
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
'''
        
        return base_compose
    
    def get_project(self, project_id: str) -> Optional[CodeProject]:
        """Get a project by ID"""
        return self.projects.get(project_id)
    
    def list_projects(self) -> List[CodeProject]:
        """List all projects"""
        return list(self.projects.values())
    
    def update_project_status(self, project_id: str, status: str) -> bool:
        """Update project status"""
        if project_id in self.projects:
            self.projects[project_id].status = status
            return True
        return False
    
    def add_file_to_project(self, project_id: str, filename: str, content: str) -> bool:
        """Add a file to a project"""
        if project_id in self.projects:
            self.projects[project_id].add_file(filename, content)
            return True
        return False
    
    def add_test_to_project(self, project_id: str, test_filename: str, content: str) -> bool:
        """Add a test file to a project"""
        if project_id in self.projects:
            self.projects[project_id].add_test(test_filename, content)
            return True
        return False
    
    def get_project_statistics(self) -> Dict[str, Any]:
        """Get statistics across all projects"""
        stats = {
            'total_projects': len(self.projects),
            'languages_used': set(),
            'project_types': set(),
            'total_files': 0,
            'total_lines_of_code': 0,
            'projects_by_status': {}
        }
        
        for project in self.projects.values():
            stats['languages_used'].add(project.language)
            stats['project_types'].add(project.project_type)
            stats['total_files'] += project.get_file_count()
            stats['total_lines_of_code'] += project.get_lines_of_code()
            
            status = project.status
            stats['projects_by_status'][status] = stats['projects_by_status'].get(status, 0) + 1
        
        # Convert sets to lists for JSON serialization
        stats['languages_used'] = list(stats['languages_used'])
        stats['project_types'] = list(stats['project_types'])
        
        return stats