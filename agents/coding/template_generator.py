"""
Template generator for various file types
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime


class TemplateGenerator:
    """Generates file templates for different project types and languages"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def generate_readme(
        self,
        project_type: str,
        language: str,
        requirements: List[str],
        project_name: str = "Project"
    ) -> str:
        """Generate README.md file"""
        return f'''# {project_name}

## Description
{language.title()} {project_type} implementing: {', '.join(requirements)}

## Features
- {chr(10).join(f"- {req}" for req in requirements[:5])}

## Installation

### Prerequisites
- {language.title()} installed
- Package manager (pip, npm, cargo, etc.)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/{project_name.lower().replace(" ", "-")}.git
cd {project_name.lower().replace(" ", "-")}

# Install dependencies
{"pip install -r requirements.txt" if language == "python" else "npm install" if language == "javascript" else "cargo build"}
```

## Usage
```bash
# Run the application
{"python main.py" if language == "python" else "npm start" if language == "javascript" else "cargo run"}

# Run tests
{"pytest" if language == "python" else "npm test" if language == "javascript" else "cargo test"}
```

## Project Structure
```
{project_name}/
├── {"src/" if language in ["rust", "java"] else ""}
├── tests/
├── docs/
├── README.md
└── {"requirements.txt" if language == "python" else "package.json" if language == "javascript" else "Cargo.toml"}
```

## API Documentation
[API documentation will be added here]

## Development

### Running Tests
```bash
{"pytest -v" if language == "python" else "npm test" if language == "javascript" else "cargo test"}
```

### Code Style
This project follows {language} best practices and coding standards.

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Built with {language.title()}
- Created on {datetime.now().strftime("%Y-%m-%d")}
'''
    
    async def generate_gitignore(self, language: str) -> str:
        """Generate .gitignore file"""
        common = '''# General
.DS_Store
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
*.log
logs/
temp/
tmp/
.idea/
.vscode/
*.swp
*.swo
*~
'''
        
        language_specific = {
            'python': '''
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/
.mypy_cache/
.dmypy.json
dmypy.json
venv/
ENV/
env/
''',
            'javascript': '''
# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*
.npm
.eslintcache
.node_repl_history
*.tgz
.yarn-integrity
.env.development
.env.test
.env.production
build/
dist/
.cache/
.parcel-cache/
.next/
out/
.nuxt/
.vuepress/dist/
.serverless/
.fusebox/
.dynamodb/
.tern-port
.pnp
.pnp.js
''',
            'java': '''
# Java
*.class
*.jar
*.war
*.ear
*.iml
target/
out/
.gradle/
build/
gradle-app.setting
!gradle-wrapper.jar
.gradletasknamecache
hs_err_pid*
''',
            'rust': '''
# Rust
/target/
**/*.rs.bk
*.pdb
Cargo.lock
''',
            'go': '''
# Go
*.exe
*.exe~
*.dll
*.so
*.dylib
*.test
*.out
vendor/
go.sum
'''
        }
        
        return common + language_specific.get(language, '')
    
    async def generate_license(self, author: str = "Your Name") -> str:
        """Generate MIT license"""
        year = datetime.now().year
        return f'''MIT License

Copyright (c) {year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
    
    async def generate_contributing(self, project_name: str) -> str:
        """Generate CONTRIBUTING.md"""
        return f'''# Contributing to {project_name}

First off, thank you for considering contributing to {project_name}!

## How Can I Contribute?

### Reporting Bugs
- Use the issue tracker to report bugs
- Describe the bug in detail
- Include steps to reproduce
- Include expected vs actual behavior
- Add screenshots if applicable

### Suggesting Enhancements
- Use the issue tracker for suggestions
- Provide a clear description of the enhancement
- Explain why this would be useful
- Include examples of how it would work

### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add tests for new functionality
4. Follow the existing code style
5. Keep commits atomic and well-described

## Code Style
- Follow language-specific conventions
- Use meaningful variable and function names
- Comment complex logic
- Keep functions small and focused

## Questions?
Feel free to open an issue for any questions!
'''
    
    async def generate_dockerfile(self, language: str, project_type: str) -> str:
        """Generate Dockerfile"""
        dockerfiles = {
            'python': f'''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

{"EXPOSE 8000" if project_type in ["api", "web_app"] else ""}

CMD ["python", "main.py"]
''',
            'javascript': f'''FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

{"EXPOSE 3000" if project_type in ["api", "web_app"] else ""}

CMD ["node", "index.js"]
''',
            'java': '''FROM openjdk:17-jdk-slim

WORKDIR /app

COPY target/*.jar app.jar

EXPOSE 8080

CMD ["java", "-jar", "app.jar"]
''',
            'rust': '''FROM rust:1.70 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bullseye-slim
COPY --from=builder /app/target/release/app /usr/local/bin/app

CMD ["app"]
''',
            'go': '''FROM golang:1.20-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o main .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/

COPY --from=builder /app/main .

CMD ["./main"]
'''
        }
        
        return dockerfiles.get(language, dockerfiles['python'])
    
    async def generate_makefile(self, language: str) -> str:
        """Generate Makefile"""
        makefiles = {
            'python': '''.PHONY: install test lint clean run

install:
\tpip install -r requirements.txt

test:
\tpytest

lint:
\tflake8 .
\tblack --check .

format:
\tblack .

clean:
\tfind . -type f -name "*.pyc" -delete
\tfind . -type d -name "__pycache__" -delete
\trm -rf .pytest_cache

run:
\tpython main.py
''',
            'javascript': '''.PHONY: install test lint clean run build

install:
\tnpm install

test:
\tnpm test

lint:
\tnpm run lint

format:
\tnpm run format

clean:
\trm -rf node_modules
\trm -rf dist

build:
\tnpm run build

run:
\tnpm start
''',
            'go': '''.PHONY: build test lint clean run

build:
\tgo build -o bin/main

test:
\tgo test ./...

lint:
\tgolangci-lint run

clean:
\trm -rf bin/

run:
\tgo run main.go

install:
\tgo mod download
'''
        }
        
        return makefiles.get(language, makefiles['python'])
    
    async def generate_github_actions(self, language: str) -> str:
        """Generate GitHub Actions workflow"""
        workflows = {
            'python': '''name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest flake8
    
    - name: Lint
      run: flake8 .
    
    - name: Test
      run: pytest
''',
            'javascript': '''name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Use Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Lint
      run: npm run lint
    
    - name: Test
      run: npm test
'''
        }
        
        return workflows.get(language, workflows['python'])