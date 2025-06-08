"""
Configuration file templates (gitignore, Makefile, Dockerfile)
"""

import logging
from typing import Dict


class ConfigTemplates:
    """Generates configuration file templates"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
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