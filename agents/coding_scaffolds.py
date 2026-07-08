# agents/coding_scaffolds.py
"""Scaffold and file-generation helpers for the advanced coding agent."""

import json


class CodingScaffoldMixin:
    """Mixin providing project scaffold and initial file generation."""

    async def _generate_python_web_app(self) -> str:
        return """from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
"""

    async def _generate_python_api(self) -> str:
        return """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="API Service", version="1.0.0")

class Item(BaseModel):
    name: str
    description: str = None

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item, "status": "created"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id, "data": "sample data"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

    async def _generate_python_cli(self) -> str:
        return """import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="CLI Tool")
    parser.add_argument('--version', action='version', version='1.0.0')
    parser.add_argument('command', help='Command to execute')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        print(f"Executing command: {args.command}")

    # Add your command logic here
    print(f"Command '{args.command}' executed successfully")

if __name__ == '__main__':
    main()
"""

    async def _generate_python_main(self) -> str:
        return '''"""
Main application entry point
"""

def main():
    """Main function"""
    print("Application started")
    # Add your application logic here

if __name__ == "__main__":
    main()
'''

    async def _generate_python_requirements(self, requirements: list[str]) -> str:
        base_reqs = []

        if any("web" in req.lower() for req in requirements):
            base_reqs.extend(["flask>=2.0.0", "requests>=2.25.0"])
        if any("api" in req.lower() for req in requirements):
            base_reqs.extend(["fastapi>=0.68.0", "uvicorn>=0.15.0"])
        if any("test" in req.lower() for req in requirements):
            base_reqs.extend(["pytest>=6.0.0", "pytest-cov>=2.12.0"])
        if any("data" in req.lower() for req in requirements):
            base_reqs.extend(["pandas>=1.3.0", "numpy>=1.21.0"])

        if not base_reqs:
            base_reqs = ["requests>=2.25.0"]

        return "\n".join(sorted(set(base_reqs)))

    async def _generate_python_setup(self) -> str:
        return """from setuptools import setup, find_packages

setup(
    name="project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add dependencies here
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
"""

    async def _generate_js_package_json(self, project_type: str) -> str:
        dependencies = {}

        if project_type == "web_app":
            dependencies.update({"express": "^4.18.0", "react": "^18.0.0", "react-dom": "^18.0.0"})
        elif project_type == "api":
            dependencies.update({"express": "^4.18.0", "cors": "^2.8.5"})

        return json.dumps(
            {
                "name": "project",
                "version": "1.0.0",
                "description": "A JavaScript project",
                "main": "index.js",
                "scripts": {"start": "node index.js", "dev": "nodemon index.js", "test": "jest"},
                "dependencies": dependencies,
                "devDependencies": {"nodemon": "^2.0.0", "jest": "^28.0.0"},
                "author": "Your Name",
                "license": "MIT",
            },
            indent=2,
        )

    async def _generate_js_web_app(self) -> str:
        return """const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/api/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});

module.exports = app;
"""

    async def _generate_js_api(self) -> str:
        return """const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json());

// Sample data
let items = [];
let nextId = 1;

// Routes
app.get('/', (req, res) => {
    res.json({ message: 'API is running', version: '1.0.0' });
});

app.get('/api/items', (req, res) => {
    res.json(items);
});

app.post('/api/items', (req, res) => {
    const { name, description } = req.body;

    if (!name) {
        return res.status(400).json({ error: 'Name is required' });
    }

    const item = {
        id: nextId++,
        name,
        description: description || '',
        createdAt: new Date().toISOString()
    };

    items.push(item);
    res.status(201).json(item);
});

app.get('/api/items/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const item = items.find(item => item.id === id);

    if (!item) {
        return res.status(404).json({ error: 'Item not found' });
    }

    res.json(item);
});

app.put('/api/items/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const itemIndex = items.findIndex(item => item.id === id);

    if (itemIndex === -1) {
        return res.status(404).json({ error: 'Item not found' });
    }

    const { name, description } = req.body;
    items[itemIndex] = {
        ...items[itemIndex],
        name: name || items[itemIndex].name,
        description: description !== undefined ? description : items[itemIndex].description,
        updatedAt: new Date().toISOString()
    };

    res.json(items[itemIndex]);
});

app.delete('/api/items/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const itemIndex = items.findIndex(item => item.id === id);

    if (itemIndex === -1) {
        return res.status(404).json({ error: 'Item not found' });
    }

    items.splice(itemIndex, 1);
    res.status(204).send();
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!' });
});

// Start server
app.listen(PORT, () => {
    console.log(`API server running on http://localhost:${PORT}`);
});

module.exports = app;
"""

    async def _generate_js_main(self) -> str:
        return """#!/usr/bin/env node

/**
 * Main application entry point
 */

function main() {
    console.log('Application started');

    // Parse command line arguments
    const args = process.argv.slice(2);

    if (args.includes('--help') || args.includes('-h')) {
        showHelp();
        return;
    }

    if (args.includes('--version') || args.includes('-v')) {
        console.log('Version 1.0.0');
        return;
    }

    // Add your application logic here
    console.log('Arguments:', args);
    console.log('Application running successfully');
}

function showHelp() {
    console.log(`
Usage: node index.js [options]

Options:
  -h, --help     Show this help message
  -v, --version  Show version information

Examples:
  node index.js
  node index.js --help
    `);
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// Run main function
if (require.main === module) {
    main();
}

module.exports = { main };
"""

    async def _generate_js_tests(self) -> str:
        return """const request = require('supertest');
const app = require('../index');

describe('Application Tests', () => {
    test('should start application', () => {
        expect(app).toBeDefined();
    });

    test('should have main function', () => {
        expect(typeof app.main).toBe('function');
    });
});

describe('API Tests', () => {
    test('GET / should return success message', async () => {
        const response = await request(app)
            .get('/')
            .expect(200);

        expect(response.body).toHaveProperty('message');
    });

    test('GET /api/health should return health status', async () => {
        const response = await request(app)
            .get('/api/health')
            .expect(200);

        expect(response.body).toHaveProperty('status', 'healthy');
    });
});

// Add more tests here
"""

    async def _generate_readme(
        self, project_type: str, language: str, requirements: list[str]
    ) -> str:
        return f"""# {project_type.title()} Project

## Description
{language.title()} {project_type} implementing: {', '.join(requirements)}

## Installation
```bash
# Add installation instructions here
```

## Usage
```bash
# Add usage instructions here
```

## Development
```bash
# Add development setup instructions here
```

## Testing
```bash
# Add testing instructions here
```

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
"""

    async def _generate_gitignore(self, language: str) -> str:
        common = """# General
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
"""

        if language == "python":
            return common + """
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
.env
venv/
ENV/
"""
        elif language == "javascript":
            return common + """
# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.eslintcache
"""
        else:
            return common

    async def _generate_license(self) -> str:
        return """MIT License

Copyright (c) 2025 Your Name

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
"""

    async def _generate_python_tests(self) -> str:
        return '''import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_example():
    """Example test function"""
    assert True

def test_main_import():
    """Test that main module can be imported"""
    try:
        import main
        assert True
    except ImportError:
        pytest.skip("Main module not found")

# Add more tests here
'''
