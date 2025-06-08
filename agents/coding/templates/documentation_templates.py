"""
Documentation file templates (README, LICENSE, CONTRIBUTING)
"""

import logging
from typing import List
from datetime import datetime


class DocumentationTemplates:
    """Generates documentation file templates"""
    
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