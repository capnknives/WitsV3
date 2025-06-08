"""Setup configuration for WitsV3."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="witsv3",
    version="3.1.0",
    author="WitsV3 Team",
    author_email="witsv3@example.com",
    description="An LLM orchestration system with CLI-first approach",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/witsv3",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "wits=witsv3.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "witsv3": ["config/*.yaml"],
    },
) 