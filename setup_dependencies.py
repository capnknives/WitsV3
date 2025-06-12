#!/usr/bin/env python
"""
Setup helper for WitsV3 dependencies.
This script ensures proper installation of dependencies with the correct versions.
"""
import subprocess
import sys
import os

def main():
    print("Setting up WitsV3 dependencies...")
    
    # Install core dependencies first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install development dependencies if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--dev":
        print("Installing development dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    
    # Fix numpy and pydantic versions
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=1.25.0,<3.0", "--force-reinstall"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic>=2.0.0", "--force-reinstall"])
    
    print("Dependencies installed successfully!")
    
    # Verify imports work
    try:
        import numpy
        import pydantic
        print(f"Numpy version: {numpy.__version__}")
        print(f"Pydantic version: {pydantic.__version__}")
        print("Verification successful.")
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
