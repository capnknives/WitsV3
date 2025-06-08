#!/usr/bin/env python3
"""
WitsV3 Installation Script
Automated setup for WitsV3 LLM orchestration system
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is 3.10+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def check_ollama():
    """Check if Ollama is available"""
    try:
        result = subprocess.run("ollama list", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is available")
            return True
        else:
            print("⚠️  Ollama not found - please install Ollama first")
            print("   Visit: https://ollama.ai/download")
            return False
    except FileNotFoundError:
        print("⚠️  Ollama not found - please install Ollama first")
        print("   Visit: https://ollama.ai/download")
        return False

def main():
    """Main installation process"""
    print("🚀 WitsV3 Installation Script")
    print("=" * 50)

    # Check prerequisites
    if not check_python_version():
        sys.exit(1)

    ollama_available = check_ollama()

    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)

    # Set up local data files
    if not run_command("python scripts/setup_local_data.py", "Setting up local data files"):
        print("❌ Failed to set up local data")
        sys.exit(1)

    # Set up authentication
    if not run_command("python setup_auth.py", "Setting up authentication system"):
        print("❌ Failed to set up authentication")
        sys.exit(1)

    # Install Ollama models if Ollama is available
    if ollama_available:
        print("\n🤖 Installing recommended Ollama models...")
        models = ["llama3", "codellama"]
        for model in models:
            run_command(f"ollama pull {model}", f"Installing {model} model")

    # Run basic tests
    print("\n🧪 Running installation verification tests...")
    if run_command("python -m pytest tests/core/test_config.py -v", "Testing configuration"):
        print("✅ Configuration tests passed")

    # Final status
    print("\n" + "=" * 50)
    print("🎉 WitsV3 Installation Complete!")
    print("\n📋 Next Steps:")
    print("1. Save your authentication token (displayed during setup)")
    print("2. Start Ollama: ollama serve")
    print("3. Run WitsV3: python run.py")
    print("4. Run tests: pytest tests/ -v")

    if not ollama_available:
        print("\n⚠️  Remember to install Ollama:")
        print("   Visit: https://ollama.ai/download")
        print("   Then run: ollama pull llama3")

    print("\n📚 Documentation: README.md")
    print("🔧 Configuration: config.yaml")

if __name__ == "__main__":
    main()
