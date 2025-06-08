#!/usr/bin/env python3
"""
WitsV3 Debug Initialization Script

This script helps initialize and validate the debug environment for WitsV3.
Run this before attempting to debug or test the application.
"""

import sys
import os
import subprocess
import importlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class DebugInitializer:
    """Comprehensive debug initialization for WitsV3."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.successes: List[str] = []

    def log_info(self, message: str) -> None:
        """Log an info message."""
        print(f"ℹ️  {message}")

    def log_success(self, message: str) -> None:
        """Log a success message."""
        print(f"✅ {message}")
        self.successes.append(message)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        print(f"⚠️  {message}")
        self.warnings.append(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        print(f"❌ {message}")
        self.issues.append(message)

    def check_python_version(self) -> None:
        """Check Python version compatibility."""
        self.log_info("Checking Python version...")
        version = sys.version_info
        if version >= (3, 10):
            self.log_success(f"Python {version.major}.{version.minor}.{version.micro} is compatible")
        else:
            self.log_error(f"Python {version.major}.{version.minor}.{version.micro} is too old. Requires 3.10+")

    def check_environment(self) -> None:
        """Check conda environment."""
        self.log_info("Checking conda environment...")

        # Check if we're in the right environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        if 'faiss' in conda_env.lower():
            self.log_success(f"Running in conda environment: {conda_env}")
        else:
            self.log_warning(f"Running in environment: {conda_env} (expected faiss-gpu-env2)")

    def check_dependencies(self) -> None:
        """Check critical dependencies."""
        self.log_info("Checking critical dependencies...")

        critical_deps = [
            'pydantic',
            'httpx',
            'yaml',
            'aiofiles',
            'pytest',
            'pytest_cov',
            'pytest_asyncio',
            'apscheduler',
            'supabase'
        ]

        for dep in critical_deps:
            try:
                if dep == 'yaml':
                    importlib.import_module('yaml')
                elif dep == 'pytest_cov':
                    importlib.import_module('pytest_cov')
                elif dep == 'pytest_asyncio':
                    importlib.import_module('pytest_asyncio')
                else:
                    importlib.import_module(dep)
                self.log_success(f"✓ {dep}")
            except ImportError:
                self.log_error(f"✗ {dep} - missing dependency")

    def check_project_structure(self) -> None:
        """Check project structure."""
        self.log_info("Checking project structure...")

        required_dirs = [
            'agents',
            'core',
            'tools',
            'tests',
            'tests/agents',
            'tests/core',
            'tests/tools',
            'logs'
        ]

        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.log_success(f"✓ {dir_path}/")
            else:
                self.log_error(f"✗ {dir_path}/ - missing directory")

        # Check key files
        required_files = [
            'config.yaml',
            'run.py',
            'pytest.ini',
            '.vscode/launch.json'
        ]

        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.log_success(f"✓ {file_path}")
            else:
                self.log_error(f"✗ {file_path} - missing file")

    def check_imports(self) -> None:
        """Check critical imports."""
        self.log_info("Checking WitsV3 module imports...")

        modules_to_test = [
            ('core.config', 'WitsV3Config'),
            ('core.llm_interface', 'BaseLLMInterface'),
            ('core.memory_manager', 'MemoryManager'),
            ('agents.base_agent', 'BaseAgent'),
            ('core.base_tool', 'BaseTool'),
        ]

        for module_name, class_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                getattr(module, class_name)
                self.log_success(f"✓ {module_name}.{class_name}")
            except ImportError as e:
                self.log_error(f"✗ {module_name}.{class_name} - import error: {e}")
            except AttributeError as e:
                self.log_error(f"✗ {module_name}.{class_name} - attribute error: {e}")

    def test_async_functionality(self) -> None:
        """Test async functionality."""
        self.log_info("Testing async functionality...")

        async def test_coroutine():
            await asyncio.sleep(0.01)
            return "async_test_passed"

        try:
            result = asyncio.run(test_coroutine())
            if result == "async_test_passed":
                self.log_success("✓ Async functionality working")
            else:
                self.log_error("✗ Async test failed")
        except Exception as e:
            self.log_error(f"✗ Async test error: {e}")

    def run_sample_test(self) -> None:
        """Run a sample test to verify pytest setup."""
        self.log_info("Running sample test...")

        try:
            # Run a simple test discovery
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                '--collect-only', '-q'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                test_count = len([line for line in result.stdout.split('\n') if '.py::' in line])
                self.log_success(f"✓ Test discovery successful - found {test_count} tests")
            else:
                self.log_error(f"✗ Test discovery failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.log_error("✗ Test discovery timed out")
        except Exception as e:
            self.log_error(f"✗ Test discovery error: {e}")

    def check_ollama_connection(self) -> None:
        """Check if Ollama is running and accessible."""
        self.log_info("Checking Ollama connection...")

        try:
            import httpx

            async def check_ollama():
                async with httpx.AsyncClient() as client:
                    try:
                        response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                        if response.status_code == 200:
                            return True, "Connection successful"
                    except Exception as e:
                        return False, str(e)
                return False, "Unknown error"

            is_running, message = asyncio.run(check_ollama())

            if is_running:
                self.log_success("✓ Ollama is running and accessible")
            else:
                self.log_warning(f"⚠ Ollama not accessible: {message}")

        except Exception as e:
            self.log_warning(f"⚠ Could not check Ollama: {e}")

    def generate_debug_info(self) -> Dict[str, Any]:
        """Generate comprehensive debug information."""
        return {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_executable": sys.executable,
            "working_directory": str(Path.cwd()),
            "project_root": str(self.project_root),
            "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'unknown'),
            "pythonpath": sys.path[:5],  # First 5 entries
            "issues_found": len(self.issues),
            "warnings_found": len(self.warnings),
            "successes": len(self.successes),
        }

    def print_summary(self) -> None:
        """Print debug summary."""
        print("\n" + "="*60)
        print("🔧 WitsV3 DEBUG INITIALIZATION SUMMARY")
        print("="*60)

        print(f"\n✅ Successes: {len(self.successes)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Issues: {len(self.issues)}")

        if self.issues:
            print(f"\n🚨 CRITICAL ISSUES TO FIX:")
            for issue in self.issues:
                print(f"   • {issue}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")

        debug_info = self.generate_debug_info()
        print(f"\n📊 DEBUG INFO:")
        for key, value in debug_info.items():
            print(f"   • {key}: {value}")

        print(f"\n🎯 NEXT STEPS:")
        if self.issues:
            print("   1. Fix critical issues listed above")
            print("   2. Re-run this script to verify fixes")
            print("   3. Try debugging again")
        else:
            print("   1. Debug environment looks good!")
            print("   2. Try running: python -m pytest tests/ -v")
            print("   3. Use VS Code debugger with updated configurations")

        print("="*60)

    def run_full_check(self) -> None:
        """Run all debug checks."""
        print("🚀 Starting WitsV3 Debug Initialization...")
        print("="*60)

        self.check_python_version()
        self.check_environment()
        self.check_dependencies()
        self.check_project_structure()
        self.check_imports()
        self.test_async_functionality()
        self.run_sample_test()
        self.check_ollama_connection()

        self.print_summary()

def main():
    """Main entry point."""
    initializer = DebugInitializer()
    initializer.run_full_check()

    # Return appropriate exit code
    if initializer.issues:
        sys.exit(1)  # Exit with error if issues found
    else:
        sys.exit(0)  # Success

if __name__ == "__main__":
    main()
