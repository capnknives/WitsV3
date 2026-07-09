#!/usr/bin/env python3
"""
WitsV3 Debug Initialization Script

Validates the local dev environment before debugging or running tests.
Run from repo root: python scripts/debug_init.py
"""

import asyncio
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DebugInitializer:
    """Comprehensive debug initialization for WitsV3."""

    def __init__(self) -> None:
        self.project_root = PROJECT_ROOT
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.successes: List[str] = []

    def log_info(self, message: str) -> None:
        print(f"ℹ️  {message}")

    def log_success(self, message: str) -> None:
        print(f"✅ {message}")
        self.successes.append(message)

    def log_warning(self, message: str) -> None:
        print(f"⚠️  {message}")
        self.warnings.append(message)

    def log_error(self, message: str) -> None:
        print(f"❌ {message}")
        self.issues.append(message)

    def check_python_version(self) -> None:
        self.log_info("Checking Python version...")
        version = sys.version_info
        if version >= (3, 10):
            self.log_success(
                f"Python {version.major}.{version.minor}.{version.micro} is compatible"
            )
        else:
            self.log_error(
                f"Python {version.major}.{version.minor}.{version.micro} is too old. "
                "Requires 3.10+"
            )

    def check_environment(self) -> None:
        self.log_info("Checking environment...")
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            self.log_success(f"Running in venv: {venv}")
        elif "faiss" in conda_env.lower():
            self.log_success(f"Running in conda environment: {conda_env}")
        else:
            self.log_warning(f"Running in environment: {conda_env}")

    def check_dependencies(self) -> None:
        self.log_info("Checking critical dependencies...")
        critical_deps = [
            "pydantic",
            "httpx",
            "yaml",
            "aiofiles",
            "pytest",
            "pytest_cov",
            "pytest_asyncio",
            "apscheduler",
            "supabase",
        ]

        for dep in critical_deps:
            try:
                if dep == "yaml":
                    importlib.import_module("yaml")
                elif dep == "pytest_cov":
                    importlib.import_module("pytest_cov")
                elif dep == "pytest_asyncio":
                    importlib.import_module("pytest_asyncio")
                else:
                    importlib.import_module(dep)
                self.log_success(f"✓ {dep}")
            except ImportError:
                self.log_error(f"✗ {dep} - missing dependency")

    def check_project_structure(self) -> None:
        self.log_info("Checking project structure...")
        required_dirs = [
            "agents",
            "core",
            "tools",
            "tests",
            "tests/agents",
            "tests/core",
            "tests/tools",
            "logs",
        ]

        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.log_success(f"✓ {dir_path}/")
            else:
                self.log_error(f"✗ {dir_path}/ - missing directory")

        required_files = ["config.yaml", "run.py", "pyproject.toml"]
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.log_success(f"✓ {file_path}")
            else:
                self.log_error(f"✗ {file_path} - missing file")

    def check_imports(self) -> None:
        self.log_info("Checking WitsV3 module imports...")
        modules_to_test = [
            ("core.config", "WitsV3Config"),
            ("core.llm_interface", "BaseLLMInterface"),
            ("core.memory_manager", "MemoryManager"),
            ("agents.base_agent", "BaseAgent"),
            ("core.base_tool", "BaseTool"),
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
        self.log_info("Testing async functionality...")

        async def test_coroutine() -> str:
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
        self.log_info("Running sample test...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                test_count = len(
                    [line for line in result.stdout.split("\n") if ".py::" in line]
                )
                self.log_success(f"✓ Test discovery successful - found {test_count} tests")
            else:
                self.log_error(f"✗ Test discovery failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            self.log_error("✗ Test discovery timed out")
        except Exception as e:
            self.log_error(f"✗ Test discovery error: {e}")

    def check_ollama_connection(self) -> None:
        self.log_info("Checking Ollama connection...")
        try:
            import httpx

            async def check_ollama() -> tuple[bool, str]:
                async with httpx.AsyncClient() as client:
                    try:
                        response = await client.get(
                            "http://localhost:11434/api/tags", timeout=5.0
                        )
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
        return {
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "python_executable": sys.executable,
            "working_directory": str(Path.cwd()),
            "project_root": str(self.project_root),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "pythonpath": sys.path[:5],
            "issues_found": len(self.issues),
            "warnings_found": len(self.warnings),
            "successes": len(self.successes),
        }

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("🔧 WitsV3 DEBUG INITIALIZATION SUMMARY")
        print("=" * 60)
        print(f"\n✅ Successes: {len(self.successes)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Issues: {len(self.issues)}")

        if self.issues:
            print("\n🚨 CRITICAL ISSUES TO FIX:")
            for issue in self.issues:
                print(f"   • {issue}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")

        debug_info = self.generate_debug_info()
        print("\n📊 DEBUG INFO:")
        for key, value in debug_info.items():
            print(f"   • {key}: {value}")

        print("\n🎯 NEXT STEPS:")
        if self.issues:
            print("   1. Fix critical issues listed above")
            print("   2. Re-run this script to verify fixes")
            print("   3. Try debugging again")
        else:
            print("   1. Debug environment looks good!")
            print("   2. Try running: python -m pytest tests/ -v")
            print("   3. Use VS Code debugger with updated configurations")
        print("=" * 60)

    def run_full_check(self) -> None:
        print("🚀 Starting WitsV3 Debug Initialization...")
        print("=" * 60)
        self.check_python_version()
        self.check_environment()
        self.check_dependencies()
        self.check_project_structure()
        self.check_imports()
        self.test_async_functionality()
        self.run_sample_test()
        self.check_ollama_connection()
        self.print_summary()


def main() -> None:
    initializer = DebugInitializer()
    initializer.run_full_check()
    sys.exit(1 if initializer.issues else 0)


if __name__ == "__main__":
    main()
