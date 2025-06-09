#!/usr/bin/env python3
"""
WitsV3 Test Runner
This script runs all the non-interactive tests for WitsV3
"""

import os
# Fix Unicode encoding issues
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import codecs
# Set UTF-8 encoding for stdout/stderr
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Fix Unicode encoding issues
import sys
sys.stdout.reconfigure(encoding='utf-8')

import sys
import subprocess
import time
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and return if it was successful."""
    print(f"\n{'-' * 40}")
    print(f"Running: {description}")
    print(f"{'-' * 40}")

    start_time = time.time()
    result = subprocess.run(command, shell=True)
    elapsed = time.time() - start_time

    print(f"{'-' * 40}")
    if result.returncode == 0:
        print(f"✅ {description} completed successfully in {elapsed:.2f}s")
        return True
    else:
        print(f"❌ {description} failed with code {result.returncode} after {elapsed:.2f}s")
        return False

def run_tests(args):
    """Run all the tests."""
    tests = []

    # Configure environment
    test_env = os.environ.copy()
    test_env["CURSOR_TEST_MODE"] = "1"
    test_env["WITSV3_TEST_MODE"] = "true"

    # LLM Interface Test
    if args.llm or args.all:
        tests.append(("python llm_diagnostic_basic.py", "Basic LLM Interface Diagnostic"))

    # Run.py Test Mode
    if args.run or args.all:
        tests.append(("python run.py --test", "Run.py Test Mode"))

    # Non-interactive Test Script
    if args.basic or args.all:
        tests.append(("python test_witsv3.py --mode basic", "Basic Functionality Tests"))

    if args.tools or args.all:
        tests.append(("python test_witsv3.py --mode tools", "Tool System Tests"))

    if args.memory or args.all:
        tests.append(("python test_witsv3.py --mode memory", "Memory System Tests"))

    # Run all the tests
    results = []
    for command, description in tests:
        success = run_command(command, description)
        results.append((description, success))

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {description}")
        if success:
            passed += 1

    print("-" * 50)
    print(f"Passed: {passed}/{len(results)} tests")

    if passed == len(results):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed")
        return 1

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WitsV3 Non-Interactive Test Runner")

    # Test selection options
    test_group = parser.add_argument_group("Test Selection")
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument("--llm", action="store_true", help="Run LLM interface tests")
    test_group.add_argument("--run", action="store_true", help="Run run.py in test mode")
    test_group.add_argument("--basic", action="store_true", help="Run basic functionality tests")
    test_group.add_argument("--tools", action="store_true", help="Run tool system tests")
    test_group.add_argument("--memory", action="store_true", help="Run memory system tests")

    # Parse arguments
    args = parser.parse_args()

    # If no tests specified, run all
    if not any([args.all, args.llm, args.run, args.basic, args.tools, args.memory]):
        args.all = True

    try:
        return run_tests(args)
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nError running tests: {e}")
        return 1

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    sys.exit(main())
