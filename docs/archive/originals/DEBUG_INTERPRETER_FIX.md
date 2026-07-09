# Debug System: Python Interpreter Fix

## Problem Resolved

When using Cursor's "Run and Debug" feature, you were encountering this error:

```
ERROR: usage: __main__.py [options] [file_or_dir] [file_or_dir] [...]
__main__.py: error: unrecognized arguments: --cov=tools --cov=core --cov=agents --cov-report=term-missing --cov-report=html
```

## Root Cause

The issue was that Cursor/VS Code was using the **wrong Python interpreter**:

- **Wrong**: `C:\Users\capta\AppData\Local\Programs\Python\Python313\python.exe` (Python 3.13 system installation)
- **Correct**: `C:\Users\capta\miniconda3\envs\faiss_gpu_env2\python.exe` (Python 3.10.17 conda environment)

The Python 3.13 system installation didn't have the required packages (`pytest-cov`, `pytest-asyncio`, etc.) that we installed in the conda environment.

## Fixes Applied

### 1. VS Code Settings (.vscode/settings.json)

```json
{
  "python.defaultInterpreterPath": "C:\\Users\\capta\\miniconda3\\envs\\faiss_gpu_env2\\python.exe",
  "python.pythonPath": "C:\\Users\\capta\\miniconda3\\envs\\faiss_gpu_env2\\python.exe"
  // ... other settings
}
```

### 2. Debug Configurations (.vscode/launch.json)

Added explicit Python interpreter paths to **all 11 debug configurations**:

```json
{
  "name": "WitsV3: All Tests",
  "type": "python",
  "request": "launch",
  "module": "pytest",
  "python": "C:\\Users\\capta\\miniconda3\\envs\\faiss_gpu_env2\\python.exe"
  // ... rest of configuration
}
```

### 3. Cleaned Up Settings

- Removed duplicate entries in VS Code settings
- Fixed pytest arguments to match our testing setup
- Ensured coverage reporting targets correct modules (core, agents, tools)

## Verification

✅ **Test Command Works**:

```powershell
C:\Users\capta\miniconda3\envs\faiss_gpu_env2\python.exe -m pytest tests/ -v --tb=short --asyncio-mode=auto --cov=core --cov=agents --cov=tools --cov-report=term-missing --cov-report=html
```

✅ **All Required Packages Available**:

```powershell
C:\Users\capta\miniconda3\envs\faiss_gpu_env2\python.exe -c "import pytest; import pytest_cov; import pytest_asyncio; print('All packages available')"
```

## How to Use Debug System Now

### Available Debug Configurations:

1. **WitsV3: Main Entry** - Debug the main application
2. **WitsV3: Main with Args** - Debug with command line arguments
3. **WitsV3: Background Agent** - Debug the background agent
4. **WitsV3: Current File** - Debug currently open file
5. **WitsV3: All Tests** - Run all tests with coverage
6. **WitsV3: Debug Tests** - Run tests with debugger enabled
7. **WitsV3: Test Current File** - Test currently open file
8. **WitsV3: Fast Tests (Unit Only)** - Quick unit tests
9. **WitsV3: Core Tests Only** - Test only core module
10. **WitsV3: Tools Tests Only** - Test only tools module
11. **WitsV3: Agent Tests Only** - Test only agents module

### To Use:

1. Open VS Code/Cursor
2. Go to Run and Debug panel (Ctrl+Shift+D)
3. Select any configuration from the dropdown
4. Click the green play button or press F5

All configurations now use the correct Python interpreter and should work without errors.

## Environment Verification

The system now correctly uses:

- **Python Version**: 3.10.17 (from faiss_gpu_env2)
- **pytest**: 8.3.5
- **pytest-cov**: 6.1.1
- **pytest-asyncio**: 0.26.0
- **All WitsV3 dependencies**: Installed and available

Your debug system is now fully operational! 🎉
