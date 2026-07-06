@echo off
rem WitsV3 Web UI launcher - double-click me (or run from anywhere)
rem Uses the project's virtual environment regardless of where you start it.
title WitsV3 Web UI
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No .venv found next to this script.
    echo Create it first:  python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" run_web.py
pause
