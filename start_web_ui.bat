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

rem Docker sandbox (security.sandbox_mode: docker) — start Desktop if needed
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\ensure_docker_desktop.ps1"
if errorlevel 1 (
    echo [ERROR] Docker Desktop is required but could not be started.
    echo Install Docker Desktop or set security.sandbox_mode to "off" in config.yaml
    pause
    exit /b 1
)

".venv\Scripts\python.exe" run_web.py
pause
