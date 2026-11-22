@echo off
rem Show the WITS owner token as a QR code - double-click me.
title WITS owner token QR
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No .venv found next to this script. Create it first, see README.md.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\show_owner_token_qr.py %*
