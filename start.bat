@echo off
REM Easy start (double-click): runs the API on http://127.0.0.1:8000
cd /d "%~dp0"
uv run python -m app
pause
