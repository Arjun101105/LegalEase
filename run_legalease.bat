@echo off
echo Starting LegalEase - Legal Text Simplification
echo ============================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Run LegalEase
cd /d "%~dp0"
python src/cli_app.py %*

if errorlevel 1 (
    echo.
    echo Error occurred. Press any key to close...
    pause >nul
)
