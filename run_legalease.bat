@echo off
REM LegalEase Quick Start Script for Windows
echo 🏛️  Starting LegalEase...

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found. Please run setup first:
    echo    python setup.py
    pause
    exit /b 1
)

REM Activate virtual environment (Windows)
call venv\Scripts\activate.bat

REM Run LegalEase
python src\cli_app.py %*

pause
