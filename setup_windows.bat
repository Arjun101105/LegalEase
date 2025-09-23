@echo off
REM LegalEase Setup Script for Windows
echo 🏛️  LegalEase Setup Script
echo ==================================================
echo 📖 Setting up Legal Text Simplification for Indian Citizens
echo 🔒 Offline ^& Privacy-focused ^| No data storage
echo ==================================================
echo.

echo 🔍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

echo 🏗️  Creating virtual environment...
if exist "venv" (
    echo ⚠️  Virtual environment already exists, removing old one...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment created
echo.

echo 📦 Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed
echo.

echo 🔄 Running initial setup...
python setup.py
if errorlevel 1 (
    echo ❌ Setup failed
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo 🚀 To start LegalEase:
echo    run_legalease.bat
echo.
echo 🧪 To test with examples:
echo    run_legalease.bat --examples
echo.
pause
