@echo off
echo 🏛️ Starting LegalEase...

:: Navigate to project directory
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found. Please run setup.sh first.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Start backend server
echo 🚀 Starting backend server...
start /B python backend/app.py

:: Wait for backend to start
timeout /t 5 /nobreak >nul

:: Start frontend server
echo 🌐 Starting frontend server...
start /B python -m http.server 3000

echo.
echo ✅ LegalEase is running!
echo 📱 Web Interface: http://localhost:3000/web_interface.html
echo 🔧 API Documentation: http://localhost:8000/docs
echo ❤️  Health Check: http://localhost:8000/health
echo.
echo Press any key to stop all services
pause >nul

:: Stop servers
taskkill /f /im python.exe >nul 2>&1