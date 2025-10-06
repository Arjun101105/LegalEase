#!/bin/bash
# LegalEase Startup Script for Linux/macOS

echo "🏛️ Starting LegalEase..."

# Navigate to project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start backend server
echo "🚀 Starting backend server..."
python backend/app.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend server
echo "🌐 Starting frontend server..."
python3 -m http.server 3000 &
FRONTEND_PID=$!

echo ""
echo "✅ LegalEase is running!"
echo "📱 Web Interface: http://localhost:3000/web_interface.html"
echo "🔧 API Documentation: http://localhost:8000/docs"
echo "❤️  Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap 'echo ""; echo "🛑 Stopping LegalEase..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' INT
wait