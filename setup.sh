#!/bin/bash
# LegalEase Setup Script
# Automated setup for LegalEase project

set -e  # Exit on any error

echo "🏛️ LegalEase Setup Script"
echo "========================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Python 3.8+ is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_status "Python $PYTHON_VERSION found"
            return 0
        else
            print_error "Python 3.8+ required. Found: $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        return 1
    fi
}

# Create virtual environment
create_venv() {
    print_info "Creating Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    print_status "Virtual environment created"
}

# Activate virtual environment and install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install main dependencies
    print_info "Installing core dependencies..."
    pip install -r requirements.txt
    
    # Install backend dependencies
    print_info "Installing backend dependencies..."
    pip install -r backend/requirements.txt
    
    print_status "Dependencies installed successfully"
}

# Download models and datasets
download_models() {
    print_info "Downloading pre-trained models and datasets..."
    print_warning "This may take 5-10 minutes depending on your internet connection..."
    
    source venv/bin/activate
    python scripts/download_datasets.py
    
    print_status "Models downloaded successfully"
}

# Create startup scripts
create_startup_scripts() {
    print_info "Creating startup scripts..."
    
    # Create start script for Linux/macOS
    cat > start_legalease.sh << 'EOF'
#!/bin/bash
# LegalEase Startup Script

echo "🏛️ Starting LegalEase..."

# Navigate to project directory
cd "$(dirname "$0")"

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
EOF

    chmod +x start_legalease.sh
    
    # Create Windows batch file
    cat > start_legalease.bat << 'EOF'
@echo off
echo 🏛️ Starting LegalEase...

:: Navigate to project directory
cd /d "%~dp0"

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
EOF

    print_status "Startup scripts created"
}

# Main setup function
main() {
    echo "Starting LegalEase setup..."
    echo ""
    
    # Check prerequisites
    if ! check_python; then
        exit 1
    fi
    
    # Setup steps
    create_venv
    install_dependencies
    download_models
    create_startup_scripts
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Start LegalEase: ./start_legalease.sh (Linux/macOS) or start_legalease.bat (Windows)"
    echo "2. Open your browser to: http://localhost:3000/web_interface.html"
    echo "3. Try simplifying some legal text!"
    echo ""
    echo "📚 Documentation: See README.md for detailed usage instructions"
    echo "🐛 Issues: Report at https://github.com/Arjun101105/LegalEase/issues"
    echo ""
}

# Run main function
main "$@"