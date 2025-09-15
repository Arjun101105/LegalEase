#!/bin/bash
# LegalEase Quick Start Script
echo "🏛️  Starting LegalEase..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python setup.py"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run LegalEase
python src/cli_app.py "$@"
