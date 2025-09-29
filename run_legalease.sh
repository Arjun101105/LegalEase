#!/bin/bash
# LegalEase Quick Run Script
echo "🏛️  Starting LegalEase..."
venv\Scripts\activate
venv\Scripts\python src/cli_app.py "$@"
