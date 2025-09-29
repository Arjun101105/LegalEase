@echo off
chcp 65001 >nul
echo 🏛️  Starting LegalEase...
venv\Scripts\activate
venv\Scripts\python src/cli_app.py %*
