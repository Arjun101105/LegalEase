#!/usr/bin/env python3
"""
Configuration settings for LegalEase Backend API
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

# API Configuration
API_HOST = os.getenv("LEGALEASE_HOST", "0.0.0.0")
API_PORT = int(os.getenv("LEGALEASE_PORT", 8000))
API_RELOAD = os.getenv("LEGALEASE_RELOAD", "true").lower() == "true"

# File upload limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_BATCH_FILES = 10
ALLOWED_EXTENSIONS = {
    'images': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    'documents': ['.pdf']
}

# Text processing limits
MAX_TEXT_LENGTH = 10000  # characters
DEFAULT_TIMEOUT = 60  # seconds

# Model configuration
MODEL_CONFIG = {
    "use_llm_enhancement": True,
    "ocr_confidence_threshold": 0.5,
    "max_tokens": 1000,
    "temperature": 0.3
}

# CORS settings
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001", 
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001"
]

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30