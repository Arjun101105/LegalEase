#!/usr/bin/env python3
"""
LegalEase FastAPI Backend
RESTful API for legal text simplification with OCR support
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid
import logging
import mimetypes

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Import LegalEase modules
from cli_app import LegalTextSimplifier
from ocr_processor import LegalOCRProcessor
from llm_integration import LLMSimplifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LegalEase API",
    description="AI-powered legal text simplification with OCR support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (loaded once at startup)
text_simplifier = None
ocr_processor = None
temp_dir = None

# Pydantic models for request/response
class TextSimplificationRequest(BaseModel):
    text: str
    use_llm: bool = True
    max_length: Optional[int] = None

class TextSimplificationResponse(BaseModel):
    success: bool
    original_text: str
    simplified_text: str
    enhancement_used: bool
    processing_time: float
    confidence_score: Optional[float] = None
    method_used: str

class OCRRequest(BaseModel):
    simplify_text: bool = True
    use_llm: bool = True

class OCRResponse(BaseModel):
    success: bool
    extracted_text: str
    simplified_text: Optional[str] = None
    confidence_score: float
    processing_time: float
    character_count: int
    enhancement_used: Optional[bool] = None
    method_used: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    system_info: Dict[str, Any]

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global text_simplifier, ocr_processor, temp_dir
    
    logger.info("🚀 Starting LegalEase API...")
    
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="legalease_api_")
        logger.info(f"📁 Temp directory: {temp_dir}")
        
        # Initialize text simplifier
        logger.info("🤖 Loading text simplification models...")
        text_simplifier = LegalTextSimplifier()
        text_simplifier.load_models()
        logger.info("✅ Text simplification models loaded")
        
        # Initialize OCR processor
        logger.info("🔍 Loading OCR processor...")
        ocr_processor = LegalOCRProcessor()
        logger.info("✅ OCR processor loaded")
        
        logger.info("🎉 LegalEase API started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start API: {str(e)}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global temp_dir
    
    logger.info("🛑 Shutting down LegalEase API...")
    
    if temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
        logger.info("🗑️ Temp directory cleaned up")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        return HealthResponse(
            status="healthy",
            models_loaded={
                "text_simplifier": text_simplifier is not None,
                "ocr_processor": ocr_processor is not None,
            },
            system_info={
                "temp_dir": temp_dir,
                "python_version": sys.version,
                "api_version": "1.0.0"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Text simplification endpoint
@app.post("/api/simplify-text", response_model=TextSimplificationResponse)
async def simplify_text(request: TextSimplificationRequest):
    """Simplify legal text"""
    import time
    
    start_time = time.time()
    
    try:
        if not text_simplifier:
            raise HTTPException(status_code=500, detail="Text simplifier not initialized")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 10000:
            raise HTTPException(status_code=400, detail="Text too long (max 10000 characters)")
        
        # Simplify text
        logger.info(f"📝 Simplifying text: {request.text[:100]}...")
        
        if request.use_llm:
            result = text_simplifier.simplify_text_enhanced(request.text)
        else:
            result = text_simplifier.simplify_text(request.text)
        
        processing_time = time.time() - start_time
        
        # Handle different result formats
        if isinstance(result, dict):
            simplified_text = result.get('simplified_text', request.text)
            enhancement_used = result.get('enhancement_used', False)
        else:
            simplified_text = result
            enhancement_used = False
        
        return TextSimplificationResponse(
            success=True,
            original_text=request.text,
            simplified_text=simplified_text,
            enhancement_used=enhancement_used,
            processing_time=processing_time,
            method_used="llm_enhanced" if request.use_llm else "rule_based"
        )
        
    except Exception as e:
        logger.error(f"❌ Error in text simplification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simplification failed: {str(e)}")

# OCR endpoint
@app.post("/api/ocr", response_model=OCRResponse)
async def process_ocr(
    file: UploadFile = File(...),
    simplify_text: bool = Form(True),
    use_llm: bool = Form(True)
):
    """Process document with OCR and optional text simplification"""
    import time
    
    start_time = time.time()
    
    try:
        if not ocr_processor:
            raise HTTPException(status_code=500, detail="OCR processor not initialized")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (max 10MB)
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Check file type
        allowed_types = {
            'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff',
            'application/pdf'
        }
        
        content_type = file.content_type
        if content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {content_type}. Supported: PDF, JPG, PNG, BMP, TIFF"
            )
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        temp_file_path = Path(temp_dir) / f"{file_id}{file_extension}"
        
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(content)
        
        logger.info(f"📄 Processing file: {file.filename} ({file_size} bytes)")
        
        # Process with OCR
        ocr_result = ocr_processor.extract_text_from_file(str(temp_file_path))
        
        if not ocr_result['success']:
            raise HTTPException(status_code=500, detail=f"OCR failed: {ocr_result.get('error', 'Unknown error')}")
        
        extracted_text = ocr_result['text']
        confidence_score = ocr_result.get('confidence', 0.0)
        
        simplified_text = None
        enhancement_used = None
        
        # Simplify text if requested
        if simplify_text and extracted_text.strip():
            logger.info("🔄 Simplifying extracted text...")
            
            if use_llm:
                simplification_result = text_simplifier.simplify_text_enhanced(extracted_text)
            else:
                simplification_result = text_simplifier.simplify_text(extracted_text)
            
            if isinstance(simplification_result, dict):
                simplified_text = simplification_result.get('simplified_text', extracted_text)
                enhancement_used = simplification_result.get('enhancement_used', False)
            else:
                simplified_text = simplification_result
                enhancement_used = False
        
        processing_time = time.time() - start_time
        
        # Cleanup temp file
        temp_file_path.unlink(missing_ok=True)
        
        return OCRResponse(
            success=True,
            extracted_text=extracted_text,
            simplified_text=simplified_text,
            confidence_score=confidence_score,
            processing_time=processing_time,
            character_count=len(extracted_text),
            enhancement_used=enhancement_used,
            method_used="ocr_with_llm" if (simplify_text and use_llm) else "ocr_only"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in OCR processing: {str(e)}")
        # Cleanup temp file on error
        if 'temp_file_path' in locals():
            temp_file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

# Batch OCR endpoint
@app.post("/api/ocr/batch")
async def process_batch_ocr(
    files: List[UploadFile] = File(...),
    simplify_text: bool = Form(True),
    use_llm: bool = Form(True)
):
    """Process multiple documents with OCR"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    
    for file in files:
        try:
            # Process each file individually
            result = await process_ocr(file, simplify_text, use_llm)
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_files": len(files),
        "results": results
    }

# Get supported file types
@app.get("/api/supported-types")
async def get_supported_types():
    """Get supported file types for OCR"""
    return {
        "success": True,
        "supported_types": {
            "images": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
            "documents": [".pdf"],
            "max_file_size": "10MB",
            "max_batch_files": 10
        }
    }

# Model information endpoint
@app.get("/api/models/info")
async def get_model_info():
    """Get information about loaded models"""
    try:
        models_info = {
            "text_simplification": {
                "inlegal_bert": text_simplifier.inlegal_model is not None if text_simplifier else False,
                "flan_t5": text_simplifier.t5_model is not None if text_simplifier else False,
                "llm_enhancement": text_simplifier.use_llm_enhancement if text_simplifier else False
            },
            "ocr": {
                "tesseract": True,  # Assuming it's available if OCR processor loaded
                "easyocr": True,
                "confidence_threshold": 0.5
            }
        }
        
        return {
            "success": True,
            "models": models_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Examples endpoint
@app.get("/api/examples")
async def get_examples():
    """Get example legal texts for testing"""
    examples = [
        {
            "id": 1,
            "title": "Writ Petition",
            "text": "The plaintiff filed a writ petition under Article 32 of the Constitution seeking mandamus against the respondent for non-compliance with statutory obligations."
        },
        {
            "id": 2,
            "title": "Contract Clause",
            "text": "The party of the first part hereby covenants and agrees to indemnify and hold harmless the party of the second part from any and all claims, damages, losses, costs and expenses."
        },
        {
            "id": 3,
            "title": "Court Judgment",
            "text": "The appellant contends that the lower court erred in not considering the precedent established in the landmark judgment."
        },
        {
            "id": 4,
            "title": "Legal Notice",
            "text": "Take notice that my client is constrained to initiate appropriate legal proceedings against you for recovery of the aforesaid amount together with interest and costs."
        }
    ]
    
    return {
        "success": True,
        "examples": examples
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "details": str(exc)
        }
    )

# Main function to run the server
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )