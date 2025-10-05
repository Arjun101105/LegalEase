#!/usr/bin/env python3
"""
OCR (Optical Character Recognition) Module for LegalEase
Supports multiple OCR engines and document formats for extracting text from images and PDFs
"""

import os
import sys
from pathlib import Path
import logging
import tempfile
import shutil
from typing import List, Dict, Union, Optional, Tuple
import json
import time

# OCR and Image Processing
try:
    import pytesseract
    import easyocr
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    from pdf2image import convert_from_path, convert_from_bytes
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  OCR dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False
    # Define dummy classes to prevent NameError
    class Image:
        class Image:
            pass
    # Import basic types for type hints even if OCR dependencies fail
    try:
        from typing import Union
    except:
        pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalOCRProcessor:
    """OCR processor specifically designed for legal documents"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the OCR processor with configuration"""
        self.config = self._load_config(config_path)
        self.easyocr_reader = None
        self.temp_dir = tempfile.mkdtemp(prefix="legalease_ocr_")
        self.logger = logger
        
        # Setup Poppler on Windows if needed
        if os.name == 'nt':  # Windows
            self._setup_poppler_path_windows()
        
        # Initialize OCR engines
        self._initialize_ocr_engines()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load OCR configuration"""
        default_config = {
            "ocr_engines": {
                "tesseract": {
                    "enabled": True,
                    "config": "--oem 3 --psm 6",  # Best for legal documents
                    "language": "eng"
                },
                "easyocr": {
                    "enabled": True,
                    "languages": ["en"],
                    "gpu": False  # Set to True if you have GPU
                }
            },
            "image_preprocessing": {
                "resize_factor": 2.0,  # Upscale for better OCR
                "contrast_enhance": 1.5,
                "brightness_enhance": 1.2,
                "denoise": True,
                "deskew": True
            },
            "output_format": {
                "confidence_threshold": 60,  # Minimum confidence for text
                "preserve_layout": True,
                "include_metadata": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_ocr_engines(self):
        """Initialize available OCR engines"""
        if not DEPENDENCIES_AVAILABLE:
            logger.error("OCR dependencies not available. Please install: pip install pytesseract easyocr Pillow pdf2image opencv-python")
            return
        
        # Initialize EasyOCR if enabled
        if self.config["ocr_engines"]["easyocr"]["enabled"]:
            try:
                languages = self.config["ocr_engines"]["easyocr"]["languages"]
                gpu = self.config["ocr_engines"]["easyocr"]["gpu"]
                self.easyocr_reader = easyocr.Reader(languages, gpu=gpu)
                logger.info("✅ EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
        
        # Check Tesseract availability
        if self.config["ocr_engines"]["tesseract"]["enabled"]:
            try:
                # Test Tesseract installation
                version = pytesseract.get_tesseract_version()
                logger.info(f"✅ Tesseract {version} available")
            except Exception as e:
                logger.warning(f"Tesseract not available: {e}")
                logger.info("📋 To install Tesseract on Windows:")
                logger.info("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
                logger.info("   2. Install and add to PATH")
                logger.info("   3. Restart your terminal")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image (upscale for better OCR)
            if self.config["image_preprocessing"]["resize_factor"] != 1.0:
                factor = self.config["image_preprocessing"]["resize_factor"]
                new_size = (int(image.width * factor), int(image.height * factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Enhance contrast
            if self.config["image_preprocessing"]["contrast_enhance"] != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(self.config["image_preprocessing"]["contrast_enhance"])
            
            # Enhance brightness
            if self.config["image_preprocessing"]["brightness_enhance"] != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(self.config["image_preprocessing"]["brightness_enhance"])
            
            # Apply sharpening filter
            image = image.filter(ImageFilter.SHARPEN)
            
            # Convert to OpenCV format for advanced preprocessing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Denoise
            if self.config["image_preprocessing"]["denoise"]:
                cv_image = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)
            
            # Convert back to PIL
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            return image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def extract_text_tesseract(self, image: Image.Image) -> Dict:
        """Extract text using Tesseract OCR"""
        try:
            config = self.config["ocr_engines"]["tesseract"]["config"]
            lang = self.config["ocr_engines"]["tesseract"]["language"]
            
            # Get text with confidence scores
            data = pytesseract.image_to_data(
                image, 
                config=config, 
                lang=lang, 
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and calculate average confidence
            words = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > self.config["output_format"]["confidence_threshold"]:
                    word = data['text'][i].strip()
                    if word:
                        words.append(word)
                        confidences.append(int(data['conf'][i]))
            
            text = ' '.join(words)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": text,
                "confidence": avg_confidence,
                "engine": "tesseract",
                "word_count": len(words),
                "processing_time": 0
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {
                "text": "",
                "confidence": 0,
                "engine": "tesseract",
                "error": str(e)
            }
    
    def extract_text_easyocr(self, image: Image.Image) -> Dict:
        """Extract text using EasyOCR"""
        try:
            if not self.easyocr_reader:
                return {
                    "text": "",
                    "confidence": 0,
                    "engine": "easyocr",
                    "error": "EasyOCR not initialized"
                }
            
            start_time = time.time()
            
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Extract text
            results = self.easyocr_reader.readtext(img_array)
            
            # Process results
            words = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > (self.config["output_format"]["confidence_threshold"] / 100):
                    words.append(text)
                    confidences.append(confidence * 100)  # Convert to percentage
            
            full_text = ' '.join(words)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            processing_time = time.time() - start_time
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "engine": "easyocr",
                "word_count": len(words),
                "processing_time": processing_time,
                "bbox_data": results if self.config["output_format"]["include_metadata"] else None
            }
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {
                "text": "",
                "confidence": 0,
                "engine": "easyocr",
                "error": str(e)
            }
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """Extract text from a single image file"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = self.preprocess_image(image)
            
            results = {
                "source_file": image_path,
                "file_type": "image",
                "ocr_results": []
            }
            
            # Try Tesseract
            if self.config["ocr_engines"]["tesseract"]["enabled"]:
                tesseract_result = self.extract_text_tesseract(processed_image)
                results["ocr_results"].append(tesseract_result)
            
            # Try EasyOCR
            if self.config["ocr_engines"]["easyocr"]["enabled"]:
                easyocr_result = self.extract_text_easyocr(processed_image)
                results["ocr_results"].append(easyocr_result)
            
            # Choose best result
            best_result = self._choose_best_result(results["ocr_results"])
            results["best_text"] = best_result["text"] if best_result else ""
            results["best_confidence"] = best_result["confidence"] if best_result else 0
            results["best_engine"] = best_result["engine"] if best_result else "none"
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to extract text from image {image_path}: {e}")
            return {
                "source_file": image_path,
                "file_type": "image",
                "error": str(e),
                "best_text": "",
                "best_confidence": 0
            }
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Extract text from PDF file"""
        try:
            # Convert PDF pages to images
            pages = convert_from_path(pdf_path, dpi=300)  # High DPI for better OCR
            
            results = {
                "source_file": pdf_path,
                "file_type": "pdf",
                "pages": [],
                "combined_text": "",
                "total_confidence": 0
            }
            
            total_confidence = 0
            page_count = 0
            
            for i, page in enumerate(pages):
                logger.info(f"Processing PDF page {i+1}/{len(pages)}")
                
                # Preprocess page image
                processed_page = self.preprocess_image(page)
                
                # Extract text from page
                page_results = {
                    "page_number": i + 1,
                    "ocr_results": []
                }
                
                # Try both OCR engines
                if self.config["ocr_engines"]["tesseract"]["enabled"]:
                    tesseract_result = self.extract_text_tesseract(processed_page)
                    page_results["ocr_results"].append(tesseract_result)
                
                if self.config["ocr_engines"]["easyocr"]["enabled"]:
                    easyocr_result = self.extract_text_easyocr(processed_page)
                    page_results["ocr_results"].append(easyocr_result)
                
                # Choose best result for this page
                best_result = self._choose_best_result(page_results["ocr_results"])
                page_results["best_text"] = best_result["text"] if best_result else ""
                page_results["best_confidence"] = best_result["confidence"] if best_result else 0
                page_results["best_engine"] = best_result["engine"] if best_result else "none"
                
                results["pages"].append(page_results)
                
                # Add to combined text
                if page_results["best_text"]:
                    results["combined_text"] += f"\n--- Page {i+1} ---\n{page_results['best_text']}\n"
                    total_confidence += page_results["best_confidence"]
                    page_count += 1
            
            # Calculate average confidence
            results["total_confidence"] = total_confidence / page_count if page_count > 0 else 0
            results["page_count"] = len(pages)
            results["successful_pages"] = page_count
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            return {
                "source_file": pdf_path,
                "file_type": "pdf",
                "error": str(e),
                "combined_text": "",
                "total_confidence": 0
            }
    
    def _choose_best_result(self, ocr_results: List[Dict]) -> Optional[Dict]:
        """Choose the best OCR result based on confidence and text length"""
        if not ocr_results:
            return None
        
        # Filter out results with errors
        valid_results = [r for r in ocr_results if "error" not in r and r["text"]]
        
        if not valid_results:
            return ocr_results[0] if ocr_results else None
        
        # Score results based on confidence and text length
        scored_results = []
        for result in valid_results:
            confidence_score = result.get("confidence", 0)
            length_score = min(len(result["text"]) / 1000, 1.0) * 20  # Bonus for longer text
            total_score = confidence_score + length_score
            
            scored_results.append((total_score, result))
        
        # Return result with highest score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[0][1]
    
    def process_document(self, file_path: str) -> Dict:
        """Process any supported document type"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                "error": f"File not found: {file_path}",
                "best_text": "",
                "best_confidence": 0
            }
        
        # Determine file type and process accordingly
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(str(file_path))
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return self.extract_text_from_image(str(file_path))
        else:
            return {
                "error": f"Unsupported file type: {file_extension}",
                "supported_types": [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
                "best_text": "",
                "best_confidence": 0
            }
    
    def process_batch(self, input_dir: str, output_dir: Optional[str] = None) -> Dict:
        """Process multiple documents in a directory"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            return {"error": f"Input directory not found: {input_dir}"}
        
        supported_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        files_to_process = [
            f for f in input_path.rglob('*') 
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        if not files_to_process:
            return {
                "error": "No supported files found",
                "supported_types": list(supported_extensions)
            }
        
        results = {
            "input_directory": str(input_path),
            "total_files": len(files_to_process),
            "processed_files": [],
            "summary": {
                "successful": 0,
                "failed": 0,
                "total_confidence": 0
            }
        }
        
        # Process each file
        for i, file_path in enumerate(files_to_process, 1):
            logger.info(f"Processing file {i}/{len(files_to_process)}: {file_path.name}")
            
            file_result = self.process_document(str(file_path))
            file_result["file_index"] = i
            
            # Save individual result if output directory specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save extracted text
                text_file = output_path / f"{file_path.stem}_extracted.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(file_result.get("best_text", ""))
                
                # Save metadata
                json_file = output_path / f"{file_path.stem}_metadata.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(file_result, f, indent=2, default=str)
                
                file_result["output_files"] = {
                    "text": str(text_file),
                    "metadata": str(json_file)
                }
            
            results["processed_files"].append(file_result)
            
            # Update summary
            if "error" not in file_result:
                results["summary"]["successful"] += 1
                results["summary"]["total_confidence"] += file_result.get("best_confidence", 0)
            else:
                results["summary"]["failed"] += 1
        
        # Calculate average confidence
        if results["summary"]["successful"] > 0:
            results["summary"]["average_confidence"] = (
                results["summary"]["total_confidence"] / results["summary"]["successful"]
            )
        
        return results
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
    
    def __del__(self):
        """Destructor to cleanup resources"""
        self.cleanup()

    def _setup_poppler_path_windows(self):
        """Setup Poppler path for Windows if not in PATH"""
        import os
        import subprocess
        
        # Common Poppler installation paths on Windows
        possible_paths = [
            r"C:\Program Files\poppler\bin",
            r"C:\Program Files (x86)\poppler\bin", 
            r"C:\poppler\bin",
            r"C:\Tools\poppler\bin",
            os.path.expanduser(r"~\scoop\apps\poppler\current\bin"),
            os.path.expanduser(r"~\AppData\Local\Programs\poppler\bin")
        ]
        
        # Check if poppler is already in PATH
        try:
            subprocess.run(['pdftoppm', '-h'], capture_output=True, check=True)
            return True  # Already available
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        # Try to find and add Poppler to PATH
        for path in possible_paths:
            if os.path.exists(path):
                pdftoppm_exe = os.path.join(path, 'pdftoppm.exe')
                if os.path.exists(pdftoppm_exe):
                    # Add to current session PATH
                    current_path = os.environ.get('PATH', '')
                    if path not in current_path:
                        os.environ['PATH'] = path + os.pathsep + current_path
                        self.logger.info(f"Added Poppler path to session: {path}")
                    return True
        
        return False


def demo_ocr():
    """Demonstrate OCR functionality"""
    if not DEPENDENCIES_AVAILABLE:
        print("❌ OCR dependencies not available. Please install:")
        print("   pip install pytesseract easyocr Pillow pdf2image opencv-python")
        return
    
    print("🔍 LegalEase OCR Demo")
    print("=" * 50)
    
    # Initialize OCR processor
    ocr = LegalOCRProcessor()
    
    print("📋 Available OCR Engines:")
    if ocr.config["ocr_engines"]["tesseract"]["enabled"]:
        print("   ✅ Tesseract OCR")
    if ocr.config["ocr_engines"]["easyocr"]["enabled"]:
        print("   ✅ EasyOCR")
    
    print("\n📄 Supported File Types:")
    print("   • PDF documents (.pdf)")
    print("   • Images (.jpg, .jpeg, .png, .bmp, .tiff)")
    
    print("\n🎯 Features:")
    print("   • Automatic image preprocessing")
    print("   • Multiple OCR engine comparison")
    print("   • Confidence scoring")
    print("   • Batch processing")
    print("   • Legal document optimization")
    
    print("\n📖 Usage Examples:")
    print("   # Process single file")
    print("   result = ocr.process_document('legal_document.pdf')")
    print("   print(result['best_text'])")
    print("")
    print("   # Process batch of files")
    print("   results = ocr.process_batch('documents/', 'output/')")
    
    # Cleanup
    ocr.cleanup()


if __name__ == "__main__":
    demo_ocr()