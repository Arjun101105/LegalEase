#!/usr/bin/env python3
"""
LegalEase - Command Line Interface for Legal Text Simplification
A simple CLI tool to simplify Indian legal texts into layman's English
"""

import os
import sys
from pathlib import Path
import logging
import warnings
import time
import json
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import OCR processor
try:
    from ocr_processor import LegalOCRProcessor
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

import torch
from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForSeq2SeqLM,
)

from llm_integration import LLMSimplifier
from local_llm_integration import LocalLegalSimplifier

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalTextSimplifier:
    """Main class for legal text simplification"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent.parent
        self.models_dir = self.project_dir / "data" / "models"
        
        # Model components
        self.inlegal_model = None
        self.inlegal_tokenizer = None
        self.t5_model = None
        self.t5_tokenizer = None
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize LLM enhancer
        self.llm_simplifier = LLMSimplifier()
        self.use_llm_enhancement = True  # Enable by default
        
        # Initialize local LLM enhancer (100% free & offline)
        self.local_llm_simplifier = LocalLegalSimplifier()
        self.use_local_llm = True  # Enable by default
    
    def load_config(self):
        """Load simplification configuration"""
        config_path = self.models_dir / "simplification_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            "max_length": 512,
            "min_length": 50,
            "num_beams": 4,
            "temperature": 0.8,
            "do_sample": True,
            "early_stopping": True,
            "no_repeat_ngram_size": 3
        }
    
    def _check_models_exist(self):
        """Check if essential models/data exist"""
        essential_paths = [
            self.project_dir / "data",
            self.project_dir / "src" / "cli_app.py",
        ]
        
        for path in essential_paths:
            if not path.exists():
                return False
        return True
    
    def load_models(self):
        """Load all required models"""
        print("🤖 Loading LegalEase Models...")
        print("=" * 50)
        
        # Check if models exist
        if not self._check_models_exist():
            print("❌ Models not found! Please run setup first:")
            print("   python setup.py")
            print("   OR")
            print("   python scripts/download_datasets.py")
            return False
        
        # Load InLegalBERT for text understanding
        print("📚 Loading InLegalBERT...")
        inlegal_path = self.models_dir / "InLegalBERT"
        if (inlegal_path.exists()):
            try:
                self.inlegal_tokenizer = AutoTokenizer.from_pretrained(str(inlegal_path))
                self.inlegal_model = AutoModel.from_pretrained(str(inlegal_path))
                print("✅ InLegalBERT loaded successfully")
            except Exception as e:
                print(f"⚠️  Error loading local model: {str(e)}")
                print("⚠️  Using InLegalBERT from Hugging Face")
                self.inlegal_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
                self.inlegal_model = AutoModel.from_pretrained("law-ai/InLegalBERT")
        else:
            print("⚠️  Using InLegalBERT from Hugging Face")
            self.inlegal_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
            self.inlegal_model = AutoModel.from_pretrained("law-ai/InLegalBERT")
        
        # Load FLAN-T5 model for simplification
        print("🔄 Loading FLAN-T5 Simplification Model...")
        # Check for enhanced model first
        enhanced_path = self.models_dir / "flan_t5_enhanced"
        flan_t5_path = self.models_dir / "flan_t5"
        
        if enhanced_path.exists():
            print("🚀 Using enhanced FLAN-T5 model")
            self.t5_tokenizer = AutoTokenizer.from_pretrained(str(enhanced_path))
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(str(enhanced_path))
        elif flan_t5_path.exists():
            print("✅ Using local FLAN-T5 model")
            self.t5_tokenizer = AutoTokenizer.from_pretrained(str(flan_t5_path))
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(str(flan_t5_path))
        else:
            print("⚠️  Local FLAN-T5 not found, downloading...")
            self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        # Set models to evaluation mode
        if self.inlegal_model:
            self.inlegal_model.eval()
        if self.t5_model:
            self.t5_model.eval()
        
        print("🎉 All models loaded successfully!")
        print()
    
    def simplify_text(self, legal_text, max_length=None):
        """
        Simplify legal text using the loaded models
        """
        if not self.t5_model or not self.t5_tokenizer:
            return "❌ Error: Models not loaded. Please run load_models() first."
        
        try:
            # First try enhanced manual simplification for better results
            print("🔧 Using enhanced rule-based simplification for reliable results...")
            manual_result = self.enhanced_manual_simplification(legal_text)
            
            # If manual result is good, return it
            if len(manual_result.strip()) > 20 and manual_result.lower() != legal_text.lower():
                return manual_result
            
            # Otherwise try FLAN-T5 with better prompting
            print("🤖 Trying FLAN-T5 for additional improvements...")
            
            # Create a more specific instruction for FLAN-T5
            instruction = f"Explain this legal text in simple words that anyone can understand: {legal_text.strip()}"
            
            # Tokenize input
            max_len = max_length or self.config.get("max_length", 512)
            inputs = self.t5_tokenizer(
                instruction,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
                padding=True
            )
            
            # Generate simplified text with better parameters
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=200,
                    min_length=20,
                    num_beams=4,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    early_stopping=True,
                    pad_token_id=self.t5_tokenizer.pad_token_id,
                    eos_token_id=self.t5_tokenizer.eos_token_id
                )
            
            # Decode output
            flan_output = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Debug: Print raw output
            print(f"🔍 Raw FLAN-T5 output: {flan_output}")
            
            # Clean up FLAN-T5 output
            cleaned_output = self.post_process_flan_output(flan_output, legal_text)
            
            # Use FLAN-T5 output if it's better than manual, otherwise use manual
            if (len(cleaned_output.strip()) > 20 and 
                cleaned_output.lower() != legal_text.lower() and 
                "explain this legal text" not in cleaned_output.lower()):
                return cleaned_output
            else:
                print("🔧 Using manual simplification as FLAN-T5 output wasn't satisfactory...")
                return manual_result
            
        except Exception as e:
            logger.error(f"Error in text simplification: {str(e)}")
            # Fallback to manual simplification
            print("🔧 Falling back to manual simplification due to error...")
            return self.enhanced_manual_simplification(legal_text)

    def simplify_text_enhanced(self, text: str) -> dict:
        """Enhanced simplification with LLM integration"""
        print("🔄 Processing legal text...")
        
        # Get current system output
        current_result = self.simplify_text(text)
        current_simplified = current_result.get('simplified_text', text)
        
        if self.use_llm_enhancement:
            # Try LLM enhancement
            enhanced_simplified = self.llm_simplifier.enhance_simplification(
                text, current_simplified
            )
            
            # Create enhanced result
            enhanced_result = current_result.copy()
            enhanced_result['simplified_text'] = enhanced_simplified
            enhanced_result['enhancement_used'] = True
            
            return enhanced_result
        else:
            current_result['enhancement_used'] = False
            return current_result
    
    def simplify_text_with_local_llm(self, text: str) -> str:
        """Enhanced simplification with local LLM integration"""
        print("🔄 Processing legal text with local LLM enhancement...")
        
        # Get current system output
        current_simplified = self.simplify_text(text)
        
        if self.use_local_llm:
            # Try local LLM enhancement (100% free & offline)
            enhanced_simplified = self.local_llm_simplifier.enhance_simplification(
                text, current_simplified
            )
            return enhanced_simplified
        else:
            return current_simplified
    
    def toggle_llm_enhancement(self):
        """Toggle LLM enhancement on/off"""
        self.use_llm_enhancement = not self.use_llm_enhancement
        status = "enabled" if self.use_llm_enhancement else "disabled"
        print(f"🤖 LLM enhancement {status}")
    
    def toggle_local_llm(self):
        """Toggle local LLM enhancement on/off"""
        self.use_local_llm = not self.use_local_llm
        status = "enabled" if self.use_local_llm else "disabled"
        print(f"🤖 Local LLM enhancement {status}")
    
    def setup_local_llms(self):
        """Setup local LLM models"""
        print("🚀 Setting up local LLM models...")
        self.local_llm_simplifier.setup_models()
    
    def post_process_flan_output(self, flan_output, original_text):
        """Clean and improve FLAN-T5 output"""
        text = flan_output.strip()
        
        # Remove instruction echoing
        prefixes_to_remove = [
            "Explain this legal text in simple words that anyone can understand:",
            "Rewrite this legal text in simple English that ordinary people can understand:",
            "In simple terms:",
            "Simply put:",
            "This means:",
        ]
        
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # If output is too similar to input, return empty to trigger fallback
        if text.lower() == original_text.lower():
            return ""
        
        # Ensure proper capitalization
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Ensure proper ending punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def analyze_legal_text(self, legal_text):
        """
        Analyze legal text using InLegalBERT (optional feature)
        """
        if not self.inlegal_model or not self.inlegal_tokenizer:
            return "InLegalBERT not available for analysis"
        
        try:
            # Tokenize and get embeddings
            inputs = self.inlegal_tokenizer(
                legal_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.inlegal_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return f"✅ Legal text analyzed - Embedding shape: {embeddings.shape}"
            
        except Exception as e:
            return f"❌ Error analyzing text: {str(e)}"
    
    
    def post_process_output(self, text):
        """Post-process FLAN-T5 output for better readability"""
        if not text or len(text.strip()) < 10:
            return "Unable to simplify this text."
        
        # Clean up common issues
        text = text.strip()
        
        # Ensure proper capitalization
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Ensure proper ending punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Remove redundant phrases
        redundant_phrases = [
            "In simple terms:",
            "Simply put:",
            "To put it simply:",
            "In other words:",
        ]
        
        for phrase in redundant_phrases:
            if text.startswith(phrase):
                text = text[len(phrase):].strip()
                if text and not text[0].isupper():
                    text = text[0].upper() + text[1:]
        
        return text

    def enhanced_manual_simplification(self, legal_text):
        """Enhanced manual simplification with better patterns"""
        
        # Start with basic manual simplification
        simplified = self.manual_simplification(legal_text)
        
        # Additional enhancements for common legal patterns
        patterns = {
            r"The plaintiff filed a writ petition under Article (\d+).*?seeking mandamus": 
                lambda m: f"The person filing the case asked the court (under Article {m.group(1)} of the Constitution) to order someone to do their legal duty",
            
            r"seeking mandamus for the enforcement of statutory obligations":
                "asking the court to order proper enforcement of legal duties",
                
            r"non-compliance with statutory provisions":
                "not following the legal requirements",
                
            r"statutory obligations":
                "legal duties required by law",
                
            r"hereby.*?covenant.*?and agree":
                "promise",
                
            r"party of the first part":
                "first person",
                
            r"party of the second part": 
                "second person",
        }
        
        import re
        result = simplified
        for pattern, replacement in patterns.items():
            if callable(replacement):
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            else:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Clean up and format
        result = result.strip()
        if result and not result[0].isupper():
            result = result[0].upper() + result[1:]
        if result and result[-1] not in '.!?':
            result += '.'
            
        return result

    def manual_simplification(self, legal_text):
        """
        Manual rule-based simplification as fallback
        """
        # Dictionary of legal terms to simple terms
        legal_to_simple = {
            "plaintiff": "person filing the case",
            "respondent": "person being sued",
            "defendant": "person being sued",
            "writ petition": "formal legal request",
            "mandamus": "court order to do something", 
            "Article 32": "constitutional right to justice",
            "seeking": "asking for",
            "filed": "submitted",
            "statutory obligations": "legal duties",
            "non-compliance": "not following the rules",
            "hereby": "",
            "covenants and agrees": "promises",
            "indemnify": "protect from loss",
            "hold harmless": "protect from blame",
            "party of the first part": "first person",
            "party of the second part": "second person", 
            "aforesaid": "mentioned above",
            "constrained": "forced",
            "initiate": "start",
            "appropriate legal proceedings": "proper legal action",
            "whereas": "since",
            "heretofore": "until now",
            "hereafter": "from now on",
            "pursuant to": "according to",
            "notwithstanding": "despite",
            "in lieu of": "instead of",
            "inter alia": "among other things",
            "vis-a-vis": "compared to",
            "prima facie": "at first look",
            "bona fide": "genuine",
            "ipso facto": "by the fact itself",
            "mutatis mutandis": "with necessary changes"
        }
        
        simplified = legal_text
        
        # Replace legal terms with simple terms
        for legal_term, simple_term in legal_to_simple.items():
            # Case-insensitive replacement
            pattern = legal_term.lower()
            if pattern in simplified.lower():
                # Find the actual case in the text and replace it
                import re
                simplified = re.sub(re.escape(legal_term), simple_term, simplified, flags=re.IGNORECASE)
        
        # Additional simplifications
        simplified = simplified.replace(" against ", " versus ")
        simplified = simplified.replace(" under ", " according to ")
        simplified = simplified.replace("Constitution", "Indian Constitution")
        
        # Clean up extra spaces and improve readability
        simplified = ' '.join(simplified.split())
        
        # Add explanation prefix for clarity
        return f"In simple terms: {simplified}"

def print_banner():
    """Print the LegalEase banner"""
    print()
    print("🏛️  LegalEase - Legal Text Simplification for Indian Citizens")
    print("=" * 60)
    print("📖 Simplifying Indian legal texts into understandable English")
    print("🔒 Offline & Privacy-focused | No data storage")
    print("=" * 60)
    print()

def print_examples():
    """Print example legal texts"""
    examples = [
        {
            "title": "Court Judgment Extract",
            "text": "The plaintiff filed a writ petition under Article 32 of the Constitution seeking mandamus against the respondent for non-compliance with statutory obligations under the Prevention of Corruption Act, 1988."
        },
        {
            "title": "Contract Clause",
            "text": "The party of the first part hereby covenants and agrees to indemnify and hold harmless the party of the second part from any and all claims, damages, losses, costs and expenses."
        },
        {
            "title": "Legal Notice",
            "text": "Take notice that my client is constrained to initiate appropriate legal proceedings against you for recovery of the aforesaid amount together with interest and costs."
        }
    ]
    
    print("📋 Example Legal Texts:")
    print("-" * 40)
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['title']}:")
        print(f"   {example['text']}")
        print()

def interactive_mode(simplifier):
    """Run interactive simplification mode"""
    print("🎯 Interactive Mode - Enter legal text to simplify")
    print("💡 Type 'examples' to see sample texts")
    if OCR_AVAILABLE:
        print("🔍 Type 'ocr' to see OCR capabilities")
    print("❌ Type 'quit' or 'exit' to stop")
    print("=" * 50)
    print()
    
    while True:
        try:
            # Get user input
            print("📝 Enter legal text to simplify:")
            legal_text = input("> ").strip()
            
            # Handle commands
            if legal_text.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using LegalEase!")
                break
            elif legal_text.lower() == 'examples':
                print_examples()
                continue
            elif legal_text.lower() == 'ocr':
                print_ocr_info()
                continue
            elif not legal_text:
                print("⚠️  Please enter some legal text to simplify.")
                continue
            
            # Process the text
            print("\n🔄 Simplifying text...")
            start_time = time.time()
            
            simplified = simplifier.simplify_text_enhanced(legal_text)
            
            processing_time = time.time() - start_time
            
            # Display results
            print("\n" + "=" * 60)
            print("📜 ORIGINAL LEGAL TEXT:")
            print("-" * 60)
            print(legal_text)
            print()
            print("✨ SIMPLIFIED VERSION:")
            print("-" * 60)
            print(simplified)
            print()
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print("=" * 60)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Thank you for using LegalEase!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def batch_mode(simplifier, input_file, output_file=None):
    """Process multiple texts from a file"""
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    texts = data
                else:
                    texts = [data.get('text', str(data))]
            else:
                # Treat as plain text, split by double newlines
                content = f.read().strip()
                texts = [text.strip() for text in content.split('\n\n') if text.strip()]
        
        print(f"📂 Processing {len(texts)} text(s) from {input_file}")
        
        results = []
        for i, text in enumerate(texts, 1):
            print(f"🔄 Processing text {i}/{len(texts)}...")
            simplified = simplifier.simplify_text_enhanced(text)
            results.append({
                'original': text,
                'simplified': simplified
            })
        
        # Save results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to {output_file}")
        else:
            # Print to console
            for i, result in enumerate(results, 1):
                print(f"\n{'='*60}")
                print(f"TEXT {i}:")
                print(f"Original: {result['original']}")
                print(f"Simplified: {result['simplified']}")
        
    except Exception as e:
        print(f"❌ Error processing batch: {str(e)}")

def ocr_mode(simplifier, document_path, output_dir=None, simplify=True):
    """Process documents using OCR and optionally simplify extracted text"""
    if not OCR_AVAILABLE:
        print("❌ OCR functionality not available. Please install OCR dependencies:")
        print("   pip install pytesseract easyocr Pillow pdf2image opencv-python")
        return
    
    try:
        # Initialize OCR processor
        print("🔍 Initializing OCR processor...")
        ocr = LegalOCRProcessor()
        
        # Process document
        print(f"📄 Processing document: {document_path}")
        result = ocr.process_document(document_path)
        
        if "error" in result:
            print(f"❌ OCR Error: {result['error']}")
            return
        
        # Extract text
        extracted_text = result.get("best_text", result.get("combined_text", ""))
        confidence = result.get("best_confidence", result.get("total_confidence", 0))
        
        if not extracted_text:
            print("❌ No text could be extracted from the document")
            return
        
        print(f"✅ Text extracted successfully (Confidence: {confidence:.1f}%)")
        print(f"📊 Extracted {len(extracted_text)} characters")
        
        # Save extracted text
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save original extracted text
            doc_name = Path(document_path).stem
            text_file = output_path / f"{doc_name}_extracted.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"💾 Extracted text saved to: {text_file}")
        
        # Display extracted text (truncated)
        print("\n" + "=" * 60)
        print("📜 EXTRACTED TEXT:")
        print("-" * 60)
        display_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        print(display_text)
        print("=" * 60)
        
        # Simplify if requested
        if simplify and extracted_text.strip():
            print("\n🔄 Simplifying extracted text...")
            
            # Split long text into chunks for better processing
            chunks = []
            sentences = extracted_text.split('.')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) < 400:  # Keep chunks under 400 chars
                    current_chunk += sentence + "."
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + "."
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            simplified_chunks = []
            for i, chunk in enumerate(chunks, 1):
                if chunk.strip():
                    print(f"  Processing chunk {i}/{len(chunks)}...")
                    simplified = simplifier.simplify_text_enhanced(chunk)
                    simplified_chunks.append(simplified)
            
            final_simplified = " ".join(simplified_chunks)
            
            # Display simplified text
            print("\n" + "=" * 60)
            print("✨ SIMPLIFIED VERSION:")
            print("-" * 60)
            print(final_simplified)
            print("=" * 60)
            
            # Save simplified text
            if output_dir:
                simplified_file = output_path / f"{doc_name}_simplified.txt"
                with open(simplified_file, 'w', encoding='utf-8') as f:
                    f.write(final_simplified)
                print(f"💾 Simplified text saved to: {simplified_file}")
        
        # Cleanup
        ocr.cleanup()
        
    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")

def ocr_batch_mode(simplifier, input_dir, output_dir, simplify=True):
    """Process multiple documents using OCR"""
    if not OCR_AVAILABLE:
        print("❌ OCR functionality not available. Please install OCR dependencies:")
        print("   pip install pytesseract easyocr Pillow pdf2image opencv-python")
        return
    
    try:
        # Initialize OCR processor
        print("🔍 Initializing OCR processor for batch processing...")
        ocr = LegalOCRProcessor()
        
        # Process batch
        print(f"📁 Processing documents in: {input_dir}")
        results = ocr.process_batch(input_dir, output_dir)
        
        if "error" in results:
            print(f"❌ Batch processing error: {results['error']}")
            return
        
        # Display summary
        summary = results["summary"]
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total files: {results['total_files']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        if summary['successful'] > 0:
            print(f"   Average confidence: {summary.get('average_confidence', 0):.1f}%")
        
        # Simplify extracted texts if requested
        if simplify and summary['successful'] > 0:
            print(f"\n🔄 Simplifying {summary['successful']} extracted texts...")
            
            output_path = Path(output_dir)
            for file_result in results["processed_files"]:
                if "error" not in file_result:
                    doc_name = Path(file_result["source_file"]).stem
                    
                    # Read extracted text
                    text_file = output_path / f"{doc_name}_extracted.txt"
                    if text_file.exists():
                        with open(text_file, 'r', encoding='utf-8') as f:
                            extracted_text = f.read()
                        
                        if extracted_text.strip():
                            print(f"  Simplifying: {doc_name}")
                            simplified = simplifier.simplify_text_enhanced(extracted_text[:1000])  # Limit length
                            
                            # Save simplified text
                            simplified_file = output_path / f"{doc_name}_simplified.txt"
                            with open(simplified_file, 'w', encoding='utf-8') as f:
                                f.write(simplified)
        
        print(f"✅ Batch processing complete. Results saved to: {output_dir}")
        
        # Cleanup
        ocr.cleanup()
        
    except Exception as e:
        print(f"❌ Error in batch OCR processing: {str(e)}")

def print_ocr_info():
    """Print OCR capabilities and usage information"""
    print("🔍 OCR (Optical Character Recognition) Features:")
    print("-" * 50)
    
    if OCR_AVAILABLE:
        print("✅ OCR functionality is available")
        print("\n📄 Supported Document Types:")
        print("   • PDF documents (.pdf)")
        print("   • Images (.jpg, .jpeg, .png, .bmp, .tiff)")
        print("\n🎯 OCR Features:")
        print("   • Multiple OCR engines (Tesseract + EasyOCR)")
        print("   • Automatic image preprocessing")
        print("   • Confidence scoring")
        print("   • Batch document processing")
        print("   • Legal document optimization")
        print("\n📖 Usage Examples:")
        print("   # Extract and simplify from PDF")
        print("   python src/cli_app.py --ocr document.pdf")
        print("")
        print("   # Extract only (no simplification)")
        print("   python src/cli_app.py --ocr document.pdf --no-simplify")
        print("")
        print("   # Batch process documents")
        print("   python src/cli_app.py --ocr-batch input_folder/ --output output_folder/")
        print("")
        print("   # Process image with output directory")
        print("   python src/cli_app.py --ocr contract.jpg --output results/")
    else:
        print("❌ OCR functionality not available")
        print("\n📦 To enable OCR, install dependencies:")
        print("   pip install pytesseract easyocr Pillow pdf2image opencv-python")
        print("\n⚠️  Note: Tesseract OCR also needs to be installed separately:")
        print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Linux: sudo apt-get install tesseract-ocr")
        print("   macOS: brew install tesseract")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="LegalEase - Legal Text Simplification CLI with OCR Support")
    parser.add_argument('--input', '-i', help='Input file for batch processing')
    parser.add_argument('--output', '-o', help='Output file/directory for results')
    parser.add_argument('--text', '-t', help='Direct text input for simplification')
    parser.add_argument('--examples', action='store_true', help='Show example legal texts')
    
    # OCR arguments
    parser.add_argument('--ocr', help='Process document using OCR (PDF/image file)')
    parser.add_argument('--ocr-batch', help='Process multiple documents using OCR (input directory)')
    parser.add_argument('--no-simplify', action='store_true', help='Extract text only, do not simplify')
    parser.add_argument('--ocr-info', action='store_true', help='Show OCR capabilities and usage')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Handle OCR info
    if args.ocr_info:
        print_ocr_info()
        return
    
    # Handle examples
    if args.examples:
        print_examples()
        return
    
    # Initialize simplifier
    simplifier = LegalTextSimplifier()
    
    # Handle OCR processing (single document)
    if args.ocr:
        # Load models only if simplification is requested
        if not args.no_simplify:
            try:
                simplifier.load_models()
            except Exception as e:
                print(f"❌ Error loading models: {str(e)}")
                print("⚠️  Continuing with OCR only (no simplification)")
                args.no_simplify = True
        
        ocr_mode(simplifier, args.ocr, args.output, not args.no_simplify)
        return
    
    # Handle OCR batch processing
    if args.ocr_batch:
        if not args.output:
            print("❌ Output directory required for batch OCR processing")
            print("Usage: --ocr-batch input_dir --output output_dir")
            return
        
        # Load models only if simplification is requested
        if not args.no_simplify:
            try:
                simplifier.load_models()
            except Exception as e:
                print(f"❌ Error loading models: {str(e)}")
                print("⚠️  Continuing with OCR only (no simplification)")
                args.no_simplify = True
        
        ocr_batch_mode(simplifier, args.ocr_batch, args.output, not args.no_simplify)
        return
    
    # Load models for text processing
    try:
        simplifier.load_models()
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        print("💡 Make sure you have run the setup scripts first:")
        print("   1. python scripts/download_datasets.py")
        print("   2. python src/data_preprocessing.py")
        print("   3. python src/model_setup.py")
        print("   4. python src/training.py")
        return
    
    # Handle direct text input
    if args.text:
        print(f"🔄 Simplifying provided text...")
        simplified = simplifier.simplify_text_enhanced(args.text)
        print(f"\n📜 Original: {args.text}")
        print(f"✨ Simplified: {simplified}")
        return
    
    # Handle batch processing
    if args.input:
        batch_mode(simplifier, args.input, args.output)
        return
    
    # Default: interactive mode
    interactive_mode(simplifier)

if __name__ == "__main__":
    main()
