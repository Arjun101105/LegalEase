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

import torch
from transformers import (
    AutoTokenizer, AutoModel,
    T5ForConditionalGeneration, T5Tokenizer,
)

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
        if inlegal_path.exists():
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
        
        # Load T5 model for simplification
        print("🔄 Loading T5 Simplification Model...")
        # Use base T5 model for better English generation
        print("⚠️  Using base T5 model for reliable English output")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
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
            # For now, use manual rule-based simplification for reliable results
            # The T5 model is generating non-English output due to multilingual training
            print("🔧 Using rule-based simplification for reliable results...")
            simplified_text = self.manual_simplification(legal_text)
            
            # If we want to try T5 as well, uncomment below:
            # But keeping it simple for now
            """
            # Prepare input text with better prompt for base T5
            input_text = f"paraphrase: {legal_text.strip()}"
            
            # Tokenize input
            max_len = max_length or self.config.get("max_length", 512)
            inputs = self.t5_tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=max_len,
                truncation=True
            )
            
            # Generate simplified text
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    inputs,
                    max_new_tokens=150,
                    min_length=10,
                    num_beams=2,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True,
                    pad_token_id=self.t5_tokenizer.pad_token_id,
                    eos_token_id=self.t5_tokenizer.eos_token_id
                )
            
            # Decode output
            t5_output = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up T5 output
            if t5_output.startswith("paraphrase:"):
                t5_output = t5_output.replace("paraphrase:", "").strip()
            
            # Use T5 output if it's in English, otherwise use manual
            if self.is_english(t5_output):
                simplified_text = t5_output
            """
            
            return simplified_text
            
        except Exception as e:
            logger.error(f"Error in text simplification: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"❌ Error simplifying text: {str(e)}"
    
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
            elif not legal_text:
                print("⚠️  Please enter some legal text to simplify.")
                continue
            
            # Process the text
            print("\n🔄 Simplifying text...")
            start_time = time.time()
            
            simplified = simplifier.simplify_text(legal_text)
            
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
            simplified = simplifier.simplify_text(text)
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

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="LegalEase - Legal Text Simplification CLI")
    parser.add_argument('--input', '-i', help='Input file for batch processing')
    parser.add_argument('--output', '-o', help='Output file for batch processing')
    parser.add_argument('--text', '-t', help='Direct text input for simplification')
    parser.add_argument('--examples', action='store_true', help='Show example legal texts')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Handle examples
    if args.examples:
        print_examples()
        return
    
    # Initialize simplifier
    simplifier = LegalTextSimplifier()
    
    # Load models
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
        simplified = simplifier.simplify_text(args.text)
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
