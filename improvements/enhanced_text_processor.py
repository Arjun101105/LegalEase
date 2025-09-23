#!/usr/bin/env python3
"""
Advanced Text Processing Pipeline for Better Simplification
Multi-stage processing with quality validation
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class EnhancedLegalSimplifier:
    """Enhanced legal text simplifier with multi-stage processing"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir
        self.load_enhanced_configs()
        self.load_models()
    
    def load_enhanced_configs(self):
        """Load enhanced configuration files"""
        config_dir = Path(__file__).parent.parent / "data" / "models"
        
        # Load enhanced generation config
        config_path = config_dir / "enhanced_generation_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.get_default_enhanced_config()
        
        # Load enhanced legal dictionary
        dict_path = config_dir / "enhanced_legal_dictionary.json"
        if dict_path.exists():
            with open(dict_path, 'r') as f:
                self.legal_dict = json.load(f)
        else:
            self.legal_dict = {}
        
        # Load post-processing rules
        rules_path = config_dir / "post_processing_rules.json"
        if rules_path.exists():
            with open(rules_path, 'r') as f:
                self.post_processing_rules = json.load(f)
        else:
            self.post_processing_rules = []
    
    def get_default_enhanced_config(self):
        """Default enhanced configuration"""
        return {
            "model_params": {
                "max_length": 200,
                "min_length": 30,
                "num_beams": 4,
                "temperature": 0.6,
                "top_p": 0.9,
                "do_sample": True,
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.2,
                "repetition_penalty": 1.1
            }
        }
    
    def load_models(self):
        """Load T5 model and tokenizer"""
        try:
            if self.model_dir and self.model_dir.exists():
                self.tokenizer = T5Tokenizer.from_pretrained(str(self.model_dir))
                self.model = T5ForConditionalGeneration.from_pretrained(str(self.model_dir))
            else:
                self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
                self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
            
            self.device = torch.device("cpu")
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.model = None
            self.tokenizer = None
    
    def preprocess_legal_text(self, text: str) -> str:
        """Enhanced preprocessing of legal text"""
        # Clean and normalize
        text = ' '.join(text.split())
        
        # Add context-aware prompting
        if not any(prompt in text.lower() for prompt in ["simplify", "explain", "rewrite"]):
            text = f"Simplify this legal text while keeping key terms with explanations: {text}"
        
        return text
    
    def contextual_term_replacement(self, text: str) -> str:
        """Replace legal terms with contextual explanations"""
        enhanced_text = text
        
        for term, info in self.legal_dict.items():
            if term.lower() in enhanced_text.lower():
                # Create contextual replacement
                if "simple" in info:
                    replacement = f"{term} ({info['simple']})"
                    pattern = r'\b' + re.escape(term) + r'\b'
                    enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
        
        return enhanced_text
    
    def generate_with_enhanced_params(self, text: str) -> str:
        """Generate text with enhanced parameters"""
        if not self.model or not self.tokenizer:
            return "Model not loaded"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate with enhanced parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **self.config["model_params"]
                )
            
            # Decode output
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove task prefix if present
            for prefix in ["simplify this legal text while keeping key terms with explanations:", 
                          "rewrite in plain english", "make this legal text understandable"]:
                if generated.lower().startswith(prefix):
                    generated = generated[len(prefix):].strip()
            
            return generated
            
        except Exception as e:
            return f"Generation error: {e}"
    
    def post_process_output(self, text: str) -> str:
        """Apply post-processing rules for better quality"""
        processed_text = text
        
        for rule in self.post_processing_rules:
            if "pattern" in rule and "replacement" in rule:
                processed_text = re.sub(
                    rule["pattern"], 
                    rule["replacement"], 
                    processed_text, 
                    flags=re.IGNORECASE
                )
        
        # Additional quality improvements
        processed_text = self.improve_sentence_flow(processed_text)
        processed_text = self.ensure_proper_capitalization(processed_text)
        
        return processed_text.strip()
    
    def improve_sentence_flow(self, text: str) -> str:
        """Improve sentence flow and coherence"""
        # Remove redundant phrases
        text = re.sub(r'\bin simple terms:?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsimplified:?\s*', '', text, flags=re.IGNORECASE)
        
        # Fix common awkward constructions
        text = re.sub(r'\bthe person who filed the case\b', 'the plaintiff', text)
        text = re.sub(r'\bthe person being sued\b', 'the defendant', text)
        text = re.sub(r'\bthe person appealing\b', 'the appellant', text)
        
        # Improve connector words
        text = re.sub(r'\bversus\b', 'against', text)
        text = re.sub(r'\bwhereas\b', 'while', text)
        
        return text
    
    def ensure_proper_capitalization(self, text: str) -> str:
        """Ensure proper capitalization"""
        if not text:
            return text
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after periods
        text = re.sub(r'(\. )([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text
    
    def quality_check(self, original: str, simplified: str) -> Dict[str, any]:
        """Check quality of simplification"""
        quality_metrics = {
            "length_ratio": len(simplified) / len(original) if original else 0,
            "has_explanations": bool(re.search(r'\([^)]+\)', simplified)),
            "maintains_key_terms": any(term in simplified.lower() for term in 
                                     ["plaintiff", "defendant", "appellant", "court", "petition"]),
            "natural_flow": not any(phrase in simplified.lower() for phrase in 
                                  ["in simple terms", "simplified:", "versus the person"]),
            "proper_grammar": simplified[0].isupper() if simplified else False
        }
        
        quality_metrics["overall_score"] = sum([
            quality_metrics["has_explanations"] * 0.3,
            quality_metrics["maintains_key_terms"] * 0.2,
            quality_metrics["natural_flow"] * 0.3,
            quality_metrics["proper_grammar"] * 0.2
        ])
        
        return quality_metrics
    
    def simplify_legal_text(self, legal_text: str) -> Dict[str, any]:
        """Main simplification method with quality assessment"""
        # Step 1: Preprocess
        preprocessed = self.preprocess_legal_text(legal_text)
        
        # Step 2: Generate simplification
        generated = self.generate_with_enhanced_params(preprocessed)
        
        # Step 3: Post-process
        final_simplified = self.post_process_output(generated)
        
        # Step 4: Quality check
        quality = self.quality_check(legal_text, final_simplified)
        
        return {
            "original": legal_text,
            "simplified": final_simplified,
            "quality_metrics": quality,
            "processing_steps": {
                "preprocessed": preprocessed,
                "generated": generated,
                "post_processed": final_simplified
            }
        }

def demo_enhanced_simplifier():
    """Demonstrate the enhanced simplifier"""
    simplifier = EnhancedLegalSimplifier()
    
    test_text = "The plaintiff filed a writ petition under Article 32 of the Constitution seeking mandamus against the respondent for non-compliance with statutory obligations."
    
    result = simplifier.simplify_legal_text(test_text)
    
    print("🔍 Enhanced Legal Text Simplification Demo")
    print("=" * 50)
    print(f"Original: {result['original']}")
    print(f"Simplified: {result['simplified']}")
    print(f"\n📊 Quality Score: {result['quality_metrics']['overall_score']:.2f}")
    print("\n📋 Quality Metrics:")
    for metric, value in result['quality_metrics'].items():
        if metric != 'overall_score':
            print(f"  • {metric}: {value}")

if __name__ == "__main__":
    demo_enhanced_simplifier()
