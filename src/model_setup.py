#!/usr/bin/env python3
"""
Model Setup Script for LegalEase Project
Sets up InLegalBERT for legal text simplification with CPU optimization
"""

import torch
import torch.quantization
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    T5ForConditionalGeneration, T5Tokenizer,
    Trainer, TrainingArguments
)
import json
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

class InLegalBERTSetup:
    """Class to handle InLegalBERT model setup and optimization"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or MODELS_DIR / "InLegalBERT"
        self.optimized_model_dir = MODELS_DIR / "InLegalBERT_optimized"
        
        # CPU-only configuration
        self.device = torch.device("cpu")
        torch.set_num_threads(4)  # Optimize for 4-core CPUs
        
    def load_model(self):
        """Load InLegalBERT model and tokenizer"""
        try:
            if self.model_dir.exists():
                logger.info(f"Loading InLegalBERT from local directory: {self.model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self.model = AutoModel.from_pretrained(str(self.model_dir))
            else:
                logger.info("Loading InLegalBERT from Hugging Face Hub...")
                model_name = "law-ai/InLegalBERT"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                
                # Save locally for future use
                self.model_dir.mkdir(parents=True, exist_ok=True)
                self.tokenizer.save_pretrained(str(self.model_dir))
                self.model.save_pretrained(str(self.model_dir))
                logger.info(f"Model saved locally to: {self.model_dir}")
            
            # Move to CPU
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info("✅ InLegalBERT loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load InLegalBERT: {e}")
            return False
    
    def optimize_for_cpu(self):
        """Optimize model for CPU inference"""
        try:
            logger.info("🔧 Optimizing model for CPU inference...")
            
            # Create optimized model directory
            self.optimized_model_dir.mkdir(parents=True, exist_ok=True)
            
            # Apply dynamic quantization for CPU
            logger.info("   Applying dynamic quantization...")
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},  # Quantize linear layers
                dtype=torch.qint8   # Use 8-bit integers
            )
            
            # Save optimized model
            logger.info("   Saving optimized model...")
            torch.save(quantized_model.state_dict(), self.optimized_model_dir / "pytorch_model_quantized.bin")
            
            # Copy tokenizer and config
            self.tokenizer.save_pretrained(str(self.optimized_model_dir))
            config = AutoConfig.from_pretrained(str(self.model_dir))
            config.save_pretrained(str(self.optimized_model_dir))
            
            # Save optimization info
            optimization_info = {
                "optimization_type": "dynamic_quantization",
                "quantization_dtype": "qint8",
                "target_device": "cpu",
                "original_model_size": self.get_model_size(self.model),
                "optimized_model_size": self.get_model_size(quantized_model),
                "cpu_threads": torch.get_num_threads()
            }
            
            with open(self.optimized_model_dir / "optimization_info.json", "w") as f:
                json.dump(optimization_info, f, indent=2)
            
            logger.info("✅ Model optimization completed")
            logger.info(f"   Optimized model saved to: {self.optimized_model_dir}")
            
            # Update model reference
            self.model = quantized_model
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model optimization failed: {e}")
            return False
    
    def get_model_size(self, model):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        return round(model_size_mb, 2)
    
    def test_model_inference(self):
        """Test model inference with sample legal text"""
        try:
            logger.info("🧪 Testing model inference...")
            
            # Sample legal text
            test_text = "The plaintiff filed a writ petition under Article 32 seeking mandamus against the respondent."
            
            # Tokenize
            inputs = self.tokenizer(
                test_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get embeddings
            embeddings = outputs.last_hidden_state
            logger.info(f"   Input text: {test_text[:100]}...")
            logger.info(f"   Output embeddings shape: {embeddings.shape}")
            logger.info("✅ Model inference test successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model inference test failed: {e}")
            return False
    
    def create_simplification_model_config(self):
        """Create configuration for text simplification fine-tuning"""
        config = {
            "model_type": "text2text-generation",
            "base_model": "InLegalBERT",
            "task": "legal_text_simplification",
            "max_input_length": 512,
            "max_output_length": 256,
            "training_params": {
                "learning_rate": 5e-5,
                "batch_size": 4,  # Small batch size for low-spec hardware
                "num_epochs": 3,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "gradient_accumulation_steps": 4  # Simulate larger batch size
            },
            "hardware_config": {
                "device": "cpu",
                "cpu_threads": 4,
                "max_memory_gb": 4,
                "use_quantization": True
            }
        }
        
        config_path = MODELS_DIR / "simplification_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Simplification config saved to: {config_path}")
        return config

class T5SimplificationModel:
    """Alternative T5-based model for text simplification"""
    
    def __init__(self):
        self.model_name = "t5-small"  # Lightweight T5 model
        self.model_dir = MODELS_DIR / "t5_simplification"
        self.device = torch.device("cpu")
    
    def setup_t5_model(self):
        """Setup T5 model for simplification task"""
        try:
            logger.info("🔧 Setting up T5 model for simplification...")
            
            # Load T5 model and tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            # Move to CPU
            self.model.to(self.device)
            
            # Save locally
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(str(self.model_dir))
            self.model.save_pretrained(str(self.model_dir))
            
            logger.info("✅ T5 model setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ T5 model setup failed: {e}")
            return False
    
    def test_simplification(self):
        """Test T5 model for text simplification"""
        try:
            logger.info("🧪 Testing T5 simplification...")
            
            # Sample legal text with T5 prefix
            legal_text = "simplify: The plaintiff filed a writ petition under Article 32 seeking mandamus."
            
            # Tokenize
            inputs = self.tokenizer(
                legal_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_beams=2,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Decode
            simplified_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"   Legal text: {legal_text}")
            logger.info(f"   Simplified: {simplified_text}")
            logger.info("✅ T5 simplification test successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ T5 simplification test failed: {e}")
            return False

def check_system_resources():
    """Check system resources and provide recommendations"""
    logger.info("🖥️  Checking system resources...")
    
    # Check CPU cores
    cpu_count = os.cpu_count() or 1
    
    # Check available memory (approximate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        memory_gb = "unknown"
        available_memory_gb = "unknown"
    
    resources = {
        "cpu_cores": cpu_count,
        "total_memory_gb": memory_gb,
        "available_memory_gb": available_memory_gb,
        "pytorch_cpu_threads": torch.get_num_threads(),
        "recommendations": []
    }
    
    # Add recommendations
    if isinstance(memory_gb, (int, float)) and memory_gb < 8:
        resources["recommendations"].append("Consider using model quantization for memory optimization")
    
    if cpu_count < 4:
        resources["recommendations"].append("Consider using smaller batch sizes for training")
    
    if isinstance(available_memory_gb, (int, float)) and available_memory_gb < 2:
        resources["recommendations"].append("Close other applications to free up memory")
    
    logger.info(f"   CPU cores: {cpu_count}")
    logger.info(f"   Total memory: {memory_gb} GB")
    logger.info(f"   Available memory: {available_memory_gb} GB")
    
    # Save system info
    system_info_path = MODELS_DIR / "system_info.json"
    with open(system_info_path, "w") as f:
        json.dump(resources, f, indent=2, default=str)
    
    return resources

def main():
    """Main model setup function"""
    print("🤖 Starting Model Setup for LegalEase")
    print("="*50)
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check system resources
    resources = check_system_resources()
    
    success_count = 0
    total_tasks = 4
    
    # Setup InLegalBERT
    print("\n1️⃣ Setting up InLegalBERT...")
    inlegal_setup = InLegalBERTSetup()
    
    if inlegal_setup.load_model():
        success_count += 1
        
        # Optimize for CPU
        if inlegal_setup.optimize_for_cpu():
            success_count += 0.5  # Partial success
        
        # Test inference
        if inlegal_setup.test_model_inference():
            success_count += 0.5  # Partial success
    
    # Create simplification config
    print("\n2️⃣ Creating simplification configuration...")
    try:
        config = inlegal_setup.create_simplification_model_config()
        success_count += 1
        logger.info("✅ Simplification config created")
    except Exception as e:
        logger.error(f"❌ Failed to create config: {e}")
    
    # Setup T5 alternative (optional)
    print("\n3️⃣ Setting up T5 alternative model...")
    t5_setup = T5SimplificationModel()
    
    if t5_setup.setup_t5_model():
        success_count += 1
        
        # Test T5 simplification
        if t5_setup.test_simplification():
            success_count += 0.5  # Partial success
    
    # Final summary
    print("="*50)
    print(f"📊 Model Setup Summary: {success_count}/{total_tasks} tasks completed")
    
    if success_count >= 3:
        print("🎉 Model setup completed successfully!")
        print("\n📋 Next Steps:")
        print("   1. Run training: python src/training.py")
        print("   2. Test GUI: python src/gui_app.py")
        print("   3. Start simplifying legal texts!")
    else:
        print("⚠️  Some setup tasks failed. Check the logs above.")
        print("   You may need to install missing dependencies.")
    
    return success_count >= 3

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
