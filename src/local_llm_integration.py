#!/usr/bin/env python3
"""
Local LLM Integration for LegalEase - 100% Free & Offline
Cross-platform support with automatic GPU detection
Works on Windows, Linux, and macOS
"""

import os
import json
import subprocess
import requests
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, BitsAndBytesConfig
)
import logging

logger = logging.getLogger(__name__)

class SystemDetector:
    """Detect system capabilities and optimize accordingly"""
    
    def __init__(self):
        self.os_type = platform.system().lower()
        self.is_windows = self.os_type == "windows"
        self.is_linux = self.os_type == "linux"
        self.is_mac = self.os_type == "darwin"
        
        # Detect GPU capabilities
        self.gpu_info = self.detect_gpu()
        self.device = self.get_optimal_device()
        
    def detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU capabilities across platforms"""
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "gpu_count": 0,
            "gpu_memory": 0,
            "gpu_names": [],
            "recommended_settings": {}
        }
        
        # CUDA (NVIDIA) detection
        if gpu_info["cuda_available"]:
            gpu_info["gpu_count"] = torch.cuda.device_count()
            for i in range(gpu_info["gpu_count"]):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_info["gpu_names"].append(f"{gpu_name} ({gpu_memory:.1f}GB)")
                gpu_info["gpu_memory"] = max(gpu_info["gpu_memory"], gpu_memory)
        
        # MPS (Apple Silicon) detection
        elif gpu_info["mps_available"]:
            gpu_info["gpu_count"] = 1
            gpu_info["gpu_names"] = ["Apple Silicon GPU"]
            gpu_info["gpu_memory"] = 8  # Estimate for most M1/M2 Macs
        
        # Set recommended settings based on GPU memory
        if gpu_info["gpu_memory"] >= 8:
            gpu_info["recommended_settings"] = {
                "max_model_size": "7b",
                "batch_size": 4,
                "precision": "float16"
            }
        elif gpu_info["gpu_memory"] >= 4:
            gpu_info["recommended_settings"] = {
                "max_model_size": "3b",
                "batch_size": 2,
                "precision": "float16"
            }
        else:
            gpu_info["recommended_settings"] = {
                "max_model_size": "1b",
                "batch_size": 1,
                "precision": "float32"
            }
        
        return gpu_info
    
    def get_optimal_device(self) -> str:
        """Get the optimal device for inference"""
        if self.gpu_info["cuda_available"]:
            return "cuda"
        elif self.gpu_info["mps_available"]:
            return "mps"
        else:
            return "cpu"
    
    def print_system_info(self):
        """Print detailed system information"""
        print("🖥️  System Information")
        print("=" * 40)
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.machine()}")
        print(f"Python: {platform.python_version()}")
        print()
        
        print("🚀 GPU Information")
        print("-" * 20)
        if self.gpu_info["cuda_available"]:
            print("✅ NVIDIA CUDA GPU detected!")
            for gpu_name in self.gpu_info["gpu_names"]:
                print(f"   🎮 {gpu_name}")
            print(f"   💾 Total GPU Memory: {self.gpu_info['gpu_memory']:.1f}GB")
        elif self.gpu_info["mps_available"]:
            print("✅ Apple Silicon GPU detected!")
            print("   🍎 Metal Performance Shaders available")
        else:
            print("💻 CPU-only mode (no GPU acceleration)")
        
        print(f"🎯 Optimal device: {self.device.upper()}")
        
        # Recommendations
        settings = self.gpu_info["recommended_settings"]
        if settings:
            print("\n💡 Recommended Settings:")
            print(f"   Max model size: {settings['max_model_size']}")
            print(f"   Batch size: {settings['batch_size']}")
            print(f"   Precision: {settings['precision']}")
        print()

class LocalLLMManager:
    """Cross-platform local LLM manager with GPU optimization"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent.parent
        self.models_dir = self.project_dir / "data" / "models" / "local_llms"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # System detection
        self.system = SystemDetector()
        self.device = self.system.device
        
        # GPU-optimized model configurations
        self.available_models = self.get_gpu_optimized_models()
        self.ollama_models = self.get_ollama_models_config()
        
        self.loaded_models = {}
        self.ollama_available = self.check_ollama()
    
    def get_gpu_optimized_models(self) -> Dict[str, Dict]:
        """Get models optimized for current hardware"""
        base_models = {
            "tiny_fast": {
                "name": "microsoft/DialoGPT-small",
                "size": "117MB",
                "speed": "Very Fast",
                "quality": "Basic",
                "hf_model": "microsoft/DialoGPT-small",
                "min_memory": 1
            },
            "small_balanced": {
                "name": "distilgpt2", 
                "size": "319MB",
                "speed": "Fast",
                "quality": "Good",
                "hf_model": "distilgpt2",
                "min_memory": 2
            },
            "medium_quality": {
                "name": "gpt2",
                "size": "548MB",
                "speed": "Medium",
                "quality": "Better",
                "hf_model": "gpt2",
                "min_memory": 3
            }
        }
        
        # Add GPU-specific models if available
        if self.system.gpu_info["gpu_memory"] >= 4:
            base_models["large_gpu"] = {
                "name": "gpt2-medium",
                "size": "1.5GB",
                "speed": "Slower",
                "quality": "High Quality",
                "hf_model": "gpt2-medium", 
                "min_memory": 4
            }
        
        if self.system.gpu_info["gpu_memory"] >= 8:
            base_models["xl_gpu"] = {
                "name": "gpt2-large",
                "size": "3.2GB", 
                "speed": "Slow",
                "quality": "Excellent",
                "hf_model": "gpt2-large",
                "min_memory": 8
            }
        
        return base_models
    
    def get_ollama_models_config(self) -> Dict[str, Dict]:
        """Get Ollama models optimized for current hardware"""
        base_models = {
            "tinyllama": {
                "name": "tinyllama:1.1b",
                "size": "637MB",
                "speed": "Very Fast", 
                "quality": "Basic",
                "min_memory": 1
            },
            "phi": {
                "name": "phi:2.7b",
                "size": "1.6GB",
                "speed": "Fast",
                "quality": "Excellent",
                "min_memory": 2
            }
        }
        
        # Add larger models for high-memory systems
        if self.system.gpu_info["gpu_memory"] >= 6 or (self.device == "cpu" and self.get_system_ram() >= 8):
            base_models["llama2_chat"] = {
                "name": "llama2:7b-chat",
                "size": "3.8GB",
                "speed": "Medium",
                "quality": "Very High",
                "min_memory": 6
            }
        
        if self.system.gpu_info["gpu_memory"] >= 12 or (self.device == "cpu" and self.get_system_ram() >= 16):
            base_models["llama2_13b"] = {
                "name": "llama2:13b-chat", 
                "size": "7.3GB",
                "speed": "Slow",
                "quality": "Excellent",
                "min_memory": 12
            }
        
        return base_models
    
    def get_system_ram(self) -> float:
        """Get system RAM in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 8  # Conservative estimate
    
    def check_ollama(self) -> bool:
        """Check if Ollama is installed and running (cross-platform)"""
        try:
            # Try different ports that Ollama might use
            ports = [11434, 11435]
            for port in ports:
                try:
                    response = requests.get(f"http://localhost:{port}/api/tags", timeout=3)
                    if response.status_code == 200:
                        return True
                except:
                    continue
            return False
        except:
            return False
    
    def install_ollama_guide(self):
        """Cross-platform Ollama installation guide"""
        print("🦙 Ollama Installation Guide (Cross-Platform)")
        print("=" * 50)
        print("Ollama provides the best quality local LLMs!")
        print()
        
        if self.system.is_windows:
            print("🪟 Windows Installation:")
            print("1. Download: https://ollama.ai/download/windows")
            print("2. Run the installer")
            print("3. Open Command Prompt or PowerShell")
            print("4. Test: ollama --version")
        
        elif self.system.is_linux:
            print("🐧 Linux Installation:")
            print("1. Run: curl -fsSL https://ollama.ai/install.sh | sh")
            print("2. Or download from: https://ollama.ai/download/linux")
            print("3. Test: ollama --version")
        
        elif self.system.is_mac:
            print("🍎 macOS Installation:")
            print("1. Download: https://ollama.ai/download/mac")
            print("2. Install the .dmg file")
            print("3. Open Terminal")
            print("4. Test: ollama --version")
        
        print()
        print("🎯 Recommended first model based on your system:")
        
        # Recommend based on available memory
        gpu_memory = self.system.gpu_info["gpu_memory"]
        system_ram = self.get_system_ram()
        
        if gpu_memory >= 8 or system_ram >= 16:
            print("   ollama pull llama2:7b-chat  # Best quality")
        elif gpu_memory >= 4 or system_ram >= 8:
            print("   ollama pull phi:2.7b        # Best balance")
        else:
            print("   ollama pull tinyllama:1.1b  # Fastest")
        
        print()
    
    def load_hf_model(self, model_key: str):
        """Load Hugging Face model with GPU optimization"""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        if model_key not in self.available_models:
            print(f"❌ Unknown model: {model_key}")
            return None
        
        model_info = self.available_models[model_key]
        model_path = self.models_dir / model_key
        
        # Check memory requirements
        required_memory = model_info.get("min_memory", 2)
        if self.device == "cuda" and self.system.gpu_info["gpu_memory"] < required_memory:
            print(f"⚠️  Model {model_key} requires {required_memory}GB GPU memory, falling back to CPU")
            device = "cpu"
        else:
            device = self.device
        
        # Download if not exists
        if not model_path.exists():
            print(f"📥 Downloading {model_info['name']} ({model_info['size']})...")
            if not self.download_hf_model(model_key):
                return None
        
        try:
            print(f"🔄 Loading {model_key} on {device.upper()}...")
            
            # Configure for optimal performance
            model_kwargs = {}
            if device == "cuda":
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                })
            elif device == "mps":
                model_kwargs.update({
                    "torch_dtype": torch.float16
                })
            else:
                model_kwargs.update({
                    "torch_dtype": torch.float32
                })
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                model = model.to(device)
            
            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create optimized pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                torch_dtype=model_kwargs["torch_dtype"],
                do_sample=True,
                temperature=0.7,
                max_new_tokens=200,
                pad_token_id=tokenizer.eos_token_id
            )
            
            self.loaded_models[model_key] = {
                "generator": generator,
                "tokenizer": tokenizer,
                "model": model,
                "device": device
            }
            
            print(f"✅ {model_key} loaded successfully on {device.upper()}!")
            return self.loaded_models[model_key]
            
        except Exception as e:
            print(f"❌ Error loading {model_key}: {e}")
            return None
    
    def download_hf_model(self, model_key: str) -> bool:
        """Download Hugging Face model (cross-platform)"""
        if model_key not in self.available_models:
            print(f"❌ Unknown model: {model_key}")
            return False
        
        model_info = self.available_models[model_key]
        model_path = self.models_dir / model_key
        
        try:
            print(f"📥 Downloading {model_info['name']} ({model_info['size']})...")
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_info['hf_model'])
            
            # Configure download based on system
            model_kwargs = {"low_cpu_mem_usage": True}
            if self.device == "cpu":
                model_kwargs["torch_dtype"] = torch.float32
            else:
                model_kwargs["torch_dtype"] = torch.float16
            
            model = AutoModelForCausalLM.from_pretrained(
                model_info['hf_model'],
                **model_kwargs
            )
            
            # Save locally
            model_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(model_path))
            model.save_pretrained(str(model_path))
            
            # Save system-specific info
            info_file = model_path / "model_info.json"
            with open(info_file, 'w') as f:
                json.dump({
                    **model_info,
                    "download_date": "2025-09-29",
                    "downloaded_for_device": self.device,
                    "system_info": {
                        "os": self.system.os_type,
                        "gpu_available": self.device != "cpu"
                    }
                }, f, indent=2)
            
            print(f"✅ {model_info['name']} downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {model_info['name']}: {e}")
            return False
    
    def list_available_models(self):
        """List all available models optimized for current system"""
        print("🤖 Available Local LLM Models")
        print("=" * 50)
        
        # Show system info first
        self.system.print_system_info()
        
        print("📦 Hugging Face Models:")
        for key, model in self.available_models.items():
            status = "✅ Ready" if self.is_hf_model_available(key) else "⬇️ Will download"
            device_info = f"({self.device.upper()})" if model.get("min_memory", 0) <= self.system.gpu_info.get("gpu_memory", 0) else "(CPU)"
            print(f"   {status} {model['name']} {device_info}")
            print(f"      Size: {model['size']} | Quality: {model['quality']}")
        
        print()
        if self.ollama_available:
            print("🦙 Ollama Models:")
            ollama_models = self.get_ollama_models()
            if ollama_models:
                for model in ollama_models:
                    print(f"   ✅ {model['name']}")
            else:
                print("   No models installed. Install with: ollama pull <model>")
        else:
            print("🦙 Ollama: Not installed")
            print("   Install for better models: https://ollama.ai/download")
        print()
    
    def is_hf_model_available(self, model_key: str) -> bool:
        """Check if Hugging Face model is downloaded"""
        model_path = self.models_dir / model_key
        return model_path.exists() and any(model_path.iterdir())
    
    def get_ollama_models(self) -> List[Dict]:
        """Get list of installed Ollama models"""
        if not self.ollama_available:
            return []
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                return response.json().get("models", [])
        except:
            pass
        return []
    
    def simplify_with_hf_model(self, legal_text: str, model_key: str = "medium_quality") -> Optional[str]:
        """Simplify legal text using GPU-optimized Hugging Face model"""
        model_data = self.load_hf_model(model_key)
        if not model_data:
            return None
        
        try:
            # Create specialized prompt
            prompt = f"""Legal Document Simplification:

Complex Legal Text: "{legal_text}"

Simple Explanation: """
            
            generator = model_data["generator"]
            device = model_data["device"]
            
            print(f"🔄 Processing on {device.upper()}...")
            
            # Generate with device-optimized settings
            result = generator(
                prompt,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            # Extract simplified text
            generated = result[0]["generated_text"]
            if "Simple Explanation:" in generated:
                simplified = generated.split("Simple Explanation:")[-1].strip()
                return self.clean_generated_text(simplified, legal_text)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in HF model simplification: {e}")
            return None
    
    def clean_generated_text(self, generated_text: str, original_text: str) -> str:
        """Clean and improve generated text"""
        if not generated_text:
            return ""
        
        text = generated_text.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Simple Explanation:", "In simple terms:", "Simply put:",
            "This means:", "Plain English:", "Explanation:"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Basic cleanup
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def get_best_available_model(self) -> tuple:
        """Get the best model for current hardware"""
        # Check Ollama first
        if self.ollama_available:
            ollama_models = self.get_ollama_models()
            if ollama_models:
                for model in ollama_models:
                    if "phi" in model["name"]:
                        return ("ollama", "phi:2.7b")
                    elif "llama2" in model["name"] and "chat" in model["name"]:
                        return ("ollama", model["name"])
                return ("ollama", ollama_models[0]["name"])
        
        # Use best HF model for hardware
        gpu_memory = self.system.gpu_info["gpu_memory"]
        if gpu_memory >= 8 and "xl_gpu" in self.available_models:
            return ("huggingface", "xl_gpu")
        elif gpu_memory >= 4 and "large_gpu" in self.available_models:
            return ("huggingface", "large_gpu")
        elif gpu_memory >= 2:
            return ("huggingface", "medium_quality")
        else:
            return ("huggingface", "small_balanced")
    
    def simplify_legal_text(self, legal_text: str) -> Optional[str]:
        """Main method with hardware optimization"""
        if not legal_text or len(legal_text.strip()) < 10:
            return None
        
        print("🤖 Using hardware-optimized local LLM...")
        
        # Get best model for hardware
        model_type, model_name = self.get_best_available_model()
        print(f"   Model: {model_name} ({model_type}) on {self.device.upper()}")
        
        # Try simplification
        if model_type == "ollama":
            result = self.simplify_with_ollama(legal_text, model_name)
        elif model_type == "huggingface":
            result = self.simplify_with_hf_model(legal_text, model_name)
        else:
            result = None
        
        if result and len(result.strip()) > 20:
            print("✅ Hardware-optimized LLM simplification successful!")
            return result
        else:
            print("⚠️  LLM simplification failed, using fallback")
            return None
    
    def simplify_with_ollama(self, legal_text: str, model_name: str) -> Optional[str]:
        """Simplify using Ollama (same as before but with better error handling)"""
        if not self.ollama_available:
            return None
        
        try:
            prompt = f"""You are a legal expert who explains complex legal documents in simple English.

Task: Rewrite this legal text in plain, everyday language.

Legal Text: {legal_text}

Simple Explanation:""
            
            data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 200
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                simplified = result.get("response", "").strip()
                return self.clean_generated_text(simplified, legal_text)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in Ollama simplification: {e}")
            return None

class LocalLegalSimplifier:
    """Enhanced legal text simplifier with local LLM integration"""
    
    def __init__(self):
        self.llm_manager = LocalLLMManager()
        
    def enhance_simplification(self, legal_text: str, fallback_simplified: str) -> str:
        """Enhance simplification using local LLMs with fallback"""
        
        # Try local LLM enhancement
        llm_result = self.llm_manager.simplify_legal_text(legal_text)
        
        if llm_result:
            # Post-process and combine with rule-based improvements
            enhanced = self.post_process_llm_output(llm_result)
            return enhanced
        else:
            # Fall back to current system
            print("🔧 Using current FLAN-T5 system as LLM enhancement unavailable")
            return fallback_simplified
    
    def post_process_llm_output(self, llm_output: str) -> str:
        """Post-process LLM output for better readability"""
        # Additional legal term replacements
        legal_replacements = {
            "impugned": "challenged",
            "per contra": "on the other hand", 
            "inter alia": "among other things",
            "vis-a-vis": "compared to",
            "suo moto": "on its own",
            "prima facie": "at first glance",
            "bona fide": "genuine",
            "ipso facto": "by the fact itself",
            "mandamus": "court order to do something",
            "writ petition": "formal legal request"
        }
        
        result = llm_output
        for legal_term, simple_term in legal_replacements.items():
            result = result.replace(legal_term, simple_term)
            result = result.replace(legal_term.title(), simple_term.title())
        
        return result
    
    def setup_models(self):
        """Interactive setup for local models"""
        print("🚀 Local LLM Setup for LegalEase")
        print("=" * 40)
        
        # Show available models
        self.llm_manager.list_available_models()
        
        print("\n🎯 Recommended Setup:")
        print("1. Start with TinyLlama (smallest, fastest)")
        print("2. Upgrade to Phi-2 (best balance)")
        print("3. Add Llama2-7B (highest quality)")
        
        print("\n" + "="*40)
        
        # Install Ollama if not available
        if not self.llm_manager.ollama_available:
            print("🏠 Ollama Setup (Recommended)")
            if input("Install Ollama for better LLM models? (y/n): ").lower().startswith('y'):
                self.llm_manager.install_ollama_guide()
                print("\nAfter installing Ollama, run this setup again!")
                return
        
        # Setup Hugging Face models
        print("\n📦 Hugging Face Models Setup")
        if input("Download a local model for immediate use? (y/n): ").lower().startswith('y'):
            print("Choose model size:")
            print("1. Small & Fast (117MB) - Good for basic tasks")
            print("2. Medium & Balanced (319MB) - Recommended") 
            print("3. Large & Quality (548MB) - Best results")
            
            choice = input("Enter choice (1-3): ").strip()
            model_map = {"1": "small_fast", "2": "medium_balanced", "3": "large_quality"}
            
            if choice in model_map:
                self.llm_manager.download_hf_model(model_map[choice])
            
        print("\n✅ Setup complete! Your LegalEase now has local LLM enhancement!")

# Demo and testing functions
def demo_local_llm():
    """Demo local LLM simplification"""
    print("🧪 Local LLM Legal Simplification Demo")
    print("=" * 45)
    
    simplifier = LocalLegalSimplifier()
    
    # Sample legal texts
    samples = [
        "The plaintiff filed a writ petition under Article 32 of the Constitution seeking mandamus against the respondent for non-compliance with statutory obligations.",
        "The appellant was constrained to file this appeal challenging the impugned order passed by the learned Single Judge.",
        "The party of the first part hereby covenants and agrees to indemnify and hold harmless the party of the second part."
    ]
    
    for i, sample in enumerate(samples, 1):
        print(f"\n📝 Sample {i}:")
        print(f"Original: {sample}")
        print()
        
        # Current system output (simulated)
        current_output = f"In simple terms: {sample.lower()}"
        
        # Enhanced with local LLM
        enhanced = simplifier.enhance_simplification(sample, current_output)
        
        print("Enhanced:", enhanced)
        print("-" * 60)

if __name__ == "__main__":
    # Run setup if called directly
    simplifier = LocalLegalSimplifier()
    simplifier.setup_models()