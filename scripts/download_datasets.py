#!/usr/bin/env python3
"""
Dataset Download Script for LegalEase Project
Downloads MILDSum_Samples and prepares InLegalBERT model for offline use
"""

import os
import sys
import subprocess
import requests
import zipfile
import shutil
from pathlib import Path
import json

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models"

def ensure_git_installed():
    """Check if git is installed"""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed. Please install git to download datasets.")
        return False

def download_mildsum_samples():
    """Download MILDSum_Samples dataset from GitHub"""
    print("📥 Downloading MILDSum_Samples dataset...")
    
    mildsum_dir = RAW_DATA_DIR / "MILDSum_Samples"
    
    # Remove existing directory if it exists
    if mildsum_dir.exists():
        print(f"   Removing existing directory: {mildsum_dir}")
        shutil.rmtree(mildsum_dir)
    
    try:
        # Clone the repository
        cmd = [
            "git", "clone", 
            "https://github.com/Law-AI/MILDSum.git",
            str(mildsum_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check if the MILDSum_Samples directory exists in the cloned repo
        samples_dir = mildsum_dir / "MILDSum_Samples"
        if samples_dir.exists():
            print(f"✅ MILDSum_Samples downloaded successfully!")
            print(f"   Location: {samples_dir}")
            
            # List contents to verify
            contents = list(samples_dir.iterdir())
            print(f"   Contents: {[f.name for f in contents]}")
            return True
        else:
            print(f"⚠️  MILDSum_Samples directory not found in cloned repository")
            print(f"   Available directories: {[d.name for d in mildsum_dir.iterdir() if d.is_dir()]}")
            return True  # Still successful clone, just different structure
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone MILDSum repository:")
        print(f"   Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error downloading MILDSum_Samples: {e}")
        return False

def download_inlegalbert():
    """Download InLegalBERT model files for offline use"""
    print("📥 Downloading InLegalBERT model...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "law-ai/InLegalBERT"
        model_dir = MODELS_DIR / "InLegalBERT"
        
        print(f"   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(str(model_dir))
        
        print(f"   Downloading model...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(str(model_dir))
        
        print(f"✅ InLegalBERT downloaded successfully!")
        print(f"   Location: {model_dir}")
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "download_date": str(Path(__file__).stat().st_mtime),
            "model_size": "~110M parameters",
            "description": "Legal domain BERT model trained on Indian legal documents"
        }
        
        with open(model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        return True
        
    except ImportError:
        print("❌ Transformers library not installed. Please run: pip install transformers torch")
        return False
    except Exception as e:
        print(f"❌ Failed to download InLegalBERT: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("📥 Downloading NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        nltk_data = [
            'punkt',
            'stopwords', 
            'wordnet',
            'averaged_perceptron_tagger'
        ]
        
        for data_name in nltk_data:
            try:
                print(f"   Downloading {data_name}...")
                nltk.download(data_name, quiet=True)
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to download {data_name}: {e}")
        
        print("✅ NLTK data downloaded successfully!")
        return True
        
    except ImportError:
        print("❌ NLTK not installed. Please run: pip install nltk")
        return False
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def create_download_summary():
    """Create a summary of downloaded datasets"""
    print("📋 Creating download summary...")
    
    summary = {
        "download_status": {
            "mildsum_samples": RAW_DATA_DIR.joinpath("MILDSum_Samples").exists() or RAW_DATA_DIR.joinpath("MILDSum").exists(),
            "inlegalbert": MODELS_DIR.joinpath("InLegalBERT").exists(),
            "nltk_data": True  # Assume success if we got here
        },
        "file_structure": {},
        "next_steps": [
            "1. Run data preprocessing: python src/data_preprocessing.py",
            "2. Setup model: python src/model_setup.py", 
            "3. Begin training: python src/training.py"
        ]
    }
    
    # Check MILDSum structure
    if RAW_DATA_DIR.joinpath("MILDSum").exists():
        mildsum_contents = list(RAW_DATA_DIR.joinpath("MILDSum").iterdir())
        summary["file_structure"]["MILDSum"] = [f.name for f in mildsum_contents]
        
        # Look for samples specifically
        samples_dir = RAW_DATA_DIR.joinpath("MILDSum", "MILDSum_Samples")
        if samples_dir.exists():
            sample_contents = list(samples_dir.iterdir())
            summary["file_structure"]["MILDSum_Samples"] = [f.name for f in sample_contents]
    
    # Check InLegalBERT structure
    if MODELS_DIR.joinpath("InLegalBERT").exists():
        model_contents = list(MODELS_DIR.joinpath("InLegalBERT").iterdir())
        summary["file_structure"]["InLegalBERT"] = [f.name for f in model_contents]
    
    # Save summary
    summary_file = PROJECT_ROOT / "download_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Download summary saved to: {summary_file}")
    return summary

def main():
    """Main download function"""
    print("🚀 Starting LegalEase Dataset Download")
    print("="*50)
    
    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_downloads = 3
    
    # Check git installation
    if not ensure_git_installed():
        sys.exit(1)
    
    # Download datasets
    if download_mildsum_samples():
        success_count += 1
    
    if download_inlegalbert():
        success_count += 1
    
    if download_nltk_data():
        success_count += 1
    
    # Create summary
    summary = create_download_summary()
    
    print("="*50)
    print(f"📊 Download Summary: {success_count}/{total_downloads} successful")
    
    if success_count == total_downloads:
        print("🎉 All downloads completed successfully!")
        print("\n📋 Next Steps:")
        for step in summary["next_steps"]:
            print(f"   {step}")
    else:
        print("⚠️  Some downloads failed. Check the logs above.")
        print("   You may need to install missing dependencies or check internet connection.")
    
    return success_count == total_downloads

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
