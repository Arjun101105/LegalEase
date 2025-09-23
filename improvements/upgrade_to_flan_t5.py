#!/usr/bin/env python3
"""
Upgrade LegalEase to FLAN-T5 for Better Performance
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import torch

def upgrade_to_flan_t5():
    """Upgrade existing model to FLAN-T5"""
    print("🚀 Upgrading LegalEase to FLAN-T5...")
    
    model_name = "google/flan-t5-small"
    models_dir = Path(__file__).parent.parent / "data" / "models" / "flan_t5"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and save FLAN-T5
    print("📥 Downloading FLAN-T5...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Save locally
    tokenizer.save_pretrained(str(models_dir))
    model.save_pretrained(str(models_dir))
    
    print("✅ FLAN-T5 upgrade completed!")
    print(f"Model saved to: {models_dir}")
    
    # Test the model
    test_text = "Simplify this legal text: The plaintiff filed a writ petition seeking mandamus."
    inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=150, num_beams=3)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test result: {result}")

if __name__ == "__main__":
    upgrade_to_flan_t5()
