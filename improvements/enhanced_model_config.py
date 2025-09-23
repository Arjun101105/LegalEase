#!/usr/bin/env python3
"""
Advanced Model Configuration for Better Simplification Quality
Optimized hyperparameters and generation strategies
"""

import json
from pathlib import Path

# Enhanced T5 Generation Parameters
ENHANCED_GENERATION_CONFIG = {
    "model_params": {
        "max_length": 200,           # Longer output for better explanations
        "min_length": 30,            # Ensure substantial simplification
        "num_beams": 4,              # Better beam search for quality
        "temperature": 0.6,          # More controlled randomness
        "top_p": 0.9,               # Nucleus sampling for better flow
        "top_k": 50,                # Limit vocabulary choices
        "do_sample": True,          # Enable sampling for naturalness
        "early_stopping": True,     # Stop when appropriate
        "no_repeat_ngram_size": 3,  # Avoid repetition
        "length_penalty": 1.2,      # Encourage longer, more detailed output
        "repetition_penalty": 1.1   # Discourage repetition
    },
    
    "training_params": {
        "learning_rate": 5e-5,      # More conservative learning rate
        "batch_size": 4,            # Larger batch size if possible
        "num_epochs": 5,            # More epochs for better learning
        "warmup_steps": 100,        # Longer warmup
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,       # Gradient clipping
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "logging_steps": 10
    },
    
    "data_params": {
        "max_input_length": 512,
        "max_target_length": 256,   # Longer targets for detailed explanations
        "train_test_split": 0.2,
        "validation_split": 0.1
    }
}

# Enhanced Legal Term Dictionary with Context
ENHANCED_LEGAL_DICTIONARY = {
    # Format: "term": {"simple": "explanation", "context": "when to use"}
    "plaintiff": {
        "simple": "person who filed the case",
        "context": "civil cases",
        "example": "The plaintiff (person who filed the case) seeks damages."
    },
    "appellant": {
        "simple": "person appealing the decision", 
        "context": "appeal cases",
        "example": "The appellant (person appealing) challenges the ruling."
    },
    "mandamus": {
        "simple": "court order directing someone to do their duty",
        "context": "administrative law",
        "example": "The court issued mandamus (order to perform duty)."
    },
    "writ petition": {
        "simple": "formal legal request to a higher court",
        "context": "constitutional cases",
        "example": "Filed a writ petition (formal court request) under Article 32."
    },
    "non-compliance": {
        "simple": "failure to follow rules or orders",
        "context": "regulatory matters",
        "example": "Non-compliance (failure to follow rules) with safety standards."
    },
    "statutory obligations": {
        "simple": "duties required by law",
        "context": "legal compliance",
        "example": "Statutory obligations (legal duties) under the Contract Act."
    }
}

# Post-processing Rules for Natural Output
POST_PROCESSING_RULES = [
    {
        "pattern": r"\bIn simple terms:\s*",
        "replacement": "",
        "description": "Remove redundant prefix"
    },
    {
        "pattern": r"\baccording to constitutional right to justice of the Indian Constitution\b",
        "replacement": "under Article 32 of the Constitution",
        "description": "Fix awkward constitutional references"
    },
    {
        "pattern": r"\bversus the person being sued\b",
        "replacement": "against the respondent",
        "description": "Maintain legal precision"
    },
    {
        "pattern": r"\bcourt order to do something\b",
        "replacement": "court order (mandamus)",
        "description": "Add specific legal term with explanation"
    },
    {
        "pattern": r"\bfor not following the rules with legal duties\b", 
        "replacement": "for failing to carry out their legal duties",
        "description": "Improve sentence flow"
    }
]

def save_enhanced_config():
    """Save enhanced configuration to project"""
    config_dir = Path(__file__).parent.parent / "data" / "models"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Save enhanced generation config
    with open(config_dir / "enhanced_generation_config.json", "w") as f:
        json.dump(ENHANCED_GENERATION_CONFIG, f, indent=2)
    
    # Save enhanced dictionary
    with open(config_dir / "enhanced_legal_dictionary.json", "w") as f:
        json.dump(ENHANCED_LEGAL_DICTIONARY, f, indent=2)
    
    # Save post-processing rules
    with open(config_dir / "post_processing_rules.json", "w") as f:
        json.dump(POST_PROCESSING_RULES, f, indent=2)
    
    print("✅ Enhanced configurations saved!")

if __name__ == "__main__":
    save_enhanced_config()
