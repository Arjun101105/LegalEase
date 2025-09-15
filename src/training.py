#!/usr/bin/env python3
"""
Training Script for LegalEase Project
Fine-tune InLegalBERT for legal text simplification
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    T5ForConditionalGeneration, T5Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import json
import os
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

class LegalSimplificationDataset(Dataset):
    """Dataset class for legal text simplification"""
    
    def __init__(self, legal_texts, simplified_texts, tokenizer, max_length=512):
        self.legal_texts = legal_texts
        self.simplified_texts = simplified_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.legal_texts)
    
    def __getitem__(self, idx):
        legal_text = str(self.legal_texts[idx])
        simplified_text = str(self.simplified_texts[idx])
        
        # Tokenize legal text (input)
        legal_encoding = self.tokenizer(
            legal_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize simplified text (target)
        simplified_encoding = self.tokenizer(
            simplified_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length//2,  # Shorter target length
            return_tensors='pt'
        )
        
        return {
            'input_ids': legal_encoding['input_ids'].flatten(),
            'attention_mask': legal_encoding['attention_mask'].flatten(),
            'target_ids': simplified_encoding['input_ids'].flatten(),
            'target_attention_mask': simplified_encoding['attention_mask'].flatten(),
            'legal_text': legal_text,
            'simplified_text': simplified_text
        }

class T5SimplificationTrainer:
    """Trainer class for T5-based legal text simplification"""
    
    def __init__(self, model_dir=None, config=None):
        self.device = torch.device("cpu")
        self.model_dir = model_dir or MODELS_DIR / "t5_simplification"
        self.config = config or self.load_config()
        
        # Set CPU threads for optimization
        torch.set_num_threads(self.config.get("hardware_config", {}).get("cpu_threads", 4))
        
    def load_config(self):
        """Load training configuration"""
        config_path = MODELS_DIR / "simplification_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "training_params": {
                    "learning_rate": 3e-4,
                    "batch_size": 2,
                    "num_epochs": 3,
                    "warmup_steps": 50,
                    "weight_decay": 0.01,
                    "gradient_accumulation_steps": 8
                },
                "hardware_config": {
                    "device": "cpu",
                    "cpu_threads": 4,
                    "max_memory_gb": 4
                }
            }
    
    def setup_model(self):
        """Setup T5 model and tokenizer"""
        try:
            logger.info("🔧 Setting up T5 model for training...")
            
            if self.model_dir.exists():
                # Load from local directory
                self.tokenizer = T5Tokenizer.from_pretrained(str(self.model_dir))
                self.model = T5ForConditionalGeneration.from_pretrained(str(self.model_dir))
            else:
                # Load from Hugging Face
                model_name = "t5-small"
                self.tokenizer = T5Tokenizer.from_pretrained(model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(model_name)
                
                # Save locally
                self.model_dir.mkdir(parents=True, exist_ok=True)
                self.tokenizer.save_pretrained(str(self.model_dir))
                self.model.save_pretrained(str(self.model_dir))
            
            # Move to device
            self.model.to(self.device)
            
            logger.info("✅ T5 model setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model setup failed: {e}")
            return False
    
    def load_dataset(self):
        """Load and prepare dataset"""
        try:
            logger.info("📖 Loading dataset...")
            
            # Load processed dataset
            dataset_path = PROCESSED_DATA_DIR / "legal_simplification_dataset.csv"
            if not dataset_path.exists():
                logger.error(f"Dataset not found at {dataset_path}")
                logger.error("Please run data preprocessing first: python src/data_preprocessing.py")
                return False
            
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded dataset with {len(df)} samples")
            
            # Prepare texts for T5 (add task prefix)
            legal_texts = ["simplify legal text: " + text for text in df['legal_text'].tolist()]
            simplified_texts = df['simplified_text'].tolist()
            
            # Split dataset
            train_legal, val_legal, train_simplified, val_simplified = train_test_split(
                legal_texts, simplified_texts, 
                test_size=0.2, 
                random_state=42
            )
            
            logger.info(f"Training samples: {len(train_legal)}")
            logger.info(f"Validation samples: {len(val_legal)}")
            
            # Create datasets
            self.train_dataset = LegalSimplificationDataset(
                train_legal, train_simplified, self.tokenizer
            )
            self.val_dataset = LegalSimplificationDataset(
                val_legal, val_simplified, self.tokenizer
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Dataset loading failed: {e}")
            return False
    
    def create_data_loaders(self):
        """Create data loaders"""
        batch_size = self.config["training_params"]["batch_size"]
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Single-threaded for CPU
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Created data loaders with batch size: {batch_size}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        training_params = self.config["training_params"]
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_params["learning_rate"],
            weight_decay=training_params["weight_decay"]
        )
        
        # Scheduler
        num_training_steps = len(self.train_loader) * training_params["num_epochs"]
        num_training_steps = num_training_steps // training_params.get("gradient_accumulation_steps", 1)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=training_params["warmup_steps"],
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Setup optimizer with learning rate: {training_params['learning_rate']}")
        logger.info(f"Total training steps: {num_training_steps}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        training_params = self.config["training_params"]
        grad_accum_steps = training_params.get("gradient_accumulation_steps", 1)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            target_attention_mask = batch['target_attention_mask'].to(self.device)
            
            # Prepare labels (replace padding tokens with -100)
            labels = target_ids.clone()
            labels[target_ids == self.tokenizer.pad_token_id] = -100
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / grad_accum_steps  # Scale loss for gradient accumulation
            loss.backward()
            
            total_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(step+1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {epoch+1} - Average training loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # Prepare labels
                labels = target_ids.clone()
                labels[target_ids == self.tokenizer.pad_token_id] = -100
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                progress_bar.set_postfix({'val_loss': f'{outputs.loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"Epoch {epoch+1} - Average validation loss: {avg_loss:.4f}")
        return avg_loss
    
    def test_generation(self, sample_text=None):
        """Test text generation"""
        self.model.eval()
        
        if sample_text is None:
            sample_text = "simplify legal text: The plaintiff filed a writ petition under Article 32 of the Constitution seeking mandamus against the respondent."
        
        logger.info(f"Testing generation with: {sample_text[:100]}...")
        
        # Tokenize input
        inputs = self.tokenizer(
            sample_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=2,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Generated: {generated_text}")
        return generated_text
    
    def save_model(self, epoch=None):
        """Save the trained model"""
        save_dir = MODELS_DIR / "t5_legal_simplification_trained"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        
        # Save training info
        training_info = {
            "model_type": "T5ForConditionalGeneration",
            "base_model": "t5-small",
            "task": "legal_text_simplification",
            "training_config": self.config,
            "epoch": epoch,
            "save_date": str(pd.Timestamp.now())
        }
        
        with open(save_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Model saved to: {save_dir}")
        return save_dir
    
    def train(self):
        """Main training function"""
        try:
            logger.info("🚀 Starting training...")
            
            # Setup components
            if not self.setup_model():
                return False
            
            if not self.load_dataset():
                return False
            
            self.create_data_loaders()
            self.setup_optimizer()
            
            # Training loop
            training_params = self.config["training_params"]
            num_epochs = training_params["num_epochs"]
            
            train_losses = []
            val_losses = []
            
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                logger.info(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
                
                # Train
                train_loss = self.train_epoch(epoch)
                train_losses.append(train_loss)
                
                # Validate
                val_loss = self.validate(epoch)
                val_losses.append(val_loss)
                
                # Test generation
                if epoch % 1 == 0:  # Test every epoch for small dataset
                    self.test_generation()
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(epoch)
                    logger.info(f"✅ New best model saved (val_loss: {val_loss:.4f})")
            
            # Plot training curves
            self.plot_training_curves(train_losses, val_losses)
            
            logger.info("🎉 Training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return False
    
    def plot_training_curves(self, train_losses, val_losses):
        """Plot training and validation loss curves"""
        try:
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(train_losses) + 1)
            
            plt.plot(epochs, train_losses, 'b-', label='Training Loss')
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
            
            plt.title('Legal Text Simplification Training Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_path = MODELS_DIR / "training_curves.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved to: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save training curves: {e}")

def main():
    """Main training function"""
    print("🎓 Starting Legal Text Simplification Training")
    print("="*50)
    
    # Check if dataset exists
    dataset_path = PROCESSED_DATA_DIR / "legal_simplification_dataset.csv"
    if not dataset_path.exists():
        print("❌ Dataset not found. Please run data preprocessing first:")
        print("   python src/data_preprocessing.py")
        return False
    
    # Initialize trainer
    trainer = T5SimplificationTrainer()
    
    # Start training
    success = trainer.train()
    
    print("="*50)
    if success:
        print("🎉 Training completed successfully!")
        print("\n📋 Next Steps:")
        print("   1. Test the trained model: python src/gui_app.py")
        print("   2. Try simplifying legal texts!")
    else:
        print("❌ Training failed. Check the logs above.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
