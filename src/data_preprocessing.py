#!/usr/bin/env python3
"""
Data Preprocessing Script for LegalEase Project
Converts MILDSum_Samples .txt files into CSV training dataset
with legal-to-layman simplification pairs
"""

import os
import re
import pandas as pd
import numpy as np
import nltk
from pathlib import Path
import json
from typing import List, Tuple, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class LegalTextSimplifier:
    """Class to handle legal text simplification preprocessing"""
    
    def __init__(self):
        self.legal_to_layman_mapping = {
            # Legal terms to layman translations
            "plaintiff": "person who filed the case",
            "defendant": "person being sued",
            "appellant": "person appealing the decision",
            "respondent": "other party in the appeal",
            "petitioner": "person making the request",
            "writ": "legal order",
            "mandamus": "court order to do something",
            "certiorari": "court order to review a case",
            "habeas corpus": "order to bring someone to court",
            "judgment": "court's decision",
            "decree": "official court order",
            "injunction": "court order to stop doing something",
            "interim": "temporary",
            "ex-parte": "one-sided hearing",
            "suo moto": "on court's own initiative",
            "prima facie": "at first glance",
            "bona fide": "genuine",
            "mala fide": "done in bad faith",
            "ultra vires": "beyond legal authority",
            "intra vires": "within legal authority",
            "statute": "law made by parliament",
            "ordinance": "temporary law",
            "notification": "official announcement",
            "gazette": "official publication",
            "promulgate": "officially announce",
            "adjudicate": "decide legally",
            "litigation": "legal case",
            "tribunal": "specialized court",
            "jurisdiction": "legal authority",
            "precedent": "previous similar case decision",
            "ratio decidendi": "main reason for the decision",
            "obiter dicta": "additional comments by judge",
            "res judicata": "matter already decided",
            "sub judice": "under court consideration",
            "locus standi": "right to appear in court",
            "caveat": "legal warning",
            "affidavit": "sworn written statement",
            "deposition": "testimony given under oath",
            "subpoena": "court order to appear as witness"
        }
    
    def simplify_legal_terms(self, text: str) -> str:
        """Replace legal terms with layman equivalents"""
        simplified_text = text.lower()
        
        for legal_term, layman_term in self.legal_to_layman_mapping.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(legal_term.lower()) + r'\b'
            simplified_text = re.sub(pattern, layman_term, simplified_text)
        
        return simplified_text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or text.strip() == "":
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def create_simplification_pairs(self, judgment_text: str, summary_text: str) -> List[Tuple[str, str]]:
        """Create training pairs from judgment and summary"""
        pairs = []
        
        if not judgment_text or not summary_text:
            return pairs
        
        # Split into sentences
        judgment_sentences = nltk.sent_tokenize(judgment_text)
        summary_sentences = nltk.sent_tokenize(summary_text)
        
        # Create pairs from original judgment and simplified summary
        for summary_sent in summary_sentences:
            if len(summary_sent.strip()) > 20:  # Skip very short sentences
                # Find most relevant judgment sentence (simple approach)
                best_judgment_sent = self.find_relevant_judgment_sentence(
                    summary_sent, judgment_sentences
                )
                
                if best_judgment_sent:
                    legal_text = self.clean_text(best_judgment_sent)
                    simplified_text = self.clean_text(
                        self.simplify_legal_terms(summary_sent)
                    )
                    
                    if legal_text and simplified_text and len(legal_text) > 10:
                        pairs.append((legal_text, simplified_text))
        
        return pairs
    
    def find_relevant_judgment_sentence(self, summary_sentence: str, judgment_sentences: List[str]) -> str:
        """Find the most relevant judgment sentence for a summary sentence"""
        summary_words = set(summary_sentence.lower().split())
        best_sentence = ""
        best_score = 0
        
        for judgment_sent in judgment_sentences:
            judgment_words = set(judgment_sent.lower().split())
            
            # Calculate word overlap
            common_words = summary_words.intersection(judgment_words)
            if len(judgment_words) > 0:
                score = len(common_words) / len(judgment_words)
                
                if score > best_score and len(judgment_sent) > 20:
                    best_score = score
                    best_sentence = judgment_sent
        
        return best_sentence

def find_mildsum_samples_dir():
    """Find the MILDSum_Samples directory"""
    possible_paths = [
        RAW_DATA_DIR / "MILDSum_Samples" / "Data" / "MILDSum_Samples",
        RAW_DATA_DIR / "MILDSum_Samples",
        RAW_DATA_DIR / "MILDSum" / "MILDSum_Samples",
        RAW_DATA_DIR / "MILDSum"
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found MILDSum directory at: {path}")
            return path
    
    logger.error("MILDSum_Samples directory not found!")
    return None

def read_txt_files(samples_dir: Path) -> List[Dict]:
    """Read all .txt files in the MILDSum_Samples directory"""
    txt_files = list(samples_dir.glob("**/*.txt"))
    logger.info(f"Found {len(txt_files)} .txt files")
    
    file_data = []
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_info = {
                'filename': txt_file.name,
                'filepath': str(txt_file),
                'content': content,
                'language': 'hindi' if 'HI_' in txt_file.name or 'hindi' in txt_file.name.lower() else 'english'
            }
            file_data.append(file_info)
            
        except Exception as e:
            logger.warning(f"Failed to read {txt_file}: {e}")
    
    return file_data

def create_training_dataset(file_data: List[Dict], target_pairs: int = 30) -> pd.DataFrame:
    """Create training dataset from file data"""
    simplifier = LegalTextSimplifier()
    
    # Group files by case/document
    english_files = [f for f in file_data if f['language'] == 'english']
    hindi_files = [f for f in file_data if f['language'] == 'hindi']
    
    logger.info(f"English files: {len(english_files)}, Hindi files: {len(hindi_files)}")
    
    all_pairs = []
    
    # Process English files (judgments and summaries)
    judgment_files = [f for f in english_files if 'judgment' in f['filename'].lower() or 'EN_' in f['filename']]
    summary_files = [f for f in english_files if 'summary' in f['filename'].lower()]
    
    logger.info(f"Judgment files: {len(judgment_files)}, Summary files: {len(summary_files)}")
    
    # Create pairs from judgment-summary combinations
    processed_count = 0
    for judgment_file in judgment_files[:min(10, len(judgment_files))]:  # Limit for MVP
        for summary_file in summary_files[:min(3, len(summary_files))]:  # Multiple summaries per judgment
            try:
                pairs = simplifier.create_simplification_pairs(
                    judgment_file['content'], 
                    summary_file['content']
                )
                
                for legal_text, simplified_text in pairs:
                    all_pairs.append({
                        'legal_text': legal_text,
                        'simplified_text': simplified_text,
                        'source_judgment': judgment_file['filename'],
                        'source_summary': summary_file['filename'],
                        'pair_type': 'judgment_summary'
                    })
                
                processed_count += len(pairs)
                logger.info(f"Created {len(pairs)} pairs from {judgment_file['filename']} + {summary_file['filename']}")
                
                if processed_count >= target_pairs:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing {judgment_file['filename']}: {e}")
        
        if processed_count >= target_pairs:
            break
    
    # If we need more pairs, create some manual examples
    if len(all_pairs) < target_pairs:
        manual_pairs = create_manual_examples(target_pairs - len(all_pairs))
        all_pairs.extend(manual_pairs)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_pairs)
    
    # Remove duplicates and very short texts
    if not df.empty:
        df = df.drop_duplicates(subset=['legal_text'])
        df = df[df['legal_text'].str.len() > 20]
        df = df[df['simplified_text'].str.len() > 15]
        
        # Limit to target number of pairs
        df = df.head(target_pairs)
    
    logger.info(f"Created final dataset with {len(df)} pairs")
    return df

def create_manual_examples(count: int) -> List[Dict]:
    """Create manual legal-to-layman examples"""
    manual_examples = [
        {
            'legal_text': 'The plaintiff has filed a writ petition under Article 32 of the Constitution seeking mandamus against the respondent.',
            'simplified_text': 'The person who filed the case has asked the court to order the other party to do something using constitutional rights.',
            'source_judgment': 'manual_example',
            'source_summary': 'manual_example',
            'pair_type': 'manual'
        },
        {
            'legal_text': 'The court granted an interim injunction restraining the defendant from proceeding with the construction.',
            'simplified_text': 'The court gave a temporary order stopping the person being sued from continuing the building work.',
            'source_judgment': 'manual_example',
            'source_summary': 'manual_example', 
            'pair_type': 'manual'
        },
        {
            'legal_text': 'The appellant contends that the lower court erred in not considering the precedent established in the landmark judgment.',
            'simplified_text': 'The person appealing says the lower court made a mistake by not looking at what was decided in an important previous case.',
            'source_judgment': 'manual_example',
            'source_summary': 'manual_example',
            'pair_type': 'manual'
        },
        {
            'legal_text': 'The tribunal held that the petitioner had locus standi to challenge the notification issued by the authority.',
            'simplified_text': 'The specialized court decided that the person making the request had the right to challenge the official announcement.',
            'source_judgment': 'manual_example',
            'source_summary': 'manual_example',
            'pair_type': 'manual'
        },
        {
            'legal_text': 'The judgment was delivered ex-parte as the respondent failed to appear despite proper service of notice.',
            'simplified_text': 'The court made its decision with only one side present because the other party did not come even after being properly notified.',
            'source_judgment': 'manual_example',
            'source_summary': 'manual_example',
            'pair_type': 'manual'
        }
    ]
    
    return manual_examples[:min(count, len(manual_examples))]

def save_dataset(df: pd.DataFrame, filename: str = "legal_simplification_dataset.csv"):
    """Save the dataset to CSV"""
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to: {output_path}")
    
    # Also save as JSON for easier inspection
    json_path = PROCESSED_DATA_DIR / filename.replace('.csv', '.json')
    df.to_json(json_path, orient='records', indent=2)
    logger.info(f"Dataset also saved as JSON: {json_path}")
    
    return output_path

def create_dataset_summary(df: pd.DataFrame):
    """Create summary statistics of the dataset"""
    summary = {
        'total_pairs': len(df),
        'avg_legal_text_length': df['legal_text'].str.len().mean(),
        'avg_simplified_text_length': df['simplified_text'].str.len().mean(),
        'pair_types': df['pair_type'].value_counts().to_dict(),
        'sample_pairs': df.head(3).to_dict('records')
    }
    
    summary_path = PROCESSED_DATA_DIR / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Dataset summary saved to: {summary_path}")
    
    # Print summary
    print("\n📊 Dataset Summary:")
    print(f"   Total pairs: {summary['total_pairs']}")
    print(f"   Average legal text length: {summary['avg_legal_text_length']:.1f} characters")
    print(f"   Average simplified text length: {summary['avg_simplified_text_length']:.1f} characters")
    print(f"   Pair types: {summary['pair_types']}")
    
    return summary

def main():
    """Main preprocessing function"""
    print("🔄 Starting Data Preprocessing for LegalEase")
    print("="*50)
    
    # Ensure processed data directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find MILDSum samples directory
    samples_dir = find_mildsum_samples_dir()
    if not samples_dir:
        print("❌ Please run the download script first: python scripts/download_datasets.py")
        return False
    
    # Read all txt files
    print("📖 Reading text files...")
    file_data = read_txt_files(samples_dir)
    
    if not file_data:
        print("❌ No .txt files found in MILDSum_Samples directory")
        return False
    
    print(f"   Found {len(file_data)} text files")
    
    # Create training dataset
    print("⚙️  Creating training dataset...")
    df = create_training_dataset(file_data, target_pairs=30)  # MVP target
    
    if df.empty:
        print("❌ Failed to create training dataset")
        return False
    
    # Save dataset
    print("💾 Saving dataset...")
    output_path = save_dataset(df)
    
    # Create summary
    print("📋 Creating dataset summary...")
    summary = create_dataset_summary(df)
    
    print("="*50)
    print("✅ Data preprocessing completed successfully!")
    print(f"📁 Dataset saved to: {output_path}")
    print(f"🎯 Ready for model training with {len(df)} legal-layman pairs")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
