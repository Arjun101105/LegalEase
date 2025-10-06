#!/usr/bin/env python3
"""
Process MILDSum dataset for LegalEase evaluation
Converts legal judgment-summary pairs into evaluation format
"""

import os
import pandas as pd
import json
from pathlib import Path

def load_mildsum_data():
    """Load all MILDSum samples"""
    base_path = Path("data/raw/MILDSum_Samples/Data/MILDSum_Samples")
    samples = []
    
    print("📥 Loading MILDSum dataset...")
    
    for sample_dir in sorted(base_path.glob("Sample_*")):
        sample_num = sample_dir.name
        
        # Read judgment (source legal text)
        judgment_file = sample_dir / "EN_Judgment.txt"
        summary_file = sample_dir / "EN_Summary.txt"
        
        if judgment_file.exists() and summary_file.exists():
            with open(judgment_file, 'r', encoding='utf-8') as f:
                judgment_text = f.read().strip()
            
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_text = f.read().strip()
            
            # Extract case title from summary (usually at the end)
            lines = summary_text.split('\n')
            case_title = "Unknown Case"
            for line in reversed(lines):
                if line.startswith("Title:"):
                    case_title = line.replace("Title:", "").strip()
                    break
            
            samples.append({
                'sample_id': sample_num,
                'legal_text': judgment_text,
                'simplified_text': summary_text,
                'case_title': case_title,
                'source_judgment': f"MILDSum_{sample_num}",
                'source_summary': f"MILDSum_{sample_num}",
                'pair_type': 'mildsum_real',
                'word_count_legal': len(judgment_text.split()),
                'word_count_summary': len(summary_text.split())
            })
            
            print(f"   ✅ Loaded {sample_num}: {case_title}")
        else:
            print(f"   ⚠️  Missing files for {sample_num}")
    
    return samples

def create_evaluation_chunks(legal_text, chunk_size=500):
    """Break large legal text into smaller chunks for evaluation"""
    words = legal_text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def process_dataset():
    """Process MILDSum data into evaluation format"""
    samples = load_mildsum_data()
    
    if not samples:
        print("❌ No MILDSum samples found!")
        return
    
    print(f"\n📊 Processing {len(samples)} samples...")
    
    # Create comprehensive dataset
    eval_data = []
    chunk_data = []
    
    for sample in samples:
        # Add full judgment-summary pair
        eval_data.append({
            'legal_text': sample['legal_text'],
            'simplified_text': sample['simplified_text'],
            'source_judgment': sample['source_judgment'],
            'source_summary': sample['source_summary'],
            'pair_type': sample['pair_type'],
            'case_title': sample['case_title'],
            'chunk_id': 'full_text'
        })
        
        # Create smaller chunks for more granular evaluation
        chunks = create_evaluation_chunks(sample['legal_text'], chunk_size=300)
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                'legal_text': chunk,
                'simplified_text': f"Summary chunk {i+1} of {sample['case_title']}: " + 
                                 sample['simplified_text'][:200] + "...",
                'source_judgment': sample['source_judgment'],
                'source_summary': sample['source_summary'],
                'pair_type': f"{sample['pair_type']}_chunk",
                'case_title': sample['case_title'],
                'chunk_id': f"chunk_{i+1}"
            })
    
    # Save comprehensive dataset
    df_full = pd.DataFrame(eval_data)
    output_path = "data/processed/mildsum_evaluation_dataset.csv"
    df_full.to_csv(output_path, index=False)
    print(f"✅ Saved comprehensive dataset: {output_path}")
    print(f"   📊 {len(df_full)} full judgment-summary pairs")
    
    # Save chunked dataset for detailed evaluation
    df_chunks = pd.DataFrame(chunk_data)
    chunk_path = "data/processed/mildsum_chunked_dataset.csv"
    df_chunks.to_csv(chunk_path, index=False)
    print(f"✅ Saved chunked dataset: {chunk_path}")
    print(f"   📊 {len(df_chunks)} text chunks")
    
    # Create dataset statistics
    stats = {
        'total_samples': len(samples),
        'total_full_pairs': len(eval_data),
        'total_chunks': len(chunk_data),
        'avg_judgment_words': sum(s['word_count_legal'] for s in samples) / len(samples),
        'avg_summary_words': sum(s['word_count_summary'] for s in samples) / len(samples),
        'cases': [{'sample_id': s['sample_id'], 
                   'case_title': s['case_title'],
                   'legal_words': s['word_count_legal'],
                   'summary_words': s['word_count_summary']} for s in samples]
    }
    
    stats_path = "data/processed/mildsum_dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved dataset statistics: {stats_path}")
    
    # Print summary
    print(f"\n📈 Dataset Summary:")
    print(f"   • {len(samples)} real legal cases from MILDSum")
    print(f"   • Avg judgment length: {stats['avg_judgment_words']:.0f} words")
    print(f"   • Avg summary length: {stats['avg_summary_words']:.0f} words")
    print(f"   • Range: {min(s['word_count_legal'] for s in samples)} - {max(s['word_count_legal'] for s in samples)} words")
    
    return df_full, df_chunks, stats

if __name__ == "__main__":
    print("🚀 MILDSum Dataset Processing")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    # Process the dataset
    process_dataset()
    
    print("\n🎉 Processing completed!")
    print("\n📋 Next Steps:")
    print("   1. Run evaluation on full dataset: python scripts/evaluate_text_simplification.py --dataset mildsum")
    print("   2. Compare with manual dataset results")
    print("   3. Update report with comprehensive evaluation")