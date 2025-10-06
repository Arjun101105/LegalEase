#!/usr/bin/env python3
"""
Evaluate on chunked MILDSum data for realistic legal text simplification evaluation
"""

import pandas as pd
import os
import sys

# Add the scripts directory to Python path
sys.path.append('scripts')
from evaluate_text_simplification import evaluate

def run_chunked_evaluation():
    """Run evaluation on chunked dataset that's more manageable"""
    
    print("🚀 Running LegalEase Evaluation on Real Legal Dataset")
    print("=" * 60)
    
    # Load chunked dataset
    chunked_data = "data/processed/mildsum_chunked_dataset.csv"
    
    if not os.path.exists(chunked_data):
        print(f"❌ Chunked dataset not found: {chunked_data}")
        print("   Run: python scripts/process_mildsum_dataset.py first")
        return
    
    df = pd.read_csv(chunked_data)
    print(f"📊 Dataset loaded: {len(df)} text chunks from real legal cases")
    
    # Show sample
    print(f"\n📝 Sample chunk preview:")
    sample = df.iloc[0]
    print(f"   Case: {sample.get('case_title', 'Unknown')}")
    print(f"   Legal text: {sample['legal_text'][:200]}...")
    print(f"   Reference summary: {sample['simplified_text'][:200]}...")
    
    # Run evaluation on subset (chunks are more manageable)
    print(f"\n🔄 Running evaluation on first 10 chunks...")
    
    try:
        summary, results = evaluate(
            dataset_path=chunked_data,
            api_url="http://127.0.0.1:8000",
            limit=10,
            timeout=45,
            out_dir="results"
        )
        
        print(f"\n✅ Evaluation completed!")
        print(f"   📊 Results saved to: results/text_simplification_results_mildsum.csv")
        
        # Show comparison with manual dataset
        manual_results_path = "results/text_simplification_results.csv"
        if os.path.exists(manual_results_path):
            print(f"\n📈 Comparison Summary:")
            print(f"   MILDSum (Real Legal): BLEU={summary['bleu']:.1f}, ROUGE-L={summary['rouge_l_mean']:.3f}")
            
            # Quick comparison note
            if summary['bleu'] > 0 or summary['rouge_l_mean'] > 0:
                print(f"   ✅ Model successfully processed real legal texts!")
            else:
                print(f"   ⚠️  Low scores may indicate model challenges with complex legal language")
        
        return summary, results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print(f"   💡 Make sure backend is running: python src/gui_app.py")
        return None, None

if __name__ == "__main__":
    run_chunked_evaluation()