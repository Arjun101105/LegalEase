#!/usr/bin/env python3
"""
Evaluate text simplification quality using automatic metrics and simple heuristics.

Metrics included:
- BLEU (sacrebleu)
- ROUGE-L (rouge_score)
- Exact match rate
- Avg length reduction / compression
- Per-sample latency and summary stats

Usage:
  python3 scripts/evaluate_text_simplification.py \
      --dataset data/processed/legal_simplification_dataset.csv \
      --api-url http://127.0.0.1:8000 \
      --limit 100

Install requirements (if missing):
  pip install sacrebleu rouge-score requests pandas tqdm

Output: results/text_simplification_results.csv and printed summary
"""

import argparse
import csv
import os
import time
from statistics import mean, median

import pandas as pd
import requests
import sacrebleu
from rouge_score import rouge_scorer
from tqdm import tqdm


def safe_post_simplify(api_url, text, timeout=30):
    try:
        resp = requests.post(f"{api_url.rstrip('/')}/api/simplify-text", json={"text": text}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def evaluate(dataset_path, api_url, limit=None, timeout=30, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    # Expect CSV with columns: legal_text, simplified_text
    df = pd.read_csv(dataset_path)
    
    # Determine dataset type for naming
    dataset_name = "manual" if "legal_simplification_dataset" in dataset_path else "mildsum"
    print(f"📊 Evaluating on {dataset_name} dataset: {len(df)} samples")
    if limit:
        df = df.head(limit)

    hyps = []
    refs = []
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rows = []
    latencies = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        src = str(row.get("legal_text", "")).strip()
        ref = str(row.get("simplified_text", "")).strip()

        start = time.time()
        out = safe_post_simplify(api_url, src, timeout=timeout)
        elapsed = time.time() - start
        latencies.append(elapsed)

        if out.get("success"):
            pred = out.get("simplified_text", "").strip()
        else:
            pred = ""

        hyps.append(pred)
        refs.append(ref)

        rouge_score = rouge.score(ref, pred)["rougeL"].fmeasure if pred else 0.0
        exact = 1 if pred == ref and pred else 0
        len_src = len(src.split())
        len_pred = len(pred.split())
        compression = (len_pred / len_src) if len_src > 0 else 0.0

        rows.append({
            "source": src,
            "reference": ref,
            "prediction": pred,
            "rouge_l": rouge_score,
            "exact_match": exact,
            "src_len_words": len_src,
            "pred_len_words": len_pred,
            "compression_ratio": compression,
            "latency_s": elapsed,
        })

    # Compute BLEU (sacrebleu expects list of hypotheses and list of list of references)
    try:
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        bleu_score = bleu.score
    except Exception:
        bleu_score = 0.0

    rouge_vals = [r["rouge_l"] for r in rows]

    summary = {
        "num_samples": len(rows),
        "bleu": bleu_score,
        "rouge_l_mean": mean(rouge_vals) if rouge_vals else 0.0,
        "exact_match_rate": mean([r["exact_match"] for r in rows]) if rows else 0.0,
        "avg_src_len": mean([r["src_len_words"] for r in rows]) if rows else 0.0,
        "avg_pred_len": mean([r["pred_len_words"] for r in rows]) if rows else 0.0,
        "avg_compression": mean([r["compression_ratio"] for r in rows]) if rows else 0.0,
        "latency_mean_s": mean(latencies) if latencies else 0.0,
        "latency_median_s": median(latencies) if latencies else 0.0,
        "latency_p95_s": sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0,
    }

    # Save results with dataset-specific naming
    out_csv = os.path.join(out_dir, f"text_simplification_results_{dataset_name}.csv")
    keys = list(rows[0].keys()) if rows else ["source", "reference", "prediction"]
    with open(out_csv, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n=== {dataset_name.upper()} Dataset Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"Detailed results written to: {out_csv}")
    
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/legal_simplification_dataset.csv")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    evaluate(args.dataset, args.api_url, limit=args.limit, timeout=args.timeout, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
