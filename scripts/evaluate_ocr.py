#!/usr/bin/env python3
"""
Evaluate OCR performance using WER and CER against a labeled dataset.

Dataset expected: CSV with columns 'filename' and 'ground_truth'
The images/pdf files should be accessible under a base path passed via --files-dir

Usage:
  python3 scripts/evaluate_ocr.py --dataset data/processed/ocr_dataset.csv --files-dir data/ocr_samples --api-url http://127.0.0.1:8000

Installs (if missing): pip install jiwer python-Levenshtein requests pandas tqdm
"""

import argparse
import csv
import os
import time
from statistics import mean, median

import pandas as pd
import requests
from jiwer import wer, cer
from tqdm import tqdm


def post_ocr(api_url, file_path, timeout=60):
    files = {"file": open(file_path, "rb")}
    try:
        r = requests.post(f"{api_url.rstrip('/')}/api/ocr", files=files, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def evaluate(dataset_csv, files_dir, api_url, limit=None, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    if limit:
        df = df.head(limit)

    rows = []
    latencies = []
    wers = []
    cers = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="OCR Eval"):
        fname = r.get("filename")
        gt = str(r.get("ground_truth", "")).strip()
        path = os.path.join(files_dir, fname)
        if not os.path.exists(path):
            rows.append({"filename": fname, "error": "file_not_found"})
            continue

        start = time.time()
        out = post_ocr(api_url, path)
        elapsed = time.time() - start
        latencies.append(elapsed)

        pred = out.get("extracted_text", "") if out.get("success") else ""

        try:
            sample_wer = wer(gt, pred)
        except Exception:
            sample_wer = 1.0
        try:
            sample_cer = cer(gt, pred)
        except Exception:
            sample_cer = 1.0

        wers.append(sample_wer)
        cers.append(sample_cer)

        rows.append({
            "filename": fname,
            "ground_truth": gt,
            "prediction": pred,
            "wer": sample_wer,
            "cer": sample_cer,
            "latency_s": elapsed,
        })

    summary = {
        "num_samples": len(rows),
        "wer_mean": mean(wers) if wers else None,
        "cer_mean": mean(cers) if cers else None,
        "latency_mean_s": mean(latencies) if latencies else None,
        "latency_median_s": median(latencies) if latencies else None,
    }

    out_csv = os.path.join(out_dir, "ocr_results.csv")
    keys = list(rows[0].keys()) if rows else ["filename", "ground_truth", "prediction"]
    with open(out_csv, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\n=== OCR Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"Detailed results written to: {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--files-dir", required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    evaluate(args.dataset, args.files_dir, args.api_url, limit=args.limit, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
