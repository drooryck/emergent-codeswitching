#!/usr/bin/env python
"""Post-hoc ACS scoring of a saved run checkpoint from the LOCAL TS-subset pairs csv
(no HF download — works on offline compute nodes). One summary row per model.

Usage:
  python score_acs_local.py --model-dir <run>/models_packed/<arch>/final --out-csv out.csv
The final/ dir ships its own tokenizer, so nothing else is needed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from acs_scoring import load_acs_pairs, score_acs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--pairs-csv", type=Path, required=True,
                    help="ACS pairs CSV from metrics/fetch_acs_subset.py")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-name", default="")
    args = ap.parse_args()

    pairs = load_acs_pairs(args.pairs_csv, args.limit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device).eval()
    metrics = score_acs(model, tokenizer, pairs, device, args.max_length, args.batch_size)

    row = {"model": args.model_name or str(args.model_dir), **metrics}
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(row, flush=True)


if __name__ == "__main__":
    main()
