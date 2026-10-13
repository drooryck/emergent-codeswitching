#!/usr/bin/env python
"""Rebuild the leak-mitigated ACS German–English subset from Hugging Face.

The full ACS sentences are gated (igorsterner/acs-benchmark). This repo ships only
metrics/acs_pair_ids.txt — the 7,397 pair ids whose observed sentence also occurs in
TongueSwitcher. After you accept the ACS terms, run:

  python metrics/fetch_acs_subset.py --out $CS_ROOT/data/acs_pairs.csv

Training pools must exclude the matching TongueSwitcher tweets; see DATA.md.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=Path, default=HERE / "acs_pair_ids.txt")
    ap.add_argument("--out", type=Path, default=HERE / "acs_pairs.csv")
    ap.add_argument("--dataset", default="igorsterner/acs-benchmark")
    ap.add_argument("--config", default="de-en")
    args = ap.parse_args()

    wanted = {line.strip() for line in args.ids.read_text(encoding="utf-8").splitlines() if line.strip()}
    from datasets import get_dataset_split_names, load_dataset

    found: dict[str, dict] = {}
    splits = get_dataset_split_names(args.dataset, args.config)
    for split in splits:
        ds = load_dataset(args.dataset, args.config, split=split)
        for row in ds:
            rid = str(row.get("id", ""))
            if rid in wanted and rid not in found:
                found[rid] = {
                    "id": rid,
                    "split": split,
                    "observed_sentence": row.get("observed_sentence", ""),
                    "manipulated_sentence": row.get("manipulated_sentence", ""),
                }

    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"ACS download missing {len(missing)} of {len(wanted)} ids (e.g. {next(iter(missing))})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "split", "observed_sentence", "manipulated_sentence"])
        w.writeheader()
        for rid in sorted(found):
            w.writerow(found[rid])
    print(f"wrote {len(found)} pairs to {args.out}")


if __name__ == "__main__":
    main()
