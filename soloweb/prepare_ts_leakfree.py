#!/usr/bin/env python
"""Drop TongueSwitcher documents whose text contains an ACS observed sentence.

The paper removes every tweet that contains one of the leak-mitigated ACS
observed sentences (7,397 pairs → 7,674 tweets in the official dump) so ACS
scoring is not a train/test leak. This script does that anti-join.

Input TS jsonl: one object per tweet. Must have a ``text`` field (or ``tokens``,
from which text is joined). Optional ``tags`` / ``token_spans`` are copied through.

  python soloweb/prepare_ts_leakfree.py \
      --ts $CS_ROOT/data/tongueswitcher.jsonl \
      --acs-pairs $CS_ROOT/data/acs_pairs.csv \
      --out-train $CS_ROOT/data/tongueswitcher_train_leakfree.jsonl \
      --out-heldout $CS_ROOT/data/ts_heldout/valid_tongueswitcher_cs.jsonl \
      --heldout-n 6000
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _norm(s: str) -> str:
    return " ".join(s.split())


def _text(row: dict) -> str:
    t = row.get("text")
    if t:
        return str(t)
    toks = row.get("tokens")
    if isinstance(toks, list):
        return " ".join(str(x) for x in toks)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=Path, required=True, help="TongueSwitcher jsonl")
    ap.add_argument("--acs-pairs", type=Path, required=True, help="CSV from fetch_acs_subset.py")
    ap.add_argument("--out-train", type=Path, required=True)
    ap.add_argument("--out-excluded", type=Path, default=None)
    ap.add_argument("--out-heldout", type=Path, default=None)
    ap.add_argument("--heldout-n", type=int, default=0,
                    help="If >0 with --out-heldout, reserve this many leak-free docs for eval")
    args = ap.parse_args()

    observed = []
    with args.acs_pairs.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            s = _norm(row.get("observed_sentence", ""))
            if s:
                observed.append(s)
    # Longest first so a short sentence cannot mask a longer one in the report.
    observed.sort(key=len, reverse=True)

    n_in = n_ex = 0
    kept: list[str] = []
    excluded: list[str] = []
    with args.ts.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = _norm(_text(row))
            if not text:
                continue
            n_in += 1
            hit = any(obs in text for obs in observed)
            if hit:
                n_ex += 1
                excluded.append(line if line.endswith("\n") else line + "\n")
            else:
                if "text" not in row:
                    row["text"] = _text(row)
                kept.append(json.dumps(row, ensure_ascii=False) + "\n")

    held_n = min(args.heldout_n, len(kept)) if args.out_heldout and args.heldout_n > 0 else 0
    held, train = kept[:held_n], kept[held_n:]

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_train.write_text("".join(train), encoding="utf-8")
    if args.out_heldout and held_n:
        args.out_heldout.parent.mkdir(parents=True, exist_ok=True)
        args.out_heldout.write_text("".join(held), encoding="utf-8")
    if args.out_excluded:
        args.out_excluded.parent.mkdir(parents=True, exist_ok=True)
        args.out_excluded.write_text("".join(excluded), encoding="utf-8")

    print(
        f"ts={n_in} excluded={n_ex} train={len(train)} heldout={held_n} "
        f"acs_observed={len(observed)}",
        flush=True,
    )
    if n_ex < 1000:
        print(
            "warning: exclusion count is far below the paper's 7,674 tweets; "
            "check that --ts text fields match ACS observed sentences",
            flush=True,
        )


if __name__ == "__main__":
    main()
