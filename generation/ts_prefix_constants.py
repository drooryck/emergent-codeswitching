#!/usr/bin/env python
"""Reduce ts_prefix_tags/chunk_*.jsonl into TDE/TEN constants for a list of dose values."""
import argparse
import glob
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True, help="Directory of chunk_*.jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("doses", nargs="*", type=int)
    a = ap.parse_args()

    files = sorted(glob.glob(str(Path(a.chunks_dir) / "chunk_*.jsonl")))
    blocks = []
    for f in files:
        with open(f) as fh:
            blocks += [json.loads(l) for l in fh]
    total = sum(b["bpe_pool"] for b in blocks)
    print(f"# {len(files)} chunks, {len(blocks)} blocks, total pool tokens {total:,}")
    doses = a.doses or [983520, 9835197, 98351971, 245879928, 491759856, total]
    out = {}
    for tgt in doses:
        B = D = E = M = 0.0
        for b in blocks:
            if B + b["bpe_pool"] <= tgt:
                B += b["bpe_pool"]
                D += b["D_bpe"]
                E += b["E_bpe"]
                M += b["M_bpe"]
            else:
                f = (tgt - B) / b["bpe_pool"]
                D += f * b["D_bpe"]
                E += f * b["E_bpe"]
                M += f * b["M_bpe"]
                B = tgt
                break
        Dm = D + M
        tde = round(tgt * Dm / (Dm + E))
        ten = tgt - tde
        out[tgt] = dict(TDE=tde, TEN=ten, de_share=round(Dm / (Dm + E), 4),
                        tagged_frac=round((Dm + E) / max(B, 1), 4))
        print(f"D_cs={tgt:>13,}  TDE={tde:>13,}  TEN={ten:>12,}  de-share={out[tgt]['de_share']:.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
