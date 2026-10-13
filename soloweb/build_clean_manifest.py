#!/usr/bin/env python
"""Build the array-task manifest for the FineWeb LID decontamination job.

One line per (input shard, row-group chunk). Columns (TAB-separated):
    idx  lang  fmt  input  start  end  out  stats
Parquet shards are split into ceil(num_row_groups / CHUNK_RG) chunks; jsonl inputs are one task per file.
Set EN_GLOB / DE_GLOB (or --en-glob / --de-glob) to the downloaded shard globs."""
import os, glob, math, argparse
import pyarrow.parquet as pq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-glob", default=os.environ.get("EN_GLOB") or None)
    ap.add_argument("--de-glob", default=os.environ.get("DE_GLOB") or None)
    ap.add_argument("--out-root", default=os.environ.get("OUT_ROOT") or None)
    ap.add_argument("--chunk-rg", type=int, default=int(os.environ.get("CHUNK_RG", "120")))
    ap.add_argument("--manifest", default=os.environ.get("MANIFEST") or None)
    a = ap.parse_args()
    if not all([a.en_glob, a.de_glob, a.out_root, a.manifest]):
        raise SystemExit("Set --en-glob --de-glob --out-root --manifest (or EN_GLOB DE_GLOB OUT_ROOT MANIFEST)")

    os.makedirs(os.path.dirname(a.manifest), exist_ok=True)
    rows = []
    for lang, patt in (("en", a.en_glob), ("de", a.de_glob)):
        files = sorted(glob.glob(patt))
        if not files:
            print(f"[warn] no files for {lang}: {patt}")
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            fmt = "parquet" if f.endswith(".parquet") else "jsonl"
            if fmt == "parquet":
                nrg = pq.ParquetFile(f).num_row_groups
                nchunks = math.ceil(nrg / a.chunk_rg)
                for c in range(nchunks):
                    s, e = c * a.chunk_rg, min((c + 1) * a.chunk_rg, nrg)
                    out = f"{a.out_root}/chunks/{lang}/{base}.c{c:03d}.jsonl"
                    stats = f"{a.out_root}/stats/{lang}/{base}.c{c:03d}.json"
                    rows.append((lang, fmt, f, s, e, out, stats))
            else:
                out = f"{a.out_root}/chunks/{lang}/{base}.jsonl"
                stats = f"{a.out_root}/stats/{lang}/{base}.json"
                rows.append((lang, fmt, f, 0, -1, out, stats))

    with open(a.manifest, "w") as w:
        for idx, (lang, fmt, f, s, e, out, stats) in enumerate(rows):
            w.write(f"{idx}\t{lang}\t{fmt}\t{f}\t{s}\t{e}\t{out}\t{stats}\n")
    ntasks = len(rows)
    nen = sum(1 for r in rows if r[0] == "en"); nde = ntasks - nen
    print(f"wrote {a.manifest}: {ntasks} tasks ({nen} en + {nde} de), chunk_rg={a.chunk_rg}")
    print(f"  submit with:  bash soloweb/submit_clean_fineweb.sh")

if __name__ == "__main__":
    main()
