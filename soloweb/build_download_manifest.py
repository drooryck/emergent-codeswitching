#!/usr/bin/env python
"""Build the download manifest: which FineWeb parquet shards to pull from the HF Hub.

Defaults (chosen for ~40B raw tokens/lang so >=30B survives the strict LID clean, and for cross-language
length-matching):
  EN = HuggingFaceFW/fineweb    sample/100BT           (plain FineWeb, NOT edu; has token_count)
  DE = HuggingFaceFW/fineweb-2  data/deu_Latn/train
Takes the FIRST N shards of each (deterministic). One manifest line per shard:
    idx  lang  repo  remote_path  local_root
Set EN_REPO/EN_PREFIX/EN_SHARDS (etc.) to override; e.g. EN_REPO=HuggingFaceFW/fineweb-edu for the edu variant."""
import os, argparse
from huggingface_hub import HfApi


def pick(repo, prefix, n):
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(prefix) and f.endswith(".parquet"))
    return files[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-repo", default=os.environ.get("EN_REPO", "HuggingFaceFW/fineweb"))
    ap.add_argument("--en-prefix", default=os.environ.get("EN_PREFIX", "sample/100BT"))
    ap.add_argument("--en-shards", type=int, default=int(os.environ.get("EN_SHARDS", "60")))
    ap.add_argument("--de-repo", default=os.environ.get("DE_REPO", "HuggingFaceFW/fineweb-2"))
    ap.add_argument("--de-prefix", default=os.environ.get("DE_PREFIX", "data/deu_Latn/train"))
    ap.add_argument("--de-shards", type=int, default=int(os.environ.get("DE_SHARDS", "72")))
    ap.add_argument("--raw-root", default=os.environ.get("RAW_ROOT") or None)
    ap.add_argument("--manifest", default=os.environ.get("DL_MANIFEST") or None)
    a = ap.parse_args()
    if not a.raw_root or not a.manifest:
        raise SystemExit("Set --raw-root and --manifest (or RAW_ROOT and DL_MANIFEST)")

    os.makedirs(os.path.dirname(a.manifest), exist_ok=True)
    rows = []
    for lang, repo, prefix, n, root in (("en", a.en_repo, a.en_prefix, a.en_shards, f"{a.raw_root}/eng"),
                                        ("de", a.de_repo, a.de_prefix, a.de_shards, f"{a.raw_root}/deu")):
        got = pick(repo, prefix, n)
        print(f"{lang}: {repo} '{prefix}' -> {len(got)} shards (requested {n})")
        for rp in got:
            rows.append((lang, repo, rp, root))
    with open(a.manifest, "w") as w:
        for idx, (lang, repo, rp, root) in enumerate(rows):
            w.write(f"{idx}\t{lang}\t{repo}\t{rp}\t{root}\n")
    print(f"wrote {a.manifest}: {len(rows)} shards")
    print(f"  submit with:  bash soloweb/submit_download_fineweb.sh")

if __name__ == "__main__":
    main()
