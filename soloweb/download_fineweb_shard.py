#!/usr/bin/env python
"""Download ONE FineWeb parquet shard from the HF Hub to a local dir. Idempotent: hf_hub_download
resumes/caches and verifies the etag, and we skip if the target already exists at the expected size.
Invoked per SLURM array task (see download_fineweb.sbatch)."""
import os, sys, argparse
from huggingface_hub import hf_hub_download, HfApi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--remote", required=True)     # path within the dataset repo
    ap.add_argument("--local-root", required=True)  # files land at <local-root>/<remote>
    a = ap.parse_args()

    dest = os.path.join(a.local_root, a.remote)
    if os.path.exists(dest):
        try:
            want = HfApi().get_paths_info(a.repo, [a.remote], repo_type="dataset")[0].size
            if want and os.path.getsize(dest) == want:
                print(f"[skip] {dest} ({want} bytes)"); return
        except Exception:
            if os.path.getsize(dest) > 0:
                print(f"[skip?] {dest} exists ({os.path.getsize(dest)} bytes; size unverified)"); return

    p = hf_hub_download(repo_id=a.repo, filename=a.remote, repo_type="dataset",
                        local_dir=a.local_root)
    print(f"[done] {p} ({os.path.getsize(p)} bytes)")

if __name__ == "__main__":
    main()
