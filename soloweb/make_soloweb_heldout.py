#!/usr/bin/env python
"""Reserve the SoloWeb held-out eval split: the first N docs of each cleaned pool
(train pools must skip these via prepare_soloweb_pool.py --skip-docs N).
Basenames stay valid_monoweb_*.jsonl because the trainer looks for those names
under MONO_HELDOUT; the directory is SoloWeb."""
import json, os, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--clean-dir", required=True)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
for lang, tag in (("eng", "eng"), ("deu", "deu")):
    src = f"{a.clean_dir}/fineweb_clean_{tag}_train.jsonl"
    dst = f"{a.out_dir}/valid_monoweb_{tag}.jsonl"     # basename fixed by the train sbatch
    n = 0
    with open(src) as r, open(dst + ".tmp", "w") as w:
        for line in r:
            d = json.loads(line)
            w.write(json.dumps({"text": d["text"], "lang": tag, "source": "soloweb"},
                               ensure_ascii=False) + "\n")
            n += 1
            if n >= a.n: break
    os.replace(dst + ".tmp", dst)
    print(f"{dst}: {n} docs")
