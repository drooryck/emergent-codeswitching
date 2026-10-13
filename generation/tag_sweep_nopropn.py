#!/usr/bin/env python
"""Re-tag sw7 completions to compute minority-run structure on the PROPN-excluded lexical
sequence (oct10) -- the multiword-switch metric currently includes proper nouns; this fixes
that. Writes sw7_nopropn_runs.csv with join keys + ts_majority_np, ts_max_minority_run_np,
ts_n_minority_runs_np."""
import sys
from pathlib import Path
import pandas as pd

from ts_annotate import _setup_imports, build_tagger
from rescore_qualitative_ts import tag_texts, _has_alpha

OUT = Path(__import__("os").environ["CS_ROOT"]) / "generation_out"

def run_stats_nopropn(tags, tokens, poses):
    lex = [t for t, tok, p in zip(tags, tokens, poses)
           if t in ("D", "E") and _has_alpha(tok) and p != "PROPN"]
    nD = sum(1 for t in lex if t == "D")
    maj = "D" if nD >= len(lex) - nD else "E"
    max_run = cur = n_runs = 0
    for t in lex:
        if t != maj:
            if cur == 0: n_runs += 1
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return maj, max_run, n_runs

_setup_imports()
tagger = build_tagger()
df = pd.read_csv(OUT / "sw7_temp_sweep_generations.csv")
res = tag_texts(tagger, df["completion"].astype(str).tolist(), batch=256)
stats = [run_stats_nopropn(g, t, p) for t, g, p in res]
out = df[["run", "set", "prompt_id", "temp", "seed"]].copy()
out["ts_majority_np"] = [s[0] for s in stats]
out["ts_max_minority_run_np"] = [s[1] for s in stats]
out["ts_n_minority_runs_np"] = [s[2] for s in stats]
out.to_csv(OUT / "sw7_nopropn_runs.csv", index=False)
mw = (out.ts_max_minority_run_np >= 2).mean()
print(f"wrote sw7_nopropn_runs.csv ({len(out)} rows; multiword(>=2) share {mw:.3f})", flush=True)
