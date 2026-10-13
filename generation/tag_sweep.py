#!/usr/bin/env python
"""Tag the sw7 sweep completions with the validated TongueSwitcher pipeline. Summary columns
only (no per-token arrays -- ~1M rows): ts_pattern, ts_switches, ts_mixed, ts_n_D/E/M/U,
ts_switch_rate_100w, ts_switches_nopropn, ts_switch_rate_100w_nopropn.
Output: sw7_temp_sweep_tagged.csv"""
import sys
from pathlib import Path

import pandas as pd

from ts_annotate import _setup_imports, build_tagger
from rescore_qualitative_ts import summarize, tag_texts

OUT = Path(__import__("os").environ["CS_ROOT"]) / "generation_out"

_setup_imports()
tagger = build_tagger()
df = pd.read_csv(OUT / "sw7_temp_sweep_generations.csv")
res = tag_texts(tagger, df["completion"].astype(str).tolist(), batch=256)
summ = pd.DataFrame([summarize(g, t, p) for t, g, p in res])
for c in summ.columns:
    df[c] = summ[c].values
df.to_csv(OUT / "sw7_temp_sweep_tagged.csv", index=False)
print(f"wrote sw7_temp_sweep_tagged.csv ({len(df)} rows, mean rate "
      f"{df.ts_switch_rate_100w.mean():.2f}/100w)", flush=True)
