#!/usr/bin/env python
"""Tag temp-sweep completions with the real TongueSwitcher pipeline (CPU ok) and write
temp_sweep_tagged.csv with ts_tags/ts_pattern/ts_switches/ts_switch_rate_100w columns.
Validation of this pipeline vs gold TS held-out: 97.2% token agreement
(see rescore_qualitative_ts.py)."""
import json
import sys
from pathlib import Path

import pandas as pd

from ts_annotate import _setup_imports, build_tagger
from rescore_qualitative_ts import summarize, tag_texts

OUT = Path(__import__("os").environ["CS_ROOT"]) / "generation_out"

_setup_imports()
tagger = build_tagger()
df = pd.read_csv(OUT / "temp_sweep_generations.csv")
res = tag_texts(tagger, df["completion"].astype(str).tolist(), batch=128)
df["ts_tokens"] = [json.dumps(t, ensure_ascii=False) for t, _, _ in res]
df["ts_tags"] = [json.dumps(g, ensure_ascii=False) for _, g, _ in res]
summ = pd.DataFrame([summarize(g, t, p) for t, g, p in res])
for c in summ.columns:
    df[c] = summ[c]
df.to_csv(OUT / "temp_sweep_tagged.csv", index=False)
print(f"wrote temp_sweep_tagged.csv ({len(df)} rows, mean rate "
      f"{df.ts_switch_rate_100w.mean():.2f}/100w)", flush=True)
