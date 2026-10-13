#!/usr/bin/env python
"""Temperature-sweep generations for the FULL sw7 grid (oct2): all 49 (size, D) zero-CS SoloWeb
models x 11 temperatures (0 = greedy, pure temperature sampling above) x 363 prompts
(ts1-4sw gold-verified + clauseswitch_madeup) x 5 seeds, 48 new tokens.
Feeds the scale-vs-emergent-switching landscape. Output: sw7_temp_sweep_generations.csv."""
import glob
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

E12 = Path(__import__("os").environ["CS_ROOT"])
OUT = E12 / "generation_out"
LAB = {"t11": "11M", "t22": "22M", "gpt2sL4": "52M", "gpt2sL12": "108M",
       "gpt2mL16": "232M", "gpt2lL24": "511M", "gpt2xlL25": "817M"}
DMAP = {"D0p5B": "0.5B", "D1B": "1B", "D2p74B": "2.74B", "D5p5B": "5.5B",
        "D11B": "11B", "D24B": "24B", "D52B": "52B"}
TEMPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_NEW = 48
BATCH = 96

models = []
for d in sorted(glob.glob(str(E12 / "runs" / "exp12_sw7_*_s1"))):
    m = re.search(r"exp12_sw7_([a-z0-9L]+)_(D[0-9p]+B)_s1$", Path(d).name)
    if not m or m.group(1) not in LAB or m.group(2) not in DMAP: continue
    finals = glob.glob(f"{d}/models_packed/*/final")
    if finals:
        models.append((Path(d).name, f"{LAB[m.group(1)]}|{DMAP[m.group(2)]}", finals[0]))
prompts = [json.loads(l) for l in open(OUT / "prompts.jsonl")]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}; {len(models)} models x {len(TEMPS)} temps x {len(prompts)} prompts x 5 seeds",
      flush=True)

rows = []
for run, label, final in models:
    tok = AutoTokenizer.from_pretrained(final)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(final).to(device).eval()
    for temp in TEMPS:
        for seed in range(5):
            torch.manual_seed(10000 + 137 * seed)
            for s in range(0, len(prompts), BATCH):
                batch = prompts[s:s + BATCH]
                enc = tok([p["prompt"] for p in batch], return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    gen = model.generate(
                        **enc, max_new_tokens=MAX_NEW,
                        do_sample=temp > 0, temperature=max(temp, 1e-6) if temp > 0 else None,
                        top_k=0 if temp > 0 else None, top_p=None,
                        pad_token_id=tok.pad_token_id)
                comps = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for p, c in zip(batch, comps):
                    rows.append(dict(run=run, model=label, set=p["set"], prompt_id=p["prompt_id"],
                                     temp=temp, seed=seed, completion=c))
    print(f"{label}: done ({len(rows)} total)", flush=True)
    del model
    torch.cuda.empty_cache()

pd.DataFrame(rows).to_csv(OUT / "sw7_temp_sweep_generations.csv", index=False)
print(f"wrote sw7_temp_sweep_generations.csv ({len(rows)} rows)", flush=True)
