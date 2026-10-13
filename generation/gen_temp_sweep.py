#!/usr/bin/env python
"""Temperature sweep generations (sept24 v2): 9 models x 11 temperatures x ~363 prompts x 5 seeds.

Temps 0.0 (greedy) then 0.1..1.0 pure temperature sampling (no top-k/top-p truncation).
48 new tokens; 5 sampled completions per (model, temp, prompt) via seeds 0-4 (greedy T=0 is
deterministic, so its 5 "seeds" are identical by construction — kept for uniform aggregation).
Models = the 9 runs used in the aug12 qualitative CSVs (sw7 size/D sweep + dose + p2b skews).
Output: temp_sweep_generations.csv (model, run, set, prompt_id, temp, seed, prompt, completion).
"""
import glob
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

E12 = Path(__import__("os").environ["CS_ROOT"])
OUT = E12 / "generation_out"
MODELS = [
    ("exp12_sw7_gpt2sL4_D0p5B_s1", "52M D0.5B 50/50"),
    ("exp12_sw7_gpt2sL4_D2p74B_s1", "52M D2.74B 50/50"),
    ("exp12_sw7_gpt2sL12_D2p74B_s1", "108M D2.74B 50/50"),
    ("exp12_sw7_gpt2mL16_D2p74B_s1", "232M D2.74B 50/50"),
    ("exp12_sw7_gpt2lL24_D2p74B_s1", "511M D2.74B 50/50"),
    ("exp12_swdose_gpt2sL4_D2p74B_cs9p6M_s1", "52M +9.6M CS dose"),
    ("exp12_swdose_gpt2sL4_D2p74B_csFULL_s1", "52M +full CS dose"),
    ("exp12_swp2b_gpt2sL4_en0p125B_de2p75B_s1", "52M de-heavy (96% DE)"),
    ("exp12_swp2b_gpt2sL4_en2p75B_de0p125B_s1", "52M en-heavy (96% EN)"),
]
TEMPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_NEW = 48
BATCH = 64

prompts = [json.loads(l) for l in open(OUT / "prompts.jsonl")]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}, {len(prompts)} prompts, {len(MODELS)} models, {len(TEMPS)} temps", flush=True)

rows = []
for run, label in MODELS:
    finals = glob.glob(str(E12 / "runs" / run / "models_packed/*/final"))
    if not finals:
        print(f"!! no final model for {run} -- skipped", flush=True)
        continue
    tok = AutoTokenizer.from_pretrained(finals[0])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(finals[0]).to(device).eval()
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
                                     k_prompt_switches=p["k_prompt_switches"], temp=temp, seed=seed,
                                     prompt=p["prompt"], completion=c))
        print(f"{label}: T={temp} done ({len(rows)} total)", flush=True)
    del model
    torch.cuda.empty_cache()

pd.DataFrame(rows).to_csv(OUT / "temp_sweep_generations.csv", index=False)
print(f"wrote {OUT/'temp_sweep_generations.csv'} ({len(rows)} rows)", flush=True)
