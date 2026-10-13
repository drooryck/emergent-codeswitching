#!/usr/bin/env python
"""Build the temperature-sweep prompt sets (sept24).

TS sets: from the GOLD TS held-out (leak-free vs training pools), for k in {1,2,3,4}: up to 100
tweets whose gold tags contain exactly k D<->E switch points inside a 10-32-token prompt prefix;
the prefix is cut >=2 lexical words after the k-th switch so the switch context is established,
and the tweet must continue past the cut. Detokenized with simple no-space-before-punct rules.
Made-up set: the 5 unique clause-switching prompts from qualitative_generations_soloweb_clauseswitch.csv.
Output: prompts.jsonl with {set, prompt_id, prompt, k_prompt_switches, source_idx}.
"""
import json
import random
from pathlib import Path

import pandas as pd

E12 = Path(__import__("os").environ["CS_ROOT"])
OUT = E12 / "generation_out"
OUT.mkdir(parents=True, exist_ok=True)
GOLD = E12 / "data" / "ts_heldout" / "valid_tongueswitcher_cs.jsonl"
NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", ")", "]", "}", "%", "…", "'s", "n't", "'"}
NO_SPACE_AFTER = {"(", "[", "{", "#", "@"}


def detok(tokens):
    out = ""
    for i, t in enumerate(tokens):
        if i == 0 or t in NO_SPACE_BEFORE or (out and out[-1] in NO_SPACE_AFTER):
            out += t
        else:
            out += " " + t
    return out


def switch_positions(tags):
    pos, prev = [], None
    for i, t in enumerate(tags):
        if t in ("D", "E"):
            if prev is not None and t != prev:
                pos.append(i)
            prev = t
    return pos


rng = random.Random(0)
tweets = [json.loads(l) for l in open(GOLD)]
rng.shuffle(tweets)
rows = []
for k in (1, 2, 3, 4):
    picked = 0
    for ti, tw in enumerate(tweets):
        if picked >= 100:
            break
        toks, tags = tw["tokens"], tw["tags"]
        sw = switch_positions(tags)
        if len(sw) < k:
            continue
        kth = sw[k - 1]
        # cut >=2 lexical words after the k-th switch, before the (k+1)-th switch
        lex_after, cut = 0, None
        limit = sw[k] if len(sw) > k else len(toks)
        for j in range(kth, limit):
            if tags[j] in ("D", "E"):
                lex_after += 1
            if lex_after >= 2 and j + 1 < len(toks):
                cut = j + 1
                break
        if cut is None or not (10 <= cut <= 32) or len(toks) - cut < 4:
            continue
        if switch_positions(tags[:cut]) != sw[:k]:
            continue
        rows.append(dict(set=f"ts{k}sw", prompt_id=picked, prompt=detok(toks[:cut]),
                         k_prompt_switches=k, source_idx=ti))
        picked += 1
    print(f"ts{k}sw: {picked} prompts")

clause = Path(__import__("os").environ.get(
    "CLAUSESWITCH_CSV", str(E12 / "data" / "qualitative_generations_soloweb_clauseswitch.csv")))
if clause.exists():
    cs = pd.read_csv(clause)
    for i, p in enumerate(sorted(cs.prompt.unique())):
        rows.append(dict(set="clauseswitch_madeup", prompt_id=i, prompt=p,
                         k_prompt_switches=None, source_idx=-1))
    print(f"clauseswitch_madeup: {cs.prompt.nunique()} prompts")
else:
    print(f"skip clauseswitch_madeup (missing {clause})")

with open(OUT / "prompts.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {OUT/'prompts.jsonl'} ({len(rows)} prompts)")
