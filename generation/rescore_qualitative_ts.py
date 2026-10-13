#!/usr/bin/env python
"""Rescore ALL qualitative-generation CSVs with the real TongueSwitcher pipeline (sept24).

Pipeline = external/TongueSwitcher (igorsterner/TongueSwitcher) via ts_annotate.build_tagger:
flair upos-multi POS -> dictionaries + HanTa morphology + interlingual-homograph POS resolution
+ bigram smoothing (tongueswitcher_detect). Validated against the gold TS held-out (1,252
tweets, 32,518 lexical tokens): token agreement 97.2%, D F1 .985, E P/R .79/.94, mean
|pred - gold| switch count 0.76/tweet. An NE-guard variant (pred-E PROPN -> D) was tested and
REJECTED (agreement 96.4%, E recall .78, switch error 0.93).

Switch counting (sept24 variant test on TS gold + 2,000 Denglisch-manual gold sentences):
  PRIMARY (V2 "alpha-only"): only tokens containing an alphabetic char enter the switch
    pattern — smoothing paints punctuation/digits with a language, and counting them fabricates
    switches (the "Los Angeles, CA" island). |Dsw|/sent: TS 0.748 (raw 0.750), Denglisch 0.588
    (raw 0.596). Never worse than raw.
  SECONDARY (V3 "nopropn"): additionally excludes PROPN tokens from the pattern. Better in
    English-matrix/balanced text (Denglisch 0.486) but worse on German-matrix tweets (TS 0.777,
    flair PROPN-mislabels real switch words) — provided as extra columns, not the default.
Columns written in place (originals backed up to *.csv.bak):
  ts_tokens ts_tags ts_pattern ts_switches ts_mixed ts_n_D ts_n_E ts_n_M ts_n_U
  ts_switch_rate_100w = 100 * switches / lexical(D+E, alphabetic) words
  ts_switches_nopropn ts_switch_rate_100w_nopropn  (V3)
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from ts_annotate import _setup_imports, build_tagger

AUG = Path(__import__("os").environ.get("CS_ROOT", ".")) / "plots"
CSVS = ["qualitative_generations.csv", "qualitative_generations_soloweb.csv",
        "qualitative_generations_soloweb_clauseswitch.csv",
        "qualitative_generations_soloweb_clauseswitch_tagged.csv",
        "qualitative_generations_soloweb_highswitch.csv",
        "qualitative_generations_soloweb_multiswitch.csv"]


def _runs(seq):
    runs = []
    for t in seq:
        if not runs or runs[-1] != t:
            runs.append(t)
    return runs


def _has_alpha(t):
    return any(c.isalpha() for c in t)


def summarize(tags, tokens=None, poses=None):
    tokens = tokens if tokens is not None else [""] * len(tags)
    poses = poses if poses is not None else ["X"] * len(tags)
    # V2 primary: alphabetic tokens only (smoothing paints punctuation with a language)
    lex = [t for t, tok in zip(tags, tokens) if t in ("D", "E") and _has_alpha(tok)]
    runs = _runs(lex)
    sw = max(0, len(runs) - 1)
    # V3 secondary: additionally exclude PROPN tokens
    lex3 = [t for t, tok, p in zip(tags, tokens, poses)
            if t in ("D", "E") and _has_alpha(tok) and p != "PROPN"]
    sw3 = max(0, len(_runs(lex3)) - 1)
    # minority-run structure (oct2, standard): majority language of the completion's lexical
    # words, and the longest consecutive run of the OTHER language. run>=2 separates deliberate
    # multi-word switches from single-word loanword/insertion tags (~60-77% of raw "switches").
    nD = sum(1 for t in lex if t == "D")
    maj = "D" if nD >= len(lex) - nD else "E"
    max_run = cur = n_runs = 0
    first_min_pos = -1
    for i, t in enumerate(lex):
        if t != maj:
            if cur == 0:
                n_runs += 1
                if first_min_pos < 0:
                    first_min_pos = i
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return {"ts_pattern": "->".join(runs), "ts_switches": sw,
            "ts_mixed": int("D" in lex and "E" in lex),
            "ts_n_D": tags.count("D"), "ts_n_E": tags.count("E"),
            "ts_n_M": tags.count("M"), "ts_n_U": tags.count("U"),
            "ts_switch_rate_100w": round(100.0 * sw / max(len(lex), 1), 3),
            "ts_switches_nopropn": sw3,
            "ts_switch_rate_100w_nopropn": round(100.0 * sw3 / max(len(lex3), 1), 3),
            "ts_majority": maj, "ts_max_minority_run": max_run,
            "ts_n_minority_runs": n_runs, "ts_first_minority_pos": first_min_pos}


def tag_texts(tagger, texts, batch=64):
    from flair.data import Sentence
    out = []
    for s in range(0, len(texts), batch):
        sents = [Sentence(t if isinstance(t, str) and t.strip() else ".") for t in texts[s:s + batch]]
        tagger.flair_tagger.predict(sents, mini_batch_size=batch, verbose=False)
        for sent in sents:
            words = [{"token": tok.text, "lan": "U", "pos": tok.get_label("upos").value}
                     for tok in sent]
            full = tagger.tongueswitcher_detect(words)
            out.append(([w["token"] for w in full], [w["lan"] for w in full],
                        [w.get("pos", "X") for w in full]))
        print(f"  {min(s + batch, len(texts))}/{len(texts)}", flush=True)
    return out


def main():
    _setup_imports()
    tagger = build_tagger()
    for name in CSVS:
        f = AUG / name
        if not f.exists():
            print(f"missing {f}"); continue
        df = pd.read_csv(f)
        shutil.copy(f, f.with_suffix(".csv.bak"))
        res = tag_texts(tagger, df["completion"].astype(str).tolist())
        df["ts_tokens"] = [json.dumps(t, ensure_ascii=False) for t, _, _ in res]
        df["ts_tags"] = [json.dumps(g, ensure_ascii=False) for _, g, _ in res]
        summ = pd.DataFrame([summarize(g, t, p) for t, g, p in res])
        for c in summ.columns:
            df[c] = summ[c]
        df.to_csv(f, index=False)
        print(f"{name}: {len(df)} rows, mixed={df.ts_mixed.sum()}, "
              f"mean rate={df.ts_switch_rate_100w.mean():.2f}/100w", flush=True)


if __name__ == "__main__":
    main()
