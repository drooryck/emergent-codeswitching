#!/usr/bin/env python
"""Lexical D/E/U word tagger for German–English, using TongueSwitcher wordlists as
data (not executing their framework). CPU-only, no neural POS.

Decision order mirrors TongueSwitcher's word_processor (lexical subset):
  1) English-only word (en - de), len>2, or two-letter-EN  -> E
  2) interlingual homograph (POS-dependent in TS) -> ambiguous 'H' (resolved by --homograph-default)
  3) German word (de/big_german), len>2, or two-letter-DE -> D
  4) English word, len>2, or two-letter-EN -> E
  5) contains umlaut/ß -> D
  6) else -> U (unknown/other; punctuation, numbers, names)

NOTE (honest caveat): this does NOT exactly reproduce TongueSwitcher's gold tags — their full
system adds HanTa morphology, compound splitting, bigram smoothing, and flair-POS homograph
resolution. We report our tagger's measured agreement with TS gold below; TS gold remains
authoritative where available.
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path

RES = Path(os.environ.get("TONGUESWITCHER_ROOT", ""))
WL = RES / "resources" / "wordlists_as_text_files"
WD = RES / "data" / "wordlist_data"
HOMO = RES / "data" / "interlingual_homographs" / "il_homograph_pos_tags.json"
CLEAN = re.compile(r"[^A-Za-zÄÖÜäöüß'’.\-]")


def _load_lines(p):
    return {w.strip().lower() for w in open(p, encoding="utf-8") if w.strip()}


def _wortschatz_words(p):
    """Leipzig freq list: rank<TAB>word<TAB>freq -> alpha words, lowercased."""
    out = set()
    if not Path(p).exists():
        return out
    for line in open(p, encoding="utf-8", errors="ignore"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[1] and parts[1].isalpha():
            out.add(parts[1].lower())
    return out


def _dictcc(p):
    """dict.cc DE<TAB>EN dump (skip # comments). Returns (de_words, en_words), single alpha tokens."""
    de, en = set(), set()
    if not Path(p).exists():
        return de, en
    for line in open(p, encoding="utf-8", errors="ignore"):
        if line.startswith("#") or "\t" not in line:
            continue
        cols = line.split("\t")
        for tok, dst in ((cols[0], de), (cols[1] if len(cols) > 1 else "", en)):
            tok = tok.strip()
            if tok and " " not in tok and tok.isalpha():
                dst.add(tok.lower())
    return de, en


class LexTagger:
    def __init__(self, homograph_default="context"):
        # Pure curated wordlists only. Adding the Leipzig-news / dict.cc lists was tested and
        # REGRESSED E-precision (those German lists contain many English tokens, so English words
        # matched German first): E->D rose 0.2%->4.8%. We prioritise clean E/D separation; misses
        # go to U ("abstain"), which is safe for switch-span counting (U never creates a switch).
        self.en = _load_lines(WL / "pure_english.txt")
        self.de = _load_lines(WL / "pure_german.txt") | _load_lines(WL / "big_german.txt")
        self.cs = self.en - self.de                      # English-only ("code-switch") words
        self.en12 = _load_lines(WD / "two_letter_english_words.txt")
        self.de12 = _load_lines(WD / "two_letter_german_words.txt")
        homo = json.load(open(HOMO))
        # per-homograph majority language (used when POS unavailable)
        self.homo_major = {}
        for w, posmap in homo.items():
            lans = list(posmap.values())
            self.homo_major[w] = max(set(lans), key=lans.count)
        self.homographs = set(homo.keys())
        self.homograph_default = homograph_default

    def tag_word(self, raw, prev_lang=None, split_hint=None):
        w = CLEAN.sub("", raw)
        if not w:
            return "U"
        wl = w.lower()
        if (wl in self.cs and len(w) > 2) or wl in self.en12:
            return "E"
        if wl in self.homographs:
            if self.homograph_default == "context" and split_hint in ("E", "D"):
                return split_hint
            if self.homograph_default == "prev" and prev_lang in ("E", "D"):
                return prev_lang
            return self.homo_major.get(wl, "D")
        if (wl in self.de and len(w) > 2) or wl in self.de12:
            return "D"
        if (wl in self.en and len(w) > 2) or wl in self.en12:
            return "E"
        if any(c in wl for c in "äöüß"):
            return "D"
        return "U"

    def tag_sentence(self, tokens, split_hint=None):
        out, prev = [], None
        for t in tokens:
            lan = self.tag_word(t, prev, split_hint)
            out.append(lan)
            if lan in ("E", "D"):
                prev = lan
        return out


def _parse_list(v):
    return v if isinstance(v, list) else json.loads(v.replace("'", '"'))


def validate(tagger, gold_path, n):
    import itertools
    tot = agree = 0; conf = {}
    with open(gold_path) as f:
        for line in itertools.islice(f, n):
            r = json.loads(line)
            toks, tags = _parse_list(r["tokens"]), _parse_list(r["tags"])
            if not (toks and tags and len(toks) == len(tags)):
                continue
            pred = tagger.tag_sentence(toks)
            for g, p in zip(tags, pred):
                if g not in ("D", "E"):
                    continue
                tot += 1; agree += (p == g)
                conf[(g, p)] = conf.get((g, p), 0) + 1
    print(f"[validate] sentences≈{n}  lexical gold tokens={tot}  agreement={agree/max(tot,1):.4f}")
    for (g, p), c in sorted(conf.items()):
        print(f"  gold {g} -> pred {p}: {c} ({c/max(tot,1)*100:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate"], default="validate")
    ap.add_argument("--gold", required=True)
    ap.add_argument("-n", type=int, default=1252)
    ap.add_argument("--homograph-default", default="prev", choices=["prev", "context", "major"])
    args = ap.parse_args()
    tagger = LexTagger(homograph_default=args.homograph_default)
    print(f"loaded: en={len(tagger.en):,} de={len(tagger.de):,} cs(en-de)={len(tagger.cs):,} "
          f"homographs={len(tagger.homographs)}")
    validate(tagger, args.gold, args.n)
