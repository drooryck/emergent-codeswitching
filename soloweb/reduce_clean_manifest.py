#!/usr/bin/env python
"""Reduce step: aggregate per-chunk stats, (optionally) concatenate cleaned chunks into per-language
pool files, and write a human-readable MANIFEST.md with retention + token estimates.

Token estimate: we count whitespace words during cleaning (cheap); BPE tokens are estimated with the
joint 30k BPE word→token ratios measured on SoloWeb (en 15.19/10.46=1.452,
de 14.37/8.82=1.629). This confirms whether the ≥30B tokens/language target is met."""
import os, glob, json, argparse, subprocess
BPE_RATIO = {"en": 1.452, "de": 1.629}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=os.environ.get("OUT_ROOT") or None)
    ap.add_argument("--manifest", default=os.environ.get("MANIFEST") or None)
    ap.add_argument("--concat", action="store_true", help="cat chunks into <out_root>/fineweb_clean_{eng,deu}_train.jsonl")
    a = ap.parse_args()
    if not a.out_root or not a.manifest:
        raise SystemExit("Set --out-root and --manifest (or OUT_ROOT and MANIFEST)")

    expected = {}
    for line in open(a.manifest):
        idx, lang, fmt, f, s, e, out, stats = line.rstrip("\n").split("\t")
        expected[stats] = (lang, out)

    agg = {}; missing = []; done_chunks = {"en": [], "de": []}
    for stats, (lang, out) in expected.items():
        if not os.path.exists(stats):
            missing.append(stats); continue
        d = json.load(open(stats))
        if not d.get("done"): missing.append(stats); continue
        A = agg.setdefault(lang, dict(tasks=0, docs_in=0, docs_kept=0, dropped_offlang=0, empty=0,
                                      words_in=0, words_kept=0, chars_in=0, chars_kept=0,
                                      sents=0, sents_scored=0))
        A["tasks"] += 1
        for k in ("docs_in", "docs_kept", "dropped_offlang", "empty", "words_in", "words_kept",
                  "chars_in", "chars_kept", "sents", "sents_scored"):
            A[k] += d.get(k, 0)
        done_chunks[lang].append(out)

    lines = ["# FineWeb LID-decontamination — result manifest", "",
             f"Params from chunks (strict: drop doc if >=1 off-language sentence). Missing/undone tasks: "
             f"**{len(missing)}** of {len(expected)}.", ""]
    lines += ["| lang | tasks | docs in | docs kept | retention | off-lang dropped | words kept | ~BPE tokens (est) | >=30B? |",
              "|---|---|---|---|---|---|---|---|---|"]
    for lang in ("en", "de"):
        if lang not in agg: continue
        A = agg[lang]
        ret = 100 * A["docs_kept"] / max(A["docs_in"], 1)
        drop = 100 * A["dropped_offlang"] / max(A["docs_in"], 1)
        bpe = A["words_kept"] * BPE_RATIO[lang]
        lines.append(f"| {lang} | {A['tasks']} | {A['docs_in']:,} | {A['docs_kept']:,} | {ret:.2f}% | "
                     f"{drop:.2f}% | {A['words_kept']:,} | {bpe/1e9:.2f}B | "
                     f"{'YES' if bpe >= 30e9 else 'NO ('+format(bpe/1e9,'.1f')+'B)'} |")
    lines += ["", "_BPE estimate = words_kept x sw_union ratio (en 1.452, de 1.629); exact count comes at "
              "tokenization time._"]
    md = a.out_root + "/MANIFEST.md"
    open(md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {md}")
    if missing:
        print(f"[incomplete] {len(missing)} tasks not done; rerun the array (idempotent) before concat.")
        for m in missing[:5]: print("   missing:", m)
        return
    if a.concat:
        for lang, tag in (("en", "eng"), ("de", "deu")):
            dst = f"{a.out_root}/fineweb_clean_{tag}_train.jsonl"
            chunks = sorted(done_chunks[lang])
            print(f"concat {len(chunks)} {lang} chunks -> {dst}")
            with open(dst, "w") as w:
                for c in chunks:
                    with open(c) as r:
                        for line in r: w.write(line)
        print("concat done.")

if __name__ == "__main__":
    main()
