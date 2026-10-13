#!/usr/bin/env python
"""Worker: sentence-level LID decontamination of ONE chunk of a FineWeb/MonoWeb shard.

Reads docs [START_RG, END_RG) of a parquet shard (or a whole jsonl file), segments each document with
NLTK punkt (language-specific, the MonoWeb paper's segmenter), runs fastText lid.176 per sentence, and
DROPS any document that contains >=1 off-language sentence (top-1 LID != target lang at prob > CONF,
sentence >= MINCHARS alnum chars) — the strict filter. Kept docs are written as JSONL with provenance
(id, lang, text, url, dump[, date]) so a later anti-join on `id` can exclude them from the codeswitch set.

Idempotent + atomic: writes OUT.tmp / STATS.tmp then os.replace; if STATS exists and is marked done, exits.
Invoked per SLURM array task with all params as argv (see clean_fineweb_lid.sbatch). Single core by design;
parallelism comes from the array (one task per chunk).
"""
import os, sys, re, json, time, argparse
import pyarrow.parquet as pq

def alnum(s): return sum(ch.isalnum() for ch in s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["de", "en"])
    ap.add_argument("--fmt", required=True, choices=["parquet", "jsonl"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--start", type=int, default=0)   # row-group start (parquet) / line start (jsonl)
    ap.add_argument("--end", type=int, default=-1)     # exclusive; -1 = to end
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--minchars", type=int, default=int(os.environ.get("MINCHARS", "25")))
    ap.add_argument("--conf", type=float, default=float(os.environ.get("CONF", "0.50")))
    ap.add_argument("--nltk-data", default=os.environ.get("NLTK_DATA") or None)
    ap.add_argument("--lid-model", default=os.environ.get("LID_MODEL") or None)
    a = ap.parse_args()

    # idempotent skip
    if os.path.exists(a.stats) and os.path.exists(a.out):
        try:
            if json.load(open(a.stats)).get("done"):
                print(f"[skip] {a.stats} already done"); return
        except Exception:
            pass

    import nltk; nltk.data.path.insert(0, a.nltk_data)
    from nltk.tokenize import sent_tokenize
    import fasttext
    LID = fasttext.load_model(a.lid_model)
    NLLANG = {"de": "german", "en": "english"}[a.lang]
    TGT = f"__label__{a.lang}"

    def records():
        if a.fmt == "parquet":
            pf = pq.ParquetFile(a.input)
            end = pf.num_row_groups if a.end < 0 else a.end
            cols = [c for c in ("id", "text", "url", "dump", "date") if c in pf.schema_arrow.names]
            for rg in range(a.start, end):
                tb = pf.read_row_group(rg, columns=cols).to_pydict()
                n = len(tb["text"])
                for i in range(n):
                    yield {c: tb[c][i] for c in cols}
        else:
            with open(a.input) as fh:
                for ln, line in enumerate(fh):
                    if ln < a.start: continue
                    if a.end >= 0 and ln >= a.end: break
                    try: r = json.loads(line)
                    except Exception: continue
                    yield {"id": r.get("id") or r.get("doc_id"), "text": r.get("text", ""),
                           "url": r.get("url"), "dump": r.get("dump"), "date": r.get("date")}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(os.path.dirname(a.stats), exist_ok=True)
    st = dict(docs_in=0, docs_kept=0, dropped_offlang=0, empty=0,
              words_in=0, words_kept=0, chars_in=0, chars_kept=0, sents=0, sents_scored=0)
    t0 = time.time()
    with open(a.out + ".tmp", "w") as w:
        for r in records():
            txt = r.get("text") or ""
            st["docs_in"] += 1
            w_ct = len(txt.split()); c_ct = len(txt)
            st["words_in"] += w_ct; st["chars_in"] += c_ct
            if not txt.strip():
                st["empty"] += 1; continue
            off = False
            for s in sent_tokenize(txt, language=NLLANG):
                s = s.replace("\n", " ").strip()
                if not s: continue
                st["sents"] += 1
                if alnum(s) < a.minchars: continue
                labs, probs = LID.predict(s)
                if float(probs[0]) <= a.conf: continue
                st["sents_scored"] += 1
                if labs[0] != TGT:
                    off = True; break
            if off:
                st["dropped_offlang"] += 1; continue
            st["docs_kept"] += 1; st["words_kept"] += w_ct; st["chars_kept"] += c_ct
            w.write(json.dumps({"id": r.get("id"), "lang": a.lang, "text": txt,
                                "url": r.get("url"), "dump": r.get("dump"),
                                **({"date": str(r["date"])} if r.get("date") is not None else {})},
                               ensure_ascii=False) + "\n")
    os.replace(a.out + ".tmp", a.out)
    st.update(done=True, lang=a.lang, input=a.input, start=a.start, end=a.end,
              minchars=a.minchars, conf=a.conf, elapsed_s=round(time.time() - t0, 1))
    with open(a.stats + ".tmp", "w") as f:
        json.dump(st, f)
    os.replace(a.stats + ".tmp", a.stats)
    print(f"[done] {a.input}[{a.start}:{a.end}] kept {st['docs_kept']}/{st['docs_in']} "
          f"({100*st['docs_kept']/max(st['docs_in'],1):.1f}%) in {st['elapsed_s']}s")

if __name__ == "__main__":
    main()
