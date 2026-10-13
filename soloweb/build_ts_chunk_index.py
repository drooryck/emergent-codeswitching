#!/usr/bin/env python
"""Build the TongueSwitcher prefix/chunk index (aug18 ask 3): contiguous chunks of the leak-free TS
train pool with EXACT counted quantities, so any D_cs budget maps to switch/language token counts by
lookup (cumsum over chunks) instead of the corpus-average SPAN_RATIO approximation.

Per contiguous chunk of CHUNK docs (default 500k), records:
  docs, words, bpe_fulltext   (exact sw_union BPE of the raw texts, batch-tokenized — the packed count)
  bpe_words                   (sum of per-word BPE with GPT2 leading-space convention — attribution base)
  D_words/E_words/U_words     (lexical tagger; U = abstain, never creates a switch)
  D_bpe/E_bpe/U_bpe           (per-word BPE attributed by tag — the language-balance numbers)
  sw_D2E/sw_E2D               (lexical switch transitions = switch-POINT counts per direction)
  span_D2E_words/_bpe, span_E2D_words/_bpe  (full run after each switch, words and BPE)
Calibration to the TongueSwitcher tagger happens at read time: corpus-level truth lives in
the tag-count aggregate JSON; per-quantity scale = AGGREGATE_total / index_total. Chunk-to-chunk
composition variance is tiny (switches/doc 2.85±1.94 -> ~0.2% rel. SD per 100k docs), so calibrated
chunk counts are accurate to <<1%.
One SLURM array task per chunk; idempotent (skips existing chunk JSON). Reduce = cat the JSONs."""
import os, sys, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))
from lexical_tagger import LexTagger
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--chunk-docs", type=int, default=500_000)
    ap.add_argument("--chunk-id", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    out = os.path.join(a.out_dir, f"chunk_{a.chunk_id:04d}.json")
    if os.path.exists(out):
        print(f"[skip] {out}"); return
    os.makedirs(a.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tagger = LexTagger()
    lo, hi = a.chunk_id * a.chunk_docs, (a.chunk_id + 1) * a.chunk_docs

    st = dict(chunk_id=a.chunk_id, doc_lo=lo, doc_hi_excl=hi, docs=0, words=0,
              bpe_fulltext=0, bpe_words=0,
              D_words=0, E_words=0, U_words=0, D_bpe=0, E_bpe=0, U_bpe=0,
              sw_D2E=0, sw_E2D=0,
              span_D2E_words=0, span_D2E_bpe=0, span_E2D_words=0, span_E2D_bpe=0)
    t0 = time.time(); texts_batch = []

    def flush_fulltext(batch):
        if not batch: return 0
        enc = tok(batch, add_special_tokens=False)["input_ids"]
        return sum(len(e) for e in enc)

    with open(a.input) as fh:
        for i, line in enumerate(fh):
            if i < lo: continue
            if i >= hi: break
            text = json.loads(line).get("text", "")
            if not text: continue
            st["docs"] += 1
            texts_batch.append(text)
            if len(texts_batch) >= 512:
                st["bpe_fulltext"] += flush_fulltext(texts_batch); texts_batch = []
            words = text.split()
            st["words"] += len(words)
            # per-word BPE with GPT2 leading-space convention (approximates in-context tokenization)
            encs = tok([" " + w for w in words], add_special_tokens=False)["input_ids"]
            lens = [len(e) for e in encs]
            st["bpe_words"] += sum(lens)
            prev = None          # previous LEXICAL language (D/E); U words never create switches
            run_dir = None       # direction of the current switched-into run, e.g. "D2E"
            for w, L in zip(words, lens):
                t = tagger.tag_word(w, prev_lang=prev)
                if t == "D":
                    st["D_words"] += 1; st["D_bpe"] += L
                elif t == "E":
                    st["E_words"] += 1; st["E_bpe"] += L
                else:
                    st["U_words"] += 1; st["U_bpe"] += L
                if t in ("D", "E"):
                    if prev is not None and t != prev:
                        run_dir = f"{prev}2{t}"
                        st[f"sw_{run_dir}"] += 1
                    if run_dir is not None and t == run_dir[-1]:
                        st[f"span_{run_dir}_words"] += 1; st[f"span_{run_dir}_bpe"] += L
                    elif run_dir is not None and t != run_dir[-1]:
                        run_dir = None
                    prev = t
    st["bpe_fulltext"] += flush_fulltext(texts_batch)
    st["elapsed_s"] = round(time.time() - t0, 1)
    with open(out + ".tmp", "w") as f: json.dump(st, f)
    os.replace(out + ".tmp", out)
    print(f"[done] chunk {a.chunk_id}: {st['docs']:,} docs, {st['bpe_fulltext']/1e6:.1f}M BPE, "
          f"D/E words {st['D_words']/1e6:.1f}M/{st['E_words']/1e6:.1f}M, "
          f"switches {st['sw_D2E']+st['sw_E2D']:,} in {st['elapsed_s']}s")

if __name__ == "__main__":
    main()
