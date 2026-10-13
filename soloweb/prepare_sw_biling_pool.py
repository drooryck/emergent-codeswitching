#!/usr/bin/env python
"""Generic SoloWeb pool builder with EXACT token accounting (sept8): per-language targets + optional
leak-free TongueSwitcher prefix. Serves both new sweeps:
  - swp2b: independent D_en x D_de grid on SoloWeb (tokens-cs 0)
  - swdose: composition-compensated dose runs (eng/deu reduced by the counted language mass inside
    the TS slice; see chunk-index computation in the launcher)
Sources: fineweb_clean_{eng,deu}_train.jsonl (skipping the 6,000 reserved heldout docs — SAME offset
as the sw7 pools) and tongueswitcher_train_leakfree.jsonl (contiguous prefix, matching the chunk-index
order). Every file is cut at the exact token boundary (truncate-last-doc, EOS-per-doc accounting),
tokenizer = sw_union unless overridden. Writes <run>/data/splits/<cond>/train_{eng,deu[,cs]}.jsonl."""
import json, os, argparse, time
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--condition", required=True)
ap.add_argument("--tokens-eng", type=int, required=True)
ap.add_argument("--tokens-deu", type=int, required=True)
ap.add_argument("--tokens-cs", type=int, default=0)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--clean-dir", required=True)
ap.add_argument("--tongueswitcher", default="")
ap.add_argument("--skip-docs", type=int, default=6000)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer)
outd = os.path.join(a.run_dir, "data", "splits", a.condition)
os.makedirs(outd, exist_ok=True)

def truncate_to(text, r):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    while len(enc["input_ids"]) > r:
        text = text[:enc["offset_mapping"][r - 1][1]]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return text, len(enc["input_ids"])

def build(src, outp, tgt, skip):
    if tgt <= 0:
        return {"target": tgt, "actual": 0, "docs": 0}
    n_tok = n_docs = 0; t0 = time.time(); done = False
    batch_lines, batch_texts = [], []
    with open(src) as r, open(outp + ".tmp", "w") as w:
        def flush():
            nonlocal n_tok, n_docs, done
            if not batch_texts: return
            encs = tok(batch_texts, add_special_tokens=False)["input_ids"]
            for line, txt, e in zip(batch_lines, batch_texts, encs):
                if done: break
                n = len(e) + 1                      # +1 = packer EOS per doc
                if n_tok + n < tgt:
                    w.write(line); n_tok += n; n_docs += 1
                else:
                    rr = max(tgt - n_tok - 1, 1)
                    ttxt, tcnt = truncate_to(txt, rr)
                    w.write(json.dumps({"text": ttxt}, ensure_ascii=False) + "\n")
                    n_tok += tcnt + 1; n_docs += 1; done = True
            batch_lines.clear(); batch_texts.clear()
        for i, line in enumerate(r):
            if i < skip: continue
            if done: break
            try: t = json.loads(line)["text"]
            except Exception: continue
            if not t: continue
            batch_lines.append(line); batch_texts.append(t)
            if len(batch_texts) >= 512: flush()
        flush()
    os.replace(outp + ".tmp", outp)
    print(f"  {os.path.basename(outp)}: {n_tok:,}/{tgt:,} tokens ({n_docs:,} docs, "
          f"{n_tok-tgt:+,}) {time.time()-t0:.0f}s", flush=True)
    return {"target": tgt, "actual": n_tok, "docs": n_docs}

summary = {"tokenizer": a.tokenizer, "condition": a.condition}
summary["eng"] = build(os.path.join(a.clean_dir, "fineweb_clean_eng_train.jsonl"),
                       os.path.join(outd, "train_eng.jsonl"), a.tokens_eng, a.skip_docs)
summary["deu"] = build(os.path.join(a.clean_dir, "fineweb_clean_deu_train.jsonl"),
                       os.path.join(outd, "train_deu.jsonl"), a.tokens_deu, a.skip_docs)
if a.tokens_cs > 0:
    if not a.tongueswitcher:
        raise SystemExit("--tongueswitcher is required when --tokens-cs > 0")
    summary["cs"] = build(a.tongueswitcher, os.path.join(outd, "train_cs.jsonl"), a.tokens_cs, 0)
with open(os.path.join(a.run_dir, "data", "pool_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("done")
