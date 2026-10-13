#!/usr/bin/env python
"""Build the SoloWeb 7x7 nested train pools with EXACT token accounting (aug18: no more
chars-per-token approximation — every rung boundary is an exact tokenize-and-count cut).

Single streaming pass per language over fineweb_clean_{eng,deu}_train.jsonl (skipping the first
--skip-docs reserved as heldout): batch-tokenize, tee each doc into every rung file whose per-language
target is not yet reached. Rungs are nested prefixes by construction. Token accounting matches the
packer: each document costs len(BPE(text)) + 1 (the packer appends one end-of-sequence token per doc).
The document that CROSSES a rung's target is TRUNCATED at the exact token boundary (token->character
offset mapping, iterated to convergence), so each rung's written token count equals its target to
within ~1 token. Actual counts recorded in soloweb_pool_summary.json.

Writes: <out-root>/<rung-name>/data/splits/sw/train_{eng,deu}.jsonl  for each rung."""
import json, os, argparse, time
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--clean-dir", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out-root", required=True, help="dir that holds one pool-run dir per rung")
ap.add_argument("--rungs", nargs="+", required=True, help="name:tokens_per_language, e.g. D0p5B:250000000")
ap.add_argument("--skip-docs", type=int, default=6000)
ap.add_argument("--batch", type=int, default=512)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer)
rungs = []
for r in a.rungs:
    name, t = r.split(":"); rungs.append((name, int(t)))
rungs.sort(key=lambda x: x[1])          # smallest first; a doc goes to every unfinished rung
summary = {"skip_docs": a.skip_docs, "tokenizer": a.tokenizer, "rungs": {}}

for lang in ("eng", "deu"):
    src = f"{a.clean_dir}/fineweb_clean_{lang}_train.jsonl"
    outs, counts, docs, open_rungs = {}, {}, {}, []
    for name, tgt in rungs:
        d = os.path.join(a.out_root, name, "data", "splits", "sw")
        os.makedirs(d, exist_ok=True)
        outs[name] = open(os.path.join(d, f"train_{lang}.jsonl"), "w")
        counts[name] = 0; docs[name] = 0; open_rungs.append(name)
    t0 = time.time(); batch_lines, batch_texts = [], []

    RTGT = dict(rungs)

    def truncate_to(text, r):
        """Return (truncated_text, token_count<=r) cutting at the character offset of token r."""
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        while len(enc["input_ids"]) > r:
            cut = enc["offset_mapping"][r - 1][1]
            text = text[:cut]
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return text, len(enc["input_ids"])

    def flush():
        global open_rungs
        if not batch_texts: return
        encs = tok(batch_texts, add_special_tokens=False)["input_ids"]
        for line, txt, e in zip(batch_lines, batch_texts, encs):
            n = len(e) + 1                       # +1 = the packer's per-doc end-of-sequence token
            done = []
            for name in open_rungs:
                remaining = RTGT[name] - counts[name]
                if n < remaining:                # whole doc fits with room to spare
                    outs[name].write(line)
                    counts[name] += n; docs[name] += 1
                else:                            # this doc crosses the target: truncate to land exactly
                    r = max(remaining - 1, 1)    # tokens of text to keep (1 reserved for EOS)
                    ttxt, tcnt = truncate_to(txt, r)
                    outs[name].write(json.dumps({"text": ttxt}, ensure_ascii=False) + "\n")
                    counts[name] += tcnt + 1; docs[name] += 1
                    done.append(name)
            for name in done:
                outs[name].close()
                print(f"  [{lang}] rung {name} closed: {counts[name]:,} tokens "
                      f"({docs[name]:,} docs, {counts[name]-RTGT[name]:+,} vs target) "
                      f"{time.time()-t0:.0f}s", flush=True)
                open_rungs.remove(name)
        batch_lines.clear(); batch_texts.clear()

    with open(src) as fh:
        for i, line in enumerate(fh):
            if i < a.skip_docs: continue
            if not open_rungs: break
            try: text = json.loads(line)["text"]
            except Exception: continue
            batch_lines.append(line); batch_texts.append(text)
            if len(batch_texts) >= a.batch: flush()
        flush()
    for name in open_rungs:   # source exhausted before target (shouldn't happen)
        outs[name].close()
        print(f"  [{lang}] WARNING rung {name} EXHAUSTED at {counts[name]:,} tokens", flush=True)
    summary["rungs"].setdefault(lang, {})
    for name, tgt in rungs:
        summary["rungs"][lang][name] = {"target": tgt, "actual": counts[name], "docs": docs[name]}

with open(os.path.join(a.out_root, "soloweb_pool_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("wrote", os.path.join(a.out_root, "soloweb_pool_summary.json"))
