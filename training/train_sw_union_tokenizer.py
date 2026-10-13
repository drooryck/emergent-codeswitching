#!/usr/bin/env python
"""Train the joint 30k byte-level BPE over SoloWeb-EN, SoloWeb-DE, leak-free
TongueSwitcher, and Denglisch. SoloWeb sources skip the first 6,000 docs (held-out).
Special tokens: <s> <pad> </s> <unk> <mask> (ids 0-4). The BPE is trained with
add_prefix_space=False; the saved Hugging Face wrapper pins True.
"""
import argparse
import json
import os
import shutil
import tempfile

from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast


def _stream(path, outp, skip, char_budget):
    n_lines = n_chars = 0
    with open(path) as r, open(outp, "w") as w:
        for i, line in enumerate(r):
            if i < skip:
                continue
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if not t:
                continue
            w.write(t.replace("\n", " ") + "\n")
            n_lines += 1
            n_chars += len(t)
            if n_chars >= char_budget:
                break
    return n_lines, n_chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--soloweb-eng", required=True)
    ap.add_argument("--soloweb-deu", required=True)
    ap.add_argument("--tongueswitcher", required=True)
    ap.add_argument("--denglisch", required=True)
    ap.add_argument("--char-budget", type=int, default=400_000_000)
    ap.add_argument("--vocab-size", type=int, default=30000)
    ap.add_argument("--skip-docs", type=int, default=6000)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    prov = {"vocab_size": a.vocab_size, "char_budget_per_source": a.char_budget, "sources": {}}
    txt_files = []
    tmpdir = tempfile.mkdtemp(prefix="swtok_", dir=a.out_dir)
    sources = [
        ("soloweb_eng", a.soloweb_eng, a.skip_docs),
        ("soloweb_deu", a.soloweb_deu, a.skip_docs),
        ("tongueswitcher", a.tongueswitcher, 0),
        ("denglisch", a.denglisch, 0),
    ]
    for name, path, skip in sources:
        outp = os.path.join(tmpdir, f"{name}.txt")
        n_lines, n_chars = _stream(path, outp, skip, a.char_budget)
        prov["sources"][name] = {"lines": n_lines, "chars": n_chars}
        txt_files.append(outp)
        print(f"{name}: {n_lines:,} lines, {n_chars:,} chars", flush=True)

    tok = ByteLevelBPETokenizer(add_prefix_space=False)
    tok.train(files=txt_files, vocab_size=a.vocab_size, min_frequency=2,
              special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tok.save_model(a.out_dir)

    hf = GPT2TokenizerFast(
        vocab_file=f"{a.out_dir}/vocab.json", merges_file=f"{a.out_dir}/merges.txt",
        bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        pad_token="<pad>", mask_token="<mask>", add_prefix_space=True,
    )
    hf.save_pretrained(a.out_dir)
    with open(f"{a.out_dir}/train_sources.json", "w") as f:
        json.dump(prov, f, indent=1)
    shutil.rmtree(tmpdir)
    print(f"saved tokenizer to {a.out_dir} (vocab {hf.vocab_size})")
    s = "Ich dachte, Tiktok is my place, aber naja."
    print("sanity:", hf.decode(hf(s)["input_ids"]) == s, "|", len(hf(s)["input_ids"]), "tokens")


if __name__ == "__main__":
    main()
