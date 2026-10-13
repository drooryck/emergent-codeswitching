#!/usr/bin/env python
"""NLTK punkt + fastText LID audit on MonoWeb (same gates as clean_fineweb_lid.py).

Samples N docs per language from monoweb parquet shards; reports sentence-level contamination
and doc-level top-LID (single fastText predict on full document text).
"""
import os, json, glob, argparse
import numpy as np
import pyarrow.parquet as pq
import fasttext

MINCHARS, CONF = 25, 0.50


def alnum(s):
    return sum(ch.isalnum() for ch in s)


def sample_docs(mono_root, pool, n, seed):
    files = sorted(glob.glob(f"{mono_root}/{pool}/train-*.parquet"))
    if not files:
        raise SystemExit(f"no shards under {mono_root}/{pool}")
    rng = np.random.default_rng(seed)
    per = n // len(files) + 1
    texts = []
    for f in files:
        df = pq.ParquetFile(f).read_row_group(0, columns=["text"]).to_pandas()
        take = min(per, len(df))
        idx = rng.choice(len(df), size=take, replace=False)
        texts.extend(df["text"].iloc[idx].tolist())
    rng.shuffle(texts)
    return texts[:n]


def audit_texts(texts, lang, lid, sent_tokenize, nllang):
    tgt = f"__label__{lang}"
    pair_lang = "en" if lang == "de" else "de"
    pair = f"__label__{pair_lang}"
    n_docs = len(texts)
    any_off = pair_off = top_not_tgt_whole = lw_not_tgt = 0
    frac_offs = []
    n_sents_scored = n_sents_off = 0
    breakdown_whole = {}
    breakdown_lw = {}

    for txt in texts:
        txt = txt or ""
        flat = txt.replace("\n", " ").strip()
        if flat:
            lab, prob = lid.predict(flat)
            if float(prob[0]) > CONF and lab[0] != tgt:
                top_not_tgt_whole += 1
                wl = lab[0].replace("__label__", "")
                breakdown_whole[wl] = breakdown_whole.get(wl, 0) + 1

        lang_mass = {}
        n_sent = n_off = n_pair = 0
        for s in sent_tokenize(txt, language=nllang):
            s = s.replace("\n", " ").strip()
            a = alnum(s)
            if a < MINCHARS:
                continue
            labs, probs = lid.predict(s)
            if float(probs[0]) <= CONF:
                continue
            lang_mass[labs[0]] = lang_mass.get(labs[0], 0) + a
            n_sent += 1
            if labs[0] != tgt:
                n_off += 1
                if labs[0] == pair:
                    n_pair += 1
        if lang_mass:
            top_lw = max(lang_mass, key=lang_mass.get)
            if top_lw != tgt:
                lw_not_tgt += 1
                wl = top_lw.replace("__label__", "")
                breakdown_lw[wl] = breakdown_lw.get(wl, 0) + 1
        n_sents_scored += n_sent
        n_sents_off += n_off
        if n_sent:
            frac_offs.append(n_off / n_sent)
        if n_off >= 1:
            any_off += 1
        if n_pair >= 1:
            pair_off += 1

    return dict(
        n_docs=n_docs,
        pct_docs_any_off=100 * any_off / n_docs,
        pct_docs_pair_off=100 * pair_off / n_docs,
        pct_docs_top_lid_whole_not_tgt=100 * top_not_tgt_whole / n_docs,
        top_lid_whole_breakdown=breakdown_whole,
        pct_docs_top_lid_length_weighted_not_tgt=100 * lw_not_tgt / n_docs,
        top_lid_lw_breakdown=breakdown_lw,
        pct_sents_off=100 * n_sents_off / max(n_sents_scored, 1),
        mean_frac_off_sent=100 * np.mean(frac_offs) if frac_offs else 0.0,
        pct_docs_kept_strict=100 * (n_docs - any_off) / n_docs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--mono-root", required=True, help="Directory with deu/ and eng/ parquet shards")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--nltk-data", required=True)
    ap.add_argument("--lid-model", required=True)
    a = ap.parse_args()

    import nltk

    nltk.data.path.insert(0, a.nltk_data)
    from nltk.tokenize import sent_tokenize

    lid = fasttext.load_model(a.lid_model)
    results = {}
    for pool, lang, nllang in [("deu", "de", "german"), ("eng", "en", "english")]:
        texts = sample_docs(a.mono_root, pool, a.n, a.seed + (1 if lang == "en" else 0))
        results[pool] = audit_texts(texts, lang, lid, sent_tokenize, nllang)
        results[pool]["pool"] = f"MonoWeb-{pool}"
        results[pool]["target_lang"] = lang
        results[pool]["n_sampled"] = len(texts)

    meta = dict(
        method="NLTK punkt + fastText lid.176",
        minchars=MINCHARS,
        conf=CONF,
        seed=a.seed,
        n_per_pool=a.n,
        matches="clean_fineweb_lid.py gates",
    )
    out = {"meta": meta, "pools": results}
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
