#!/usr/bin/env python
"""Thin driver around the TongueSwitcher rule-based tagger
(https://github.com/igorsterner/TongueSwitcher), used as a read-only library.
Reuses DataLoaderCS.LoadDictionaries + TongueSwitcher(TSExecutor, Processor).

Modes:
  validate : tag gold TS sentences (GOLD tokenization) -> per-token D/E agreement + confusion.
"""
from __future__ import annotations
import sys, types, json, argparse, os, itertools
from pathlib import Path

REPO_SRC = Path(os.environ.get("TONGUESWITCHER_SRC", ""))
CFG = "../configs/tongueswitcher_detection.jsonnet"


def _setup_imports():
    if not REPO_SRC:
        raise SystemExit("Set TONGUESWITCHER_SRC to the TongueSwitcher src/ directory")
    try:
        import _jsonnet  # noqa
    except ImportError:
        import _gojsonnet as _gj
        m = types.ModuleType("_jsonnet"); m.evaluate_file = _gj.evaluate_file; m.evaluate_snippet = _gj.evaluate_snippet
        sys.modules["_jsonnet"] = m
    os.chdir(REPO_SRC)                 # relative ../data resource paths
    sys.path.insert(0, str(REPO_SRC))


def build_tagger(reset=None):
    import _jsonnet
    from easydict import EasyDict
    from data_loader_manager.data_loader_tongueswitcher import DataLoaderCS
    from tongueswitcher.tweet_level_processor import TongueSwitcher
    cfg = EasyDict(json.loads(_jsonnet.evaluate_file(CFG)))
    os.makedirs("../data/cache", exist_ok=True)          # cache save target
    # First run compiles the wordlists (~2 min) and pickles them to ../data/cache/;
    # afterwards reuse the cache (reset=False) so subsequent runs load in ~1s.
    if reset is None:
        reset = not (os.path.isfile("../data/cache/dictionaries.pkl") and
                     os.path.isfile("../data/cache/affixes.pkl"))
    cfg.args = EasyDict({"reset_dictionaries": reset})
    dl = DataLoaderCS(cfg); dl.data = EasyDict()
    dl.LoadDictionaries(cfg.data_loader.dataset_modules.module_dict["LoadDictionaries"])
    return TongueSwitcher(cfg, dl)


def _parse_list(v):
    return v if isinstance(v, list) else json.loads(v.replace("'", '"'))


def tag_tokens_full(tagger, tokens):
    """Return the full annotated word dicts (token, lan, pos, clh?) aligned to `tokens`."""
    from flair.data import Sentence
    sent = Sentence(list(tokens), use_tokenizer=False)
    tagger.flair_tagger.predict(sent)
    pos = [t.get_label("upos").value for t in sent]
    words = [{"token": tokens[i], "lan": "U", "pos": pos[i] if i < len(pos) else "X"} for i in range(len(tokens))]
    return tagger.tongueswitcher_detect(words)


def tag_tokens(tagger, tokens):
    return [w["lan"] for w in tag_tokens_full(tagger, tokens)]


def _iter_texts(path, limit):
    p = Path(path)
    if p.is_dir():                       # parquet shards (MonoWeb): text column
        import pyarrow.parquet as pq
        got = 0
        for f in sorted(p.glob("*.parquet")):
            for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["text"]):
                for t in b.column("text").to_pylist():
                    if t and str(t).strip():
                        yield str(t)
                        got += 1
                        if limit and got >= limit:
                            return
    else:                                # jsonl {"text": ...}
        got = 0
        with open(p) as fh:
            for line in fh:
                try:
                    t = json.loads(line).get("text", "")
                except Exception:
                    continue
                if t and t.strip():
                    yield t
                    got += 1
                    if limit and got >= limit:
                        return


def annotate(tagger, inp, out, limit, batch, tstamp):
    """Tag a corpus (jsonl or parquet dir); write per-doc {tokens,lan} + summary counts + timing."""
    from flair.data import Sentence
    from collections import Counter
    import time
    lan_ct = Counter(); sw = {"D_to_E": 0, "E_to_D": 0}
    n_docs = n_words = 0
    outf = open(out, "w") if out else None
    buf = []
    t0 = time.time()

    def flush(texts):
        nonlocal n_docs, n_words
        sents = [Sentence(x) for x in texts]
        tagger.flair_tagger.predict(sents, mini_batch_size=len(sents))
        for x, s in zip(texts, sents):
            toks = [tok.text for tok in s]
            words = [{"token": tok.text, "lan": "U", "pos": tok.get_label("upos").value} for tok in s]
            lans = [w["lan"] for w in tagger.tongueswitcher_detect(words)]
            n_docs += 1; n_words += len(lans); lan_ct.update(lans)
            prev = None
            for lg in lans:
                if lg in ("D", "E"):
                    if prev and lg != prev:
                        sw[f"{prev}_to_{lg}"] = sw.get(f"{prev}_to_{lg}", 0) + 1
                    prev = lg
            if outf:
                outf.write(json.dumps({"tokens": toks, "lan": lans}) + "\n")

    for t in _iter_texts(inp, limit):
        buf.append(t)
        if len(buf) >= batch:
            flush(buf); buf = []
    if buf:
        flush(buf)
    if outf:
        outf.close()
    dt = time.time() - t0
    summ = dict(input=str(inp), docs=n_docs, words=n_words, lan_counts=dict(lan_ct),
                switch_points=sw, seconds=round(dt, 1), words_per_sec=round(n_words / max(dt, 1e-9), 1),
                pct_E=round(100 * lan_ct.get("E", 0) / max(n_words, 1), 3),
                pct_D=round(100 * lan_ct.get("D", 0) / max(n_words, 1), 3),
                pct_M=round(100 * lan_ct.get("M", 0) / max(n_words, 1), 3),
                pct_U=round(100 * lan_ct.get("U", 0) / max(n_words, 1), 3))
    print(json.dumps(summ, indent=2), flush=True)
    return summ


def validate(tagger, gold_path, n, drop_punct=True):
    rows = []
    with open(gold_path) as f:
        for line in itertools.islice(f, n if n > 0 else None):
            r = json.loads(line)
            toks, tags = _parse_list(r["tokens"]), _parse_list(r["tags"])
            punct = r.get("punct")
            punct = _parse_list(punct) if punct is not None else [False] * len(toks)
            if toks and tags and len(toks) == len(tags):
                rows.append((toks, tags, punct))
    # Batch the flair POS prediction over all sentences at once (fast), then run the
    # rule detector per sentence. Feeding gold tokens with use_tokenizer=False keeps
    # perfect alignment to the gold labels.
    from flair.data import Sentence
    sents = [Sentence(list(toks), use_tokenizer=False) for toks, _, _ in rows]
    tagger.flair_tagger.predict(sents, mini_batch_size=64, verbose=False)

    agree = tot = 0; conf = {}; errs = []
    for (toks, gold, punct), sent in zip(rows, sents):
        pos = [t.get_label("upos").value for t in sent]
        words = [{"token": toks[i], "lan": "U", "pos": pos[i] if i < len(pos) else "X"}
                 for i in range(len(toks))]
        full = tagger.tongueswitcher_detect(words)
        pred = [w["lan"] for w in full]
        for i, (g, p) in enumerate(zip(gold, pred)):
            if g not in ("D", "E"):
                continue
            if drop_punct and i < len(punct) and punct[i]:   # official protocol excludes punctuation
                continue
            tot += 1
            conf[(g, p)] = conf.get((g, p), 0) + 1
            if p == g:
                agree += 1
            else:
                errs.append(dict(tok=toks[i], gold=g, pred=p, pos=full[i].get("pos", "?"),
                                 clh=full[i].get("clh", False),
                                 ctx=" ".join(toks[max(0, i-4):i]) + " [[" + toks[i] + "]] " + " ".join(toks[i+1:i+5])))
    print(f"sentences={len(rows)}  lexical gold tokens={tot}  drop_punct={drop_punct}  "
          f"agreement={agree/max(tot,1):.4f}", flush=True)
    print("confusion (gold -> pred):")
    for (g, p), c in sorted(conf.items()):
        print(f"  {g} -> {p}: {c}  ({c/max(tot,1)*100:.1f}%)")
    # token-level precision/recall/F1 on D and E (paper-style)
    for lg in ("D", "E"):
        tp = conf.get((lg, lg), 0)
        fp = sum(c for (g, p), c in conf.items() if p == lg and g != lg)
        fn = sum(c for (g, p), c in conf.items() if g == lg and p != lg)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"  {lg}: P={prec:.4f} R={rec:.4f} F1={f1:.4f} (support={tp+fn})")
    # dump examples per error type
    import collections
    by = collections.defaultdict(list)
    for e in errs:
        by[(e["gold"], e["pred"])].append(e)
    for key in sorted(by):
        exs = by[key]
        print(f"\n=== gold {key[0]} -> pred {key[1]}  ({len(exs)} cases; clh={sum(e['clh'] for e in exs)}) ===")
        for e in exs[:25]:
            print(f"  '{e['tok']}' pos={e['pos']} clh={e['clh']} | {e['ctx']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate","annotate","probe"], default="validate")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--probe", default="also,in,Male,Rock,bar,cool,sexy,clever,best,Account,Expansion,EBITDA,feel,imagine,weebs")
    args = ap.parse_args()
    _setup_imports()
    print("building tagger (dictionaries + flair upos-multi + HanTa)…", flush=True)
    tagger = build_tagger()
    print("tagger ready.", flush=True)
    if args.mode == "probe":
        dd = tagger.data_loader.data.dictionaries
        clh = tagger.clh_pos_tags
        sets = {k: getattr(dd, k) for k in ("cs", "de", "en", "not_cs", "en_one_two", "de_one_two") if hasattr(dd, k)}
        for k, v in sets.items():
            print(f"|{k}|={len(v):,}", end="  ")
        print()
        for w in args.probe.split(","):
            wl = w.strip().lower()
            mem = [k for k, s in sets.items() if wl in s]
            ch = f"CLH{list(clh.get(wl, {}).items())}" if wl in clh else ""
            print(f"  '{w.strip()}': {mem}  {ch}")
    elif args.mode == "validate":
        validate(tagger, args.gold, args.n)
    else:
        annotate(tagger, args.input, args.out, args.limit, args.batch, None)
