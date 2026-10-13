#!/usr/bin/env python
"""Score total BPB and switch-span / switch-point BPB on tagged evaluation jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


LEXICAL = {"E", "D"}


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def span_union_bytes(text: str, spans: list[tuple[int, int]]) -> int:
    if not spans:
        return 0
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(byte_len(text[start:end]) for start, end in merged)


def switch_targets(tags: list[str], pos: list | None = None, punct: list | None = None) -> list[tuple[int, str]]:
    targets: list[tuple[int, str]] = []
    prev = None
    for idx, tag in enumerate(tags):
        if pos is not None and idx < len(pos) and pos[idx] == "PUNCT":
            continue
        if punct is not None and idx < len(punct) and punct[idx] is True:
            continue
        if tag not in LEXICAL:
            continue
        if prev is not None and tag != prev:
            targets.append((idx, f"{prev}_to_{tag}"))
        prev = tag
    return targets


def nonswitch_targets(tags: list[str], pos: list | None = None, punct: list | None = None) -> list[tuple[int, str]]:
    """Complement of switch_targets: content words (tag in LEXICAL, not PUNCT) that are NOT switches
    (tag == previous content word's tag, or the first content word). Returns (word_idx, lang) with
    lang = the word's own tag (E/D) — used for the non-switch-span / non-switch-point baselines,
    broken out per language."""
    out: list[tuple[int, str]] = []
    prev = None
    for idx, tag in enumerate(tags):
        if pos is not None and idx < len(pos) and pos[idx] == "PUNCT":
            continue
        if punct is not None and idx < len(punct) and punct[idx] is True:
            continue
        if tag not in LEXICAL:
            continue
        is_switch = prev is not None and tag != prev
        if not is_switch:
            out.append((idx, tag))
        prev = tag
    return out


def token_nlls(model, tokenizer, text: str, device: torch.device, max_length: int) -> tuple[list[float], list[tuple[int, int]]]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    offsets = [(int(a), int(b)) for a, b in enc.pop("offset_mapping")[0].tolist()]
    if enc["input_ids"].shape[1] < 2:
        return [], []
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = enc["input_ids"][:, 1:]
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    # nll[i] scores token i+1, so align with offsets[1:].
    return [float(x) for x in nll.detach().cpu().tolist()], offsets[1:]


def score_rows(model, tokenizer, rows: list[dict], device: torch.device, max_length: int) -> dict:
    total_bits = 0.0
    total_bytes = 0
    total_nll = 0.0
    total_bpe_tokens = 0
    switch_bits = 0.0
    switch_token_span_bytes = 0
    switch_nll = 0.0
    switch_bpe_tokens = 0
    switch_words = 0
    # switch-POINT = only the FIRST BPE token of each switch word (the transition token: the
    # model's surprise at the instant it must produce the other language). switch-SPAN sums all
    # tokens of the switched word; switch-POINT isolates the boundary token.
    switch_point_bits = 0.0
    switch_point_bytes = 0
    switch_point_nll = 0.0
    switch_point_count = 0
    dirs: dict[str, dict[str, float]] = {}
    # NON-switch content words (complement of switches): span = all their BPE tokens, point = first
    # BPE token of each. Parallel to the switch metrics; broken out per language (E/D) in `nslang`.
    ns_span_bits = 0.0
    ns_span_bytes = 0
    ns_point_bits = 0.0
    ns_point_bytes = 0
    ns_point_count = 0
    ns_words = 0
    nslang: dict[str, dict[str, float]] = {}

    for row in rows:
        text = str(row.get("text", ""))
        if not text:
            continue
        nlls, offsets = token_nlls(model, tokenizer, text, device, max_length)
        if not nlls:
            continue
        for nll, span in zip(nlls, offsets):
            if span[1] <= span[0]:
                continue
            total_nll += nll
            total_bits += nll / math.log(2)
            total_bytes += byte_len(text[span[0]:span[1]])
            total_bpe_tokens += 1

        tags = row.get("tags")
        spans = row.get("token_spans")
        if not isinstance(tags, list) or not isinstance(spans, list):
            continue
        for word_idx, direction in switch_targets(tags, row.get("pos"), row.get("punct")):
            if word_idx >= len(spans):
                continue
            word_span = (int(spans[word_idx][0]), int(spans[word_idx][1]))
            if word_span[1] <= word_span[0]:
                continue
            word_nll = 0.0
            word_bpe = 0
            word_token_spans: list[tuple[int, int]] = []
            first_tok_nll = None            # switch-POINT: first BPE token overlapping the switch word
            first_tok_span = None
            for nll, tok_span in zip(nlls, offsets):
                if overlaps(tok_span, word_span):
                    if first_tok_nll is None:      # offsets are in order -> first hit is the boundary token
                        first_tok_nll, first_tok_span = nll, tok_span
                    word_nll += nll
                    word_bpe += 1
                    word_token_spans.append(tok_span)
            if word_bpe == 0:
                continue
            b = span_union_bytes(text, word_token_spans)
            if b == 0:
                continue
            bits = word_nll / math.log(2)
            switch_nll += word_nll
            switch_bits += bits
            switch_token_span_bytes += b
            switch_bpe_tokens += word_bpe
            switch_words += 1
            pt_bits = pt_bytes = 0.0
            if first_tok_span is not None:
                pt_bytes = byte_len(text[first_tok_span[0]:first_tok_span[1]])
                if pt_bytes > 0:
                    pt_bits = first_tok_nll / math.log(2)
                    switch_point_nll += first_tok_nll
                    switch_point_bits += pt_bits
                    switch_point_bytes += pt_bytes
                    switch_point_count += 1
            d = dirs.setdefault(direction, {"bits": 0.0, "bytes": 0, "nll": 0.0, "bpe_tokens": 0, "words": 0,
                                            "pt_bits": 0.0, "pt_bytes": 0})
            d["bits"] += bits
            d["bytes"] += b
            d["nll"] += word_nll
            d["bpe_tokens"] += word_bpe
            d["words"] += 1
            d["pt_bits"] += pt_bits
            d["pt_bytes"] += pt_bytes if (first_tok_span is not None and pt_bytes > 0) else 0

        # NON-switch content words: same span/point logic, bucketed by the word's language.
        for word_idx, lang in nonswitch_targets(tags, row.get("pos"), row.get("punct")):
            if word_idx >= len(spans):
                continue
            word_span = (int(spans[word_idx][0]), int(spans[word_idx][1]))
            if word_span[1] <= word_span[0]:
                continue
            word_nll = 0.0
            word_token_spans: list[tuple[int, int]] = []
            first_tok_nll = None
            first_tok_span = None
            for nll, tok_span in zip(nlls, offsets):
                if overlaps(tok_span, word_span):
                    if first_tok_nll is None:
                        first_tok_nll, first_tok_span = nll, tok_span
                    word_nll += nll
                    word_token_spans.append(tok_span)
            if not word_token_spans:
                continue
            b = span_union_bytes(text, word_token_spans)
            if b == 0:
                continue
            bits = word_nll / math.log(2)
            ns_span_bits += bits
            ns_span_bytes += b
            ns_words += 1
            pt_bits = pt_bytes = 0.0
            if first_tok_span is not None:
                pt_bytes = byte_len(text[first_tok_span[0]:first_tok_span[1]])
                if pt_bytes > 0:
                    pt_bits = first_tok_nll / math.log(2)
                    ns_point_bits += pt_bits
                    ns_point_bytes += pt_bytes
                    ns_point_count += 1
            nd = nslang.setdefault(lang, {"bits": 0.0, "bytes": 0, "words": 0, "pt_bits": 0.0, "pt_bytes": 0})
            nd["bits"] += bits
            nd["bytes"] += b
            nd["words"] += 1
            nd["pt_bits"] += pt_bits
            nd["pt_bytes"] += pt_bytes if (first_tok_span is not None and pt_bytes > 0) else 0

    # Non-switch span = everything in these same docs that is NOT a switch-word span.
    # This gives a within-corpus baseline (Exp11 Fix 1): excess = switch - nonswitch,
    # so the baseline is the model's cost on the *same* CS corpus, not a separate distribution.
    nonswitch_bits = total_bits - switch_bits
    nonswitch_bytes = total_bytes - switch_token_span_bytes
    no_switch_span_bpb = nonswitch_bits / nonswitch_bytes if nonswitch_bytes > 0 else None
    switch_token_span_bpb = switch_bits / switch_token_span_bytes if switch_token_span_bytes else None

    result = {
        "total_bpb": total_bits / total_bytes if total_bytes else None,
        "total_mean_token_nll": total_nll / total_bpe_tokens if total_bpe_tokens else None,
        "total_ppl": math.exp(total_nll / total_bpe_tokens) if total_bpe_tokens and total_nll / total_bpe_tokens < 50 else None,
        "n_total_bytes": total_bytes,
        "n_total_bpe_tokens": total_bpe_tokens,
        "switch_token_span_bpb": switch_token_span_bpb,
        "switch_point_bpb": switch_point_bits / switch_point_bytes if switch_point_bytes else None,
        "no_switch_span_bpb": no_switch_span_bpb,
        "switch_excess_within_bpb": (switch_token_span_bpb - no_switch_span_bpb)
        if switch_token_span_bpb is not None and no_switch_span_bpb is not None else None,
        "switch_word_mean_nll": switch_nll / switch_words if switch_words else None,
        "n_switch_token_span_bytes": switch_token_span_bytes,
        "n_switch_point_bytes": switch_point_bytes,
        "n_no_switch_span_bytes": nonswitch_bytes,
        "n_switch_bpe_tokens": switch_bpe_tokens,
        "n_switch_words": switch_words,
        "n_switch_points": switch_point_count,
        # non-switch content-word baselines (parallel to switch-span / switch-point)
        "no_switch_content_span_bpb": ns_span_bits / ns_span_bytes if ns_span_bytes else None,
        "no_switch_point_bpb": ns_point_bits / ns_point_bytes if ns_point_bytes else None,
        "n_no_switch_content_span_bytes": ns_span_bytes,
        "n_no_switch_point_bytes": ns_point_bytes,
        "n_no_switch_words": ns_words,
        "n_no_switch_points": ns_point_count,
    }
    for direction, vals in dirs.items():
        result[f"{direction}_switch_token_span_bpb"] = vals["bits"] / vals["bytes"] if vals["bytes"] else None
        result[f"{direction}_switch_point_bpb"] = vals["pt_bits"] / vals["pt_bytes"] if vals["pt_bytes"] else None
        result[f"{direction}_n_switch_words"] = vals["words"]
        result[f"{direction}_n_switch_token_span_bytes"] = vals["bytes"]
        result[f"{direction}_n_switch_bpe_tokens"] = vals["bpe_tokens"]
    # non-switch broken out by the word's own language (E/D)
    for lang, vals in nslang.items():
        result[f"{lang}_no_switch_span_bpb"] = vals["bits"] / vals["bytes"] if vals["bytes"] else None
        result[f"{lang}_no_switch_point_bpb"] = vals["pt_bits"] / vals["pt_bytes"] if vals["pt_bytes"] else None
        result[f"{lang}_n_no_switch_words"] = vals["words"]
    return result


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-name", nargs="+", required=True)
    parser.add_argument("--train-condition", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if len(args.eval_jsonl) != len(args.eval_name):
        raise SystemExit("--eval-jsonl and --eval-name must have the same length")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=True, add_prefix_space=True)
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit(f"Tokenizer must be fast to provide offset mappings: {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir)).to(device)
    model.eval()

    rows_out = []
    for name, path in zip(args.eval_name, args.eval_jsonl):
        metrics = score_rows(model, tokenizer, read_jsonl(path, args.limit), device, args.max_length)
        metrics["train_condition"] = args.train_condition
        metrics["eval_name"] = name
        metrics["model_dir"] = str(args.model_dir)
        rows_out.append(metrics)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows_out for key in row})
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    print(json.dumps(rows_out, indent=2))


if __name__ == "__main__":
    main()
