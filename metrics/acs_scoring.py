#!/usr/bin/env python
"""Shared ACS minimal-pair scorer (Sterner & Teufel 2025), used by the trainer's in-training
DynamicsScorer AND the post-hoc score_acs_local.py CLI.

Pairs come from a local CSV rebuilt by metrics/fetch_acs_subset.py (the ACS de-en pairs
whose observed sentence occurs in TongueSwitcher). The matching TS docs are removed
from training, so scoring is leak-safe. No HF download at eval time.

A model "passes" a pair when it assigns the observed (natural CS) sentence lower loss than the
minimally manipulated variant. We report accuracy under raw summed NLL and under per-byte (BPB)
normalization (the pair sentences differ in length, so BPB is the fairer headline), plus the
mean BPB margin (manipulated - observed; positive = observed preferred).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch


def load_acs_pairs(csv_path: Path, limit: int = 0) -> list[tuple[str, str]]:
    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            o, m = row.get("observed_sentence", ""), row.get("manipulated_sentence", "")
            if o and m:
                pairs.append((o, m))
            if limit and len(pairs) >= limit:
                break
    return pairs


@torch.no_grad()
def _batch_nll_bits(model, tokenizer, texts, device, max_length: int, batch_size: int):
    """Summed next-token NLL in bits + UTF-8 byte length, per text (right-padded batches)."""
    eos = tokenizer.eos_token_id
    bits, nbytes = [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        encs = [tokenizer.encode(t)[:max_length] for t in chunk]
        L = max(2, max(len(e) for e in encs))
        ids = torch.full((len(encs), L), eos, dtype=torch.long)
        mask = torch.zeros((len(encs), L), dtype=torch.long)
        for j, e in enumerate(encs):
            ids[j, :len(e)] = torch.tensor(e, dtype=torch.long)
            mask[j, :len(e)] = 1
        ids, mask = ids.to(device), mask.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)   # nats per position
        nll = (nll * mask[:, 1:]).sum(dim=1)                           # padding scores zero
        for j, t in enumerate(chunk):
            bits.append(float(nll[j]) / math.log(2))
            nbytes.append(max(1, len(t.encode("utf-8"))))
    return bits, nbytes


def score_acs(model, tokenizer, pairs, device, max_length: int = 256,
              batch_size: int = 64) -> dict:
    """Model must already be in eval mode. Returns the four acs_* metrics."""
    obs = [p[0] for p in pairs]
    man = [p[1] for p in pairs]
    ob, oby = _batch_nll_bits(model, tokenizer, obs, device, max_length, batch_size)
    mb, mby = _batch_nll_bits(model, tokenizer, man, device, max_length, batch_size)
    n = len(pairs)
    obpb = [b / y for b, y in zip(ob, oby)]
    mbpb = [b / y for b, y in zip(mb, mby)]
    return {
        "acs_acc_raw": sum(o < m for o, m in zip(ob, mb)) / n,
        "acs_acc_bpb": sum(o < m for o, m in zip(obpb, mbpb)) / n,
        "acs_margin_bpb": sum(m - o for o, m in zip(obpb, mbpb)) / n,
        "acs_n_pairs": n,
    }
