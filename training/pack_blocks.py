#!/usr/bin/env python
"""Pretokenize Exp6 raw JSONL splits into reusable packed token blocks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from transformers import GPT2TokenizerFast


def complete_metadata(path: Path) -> dict | None:
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
    block_file = Path(meta.get("block_file", ""))
    packed_tokens = int(meta.get("packed_tokens") or 0)
    if not block_file.exists() or packed_tokens <= 0:
        return None
    expected_bytes = packed_tokens * np.dtype(np.uint16).itemsize
    if block_file.stat().st_size != expected_bytes:
        return None
    return meta


def iter_jsonl_texts(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if text:
                yield text, row


def flush_blocks(out_handle, carry: list[int], block_size: int) -> tuple[int, int]:
    n_blocks = len(carry) // block_size
    if n_blocks <= 0:
        return 0, 0
    n_tokens = n_blocks * block_size
    arr = np.asarray(carry[:n_tokens], dtype=np.uint16)
    arr.tofile(out_handle)
    del carry[:n_tokens]
    return n_blocks, n_tokens


def pack_condition(
    condition_dir: Path,
    out_dir: Path,
    tokenizer: GPT2TokenizerFast,
    block_size: int,
    batch_size: int,
    log_every: int,
) -> dict:
    condition = condition_dir.name
    train_paths = sorted(condition_dir.glob("train_*.jsonl"))
    if not train_paths:
        raise ValueError(f"No train_*.jsonl files in {condition_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_bin = out_dir / "train_blocks_u16.bin"
    eos = tokenizer.eos_token or ""
    max_id = len(tokenizer) - 1
    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(f"Tokenizer too large for uint16: max id {max_id}")

    start = time.time()
    carry: list[int] = []
    rows = 0
    source_tokens = 0
    chars = 0
    bytes_ = 0
    model_tokens = 0
    blocks = 0
    batch_texts: list[str] = []
    input_files = []

    with out_bin.open("wb") as out_handle:
        for path in train_paths:
            file_rows = 0
            file_source_tokens = 0
            input_files.append(str(path))
            for text, row in iter_jsonl_texts(path):
                batch_texts.append(text + eos)
                rows += 1
                file_rows += 1
                n_source = int(row.get("num_tokens") or len(text.split()))
                source_tokens += n_source
                file_source_tokens += n_source
                chars += len(text)
                bytes_ += len(text.encode("utf-8"))
                if len(batch_texts) >= batch_size:
                    encoded = tokenizer(batch_texts, add_special_tokens=False, padding=False, truncation=False)
                    for ids in encoded["input_ids"]:
                        model_tokens += len(ids)
                        carry.extend(ids)
                    n_blocks, _ = flush_blocks(out_handle, carry, block_size)
                    blocks += n_blocks
                    batch_texts.clear()
                if log_every and rows % log_every == 0:
                    elapsed = max(time.time() - start, 1e-6)
                    print(
                        json.dumps(
                            {
                                "condition": condition,
                                "rows": rows,
                                "source_tokens": source_tokens,
                                "model_tokens": model_tokens,
                                "blocks": blocks,
                                "rows_per_sec": rows / elapsed,
                            }
                        ),
                        flush=True,
                    )
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "finished_input": str(path),
                        "file_rows": file_rows,
                        "file_source_tokens": file_source_tokens,
                    }
                ),
                flush=True,
            )
        if batch_texts:
            encoded = tokenizer(batch_texts, add_special_tokens=False, padding=False, truncation=False)
            for ids in encoded["input_ids"]:
                model_tokens += len(ids)
                carry.extend(ids)
            n_blocks, _ = flush_blocks(out_handle, carry, block_size)
            blocks += n_blocks
            batch_texts.clear()

    dropped_remainder_tokens = len(carry)
    packed_tokens = blocks * block_size
    meta = {
        "condition": condition,
        "input_files": input_files,
        "block_file": str(out_bin),
        "dtype": "uint16",
        "block_size": block_size,
        "rows": rows,
        "source_tokens": source_tokens,
        "chars": chars,
        "bytes": bytes_,
        "model_tokens_with_eos": model_tokens,
        "packed_blocks": blocks,
        "packed_tokens": packed_tokens,
        "dropped_remainder_tokens": dropped_remainder_tokens,
        "tokenizer_vocab_size": len(tokenizer),
        "elapsed_sec": time.time() - start,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"done_condition": condition, **meta}), flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--condition", action="append", default=None)
    parser.add_argument("--log-every", type=int, default=100_000)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained(str(args.tokenizer_dir), add_prefix_space=True)
    selected = set(args.condition or [])
    condition_dirs = sorted(path for path in args.splits_dir.iterdir() if path.is_dir())
    if selected:
        condition_dirs = [path for path in condition_dirs if path.name in selected]
    if not condition_dirs:
        raise SystemExit("No condition directories selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_meta = []
    start = time.time()
    for condition_dir in condition_dirs:
        condition_out = args.out_dir / condition_dir.name
        if args.skip_existing:
            meta = complete_metadata(condition_out)
            if meta is not None:
                print(json.dumps({"skip_existing_condition": condition_dir.name, "metadata": str(condition_out / "metadata.json")}), flush=True)
                all_meta.append(meta)
                continue
        all_meta.append(
            pack_condition(
                condition_dir,
                condition_out,
                tokenizer,
                args.block_size,
                args.batch_size,
                args.log_every,
            )
        )
    manifest = {
        "splits_dir": str(args.splits_dir),
        "tokenizer_dir": str(args.tokenizer_dir),
        "out_dir": str(args.out_dir),
        "block_size": args.block_size,
        "conditions": all_meta,
        "elapsed_sec": time.time() - start,
    }
    (args.out_dir / "packed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"done": str(args.out_dir), "conditions": len(all_meta), "elapsed_sec": manifest["elapsed_sec"]}), flush=True)


if __name__ == "__main__":
    main()
