#!/usr/bin/env python
"""Train GPT-2 from pretokenized packed blocks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import Dataset


def _is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _unwrap(model):
    # under DDP the Trainer passes a DistributedDataParallel wrapper; score the raw module so the
    # forward does NOT trigger DDP's gradient-sync hooks (which would deadlock without a backward).
    return model.module if hasattr(model, "module") else model
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'metrics'))
from score_switch_bpb import read_jsonl, score_rows
from acs_scoring import load_acs_pairs, score_acs


class PackedBlockDataset(Dataset):
    def __init__(
        self,
        bin_path: Path,
        block_size: int,
        start_block: int = 0,
        n_blocks: int = 0,
        block_indices: np.ndarray | None = None,
    ) -> None:
        self.bin_path = Path(bin_path)
        self.block_size = int(block_size)
        self.start_block = int(start_block)
        self.block_indices = block_indices
        self.n_blocks = len(block_indices) if block_indices is not None else int(n_blocks)
        self._data = None

    def __len__(self) -> int:
        return self.n_blocks

    @property
    def data(self):
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        return self._data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.n_blocks:
            raise IndexError(idx)
        block_idx = int(self.block_indices[idx]) if self.block_indices is not None else self.start_block + idx
        start = block_idx * self.block_size
        arr = np.asarray(self.data[start : start + self.block_size], dtype=np.int64)
        input_ids = torch.from_numpy(arr.copy())
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data"] = None
        return state


class DynamicsScorer(TrainerCallback):
    def __init__(
        self,
        tokenizer: GPT2TokenizerFast,
        eval_rows: dict[str, list[dict]],
        out_csv: Path,
        condition: str,
        max_length: int,
        score_points: int,
        max_steps: int,
        acs_pairs: list | None = None,
        acs_batch_size: int = 64,
        acs_max_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_rows = eval_rows
        self.out_csv = out_csv
        self.condition = condition
        self.max_length = max_length
        self.max_steps = max_steps
        self.acs_pairs = acs_pairs
        self.acs_batch_size = acs_batch_size
        self.acs_max_length = acs_max_length
        # LOG-spaced eval points from 1..max_steps (step 0 handled separately in on_train_begin),
        # so the fast early dynamics are well resolved with few points. Always includes max_steps.
        n = max(1, score_points)
        pts = np.unique(np.geomspace(1, max(1, max_steps), n).round().astype(int))
        self.targets = sorted(set(int(x) for x in pts if 1 <= x <= max_steps) | {max_steps})
        self._idx = 0
        self.rows: list[dict] = []

    def _score(self, model, step: int) -> None:
        # DDP-safe: only rank 0 runs the eval forward (on the unwrapped module, so no gradient-sync
        # hooks fire); other ranks wait at the barrier. Params are identical across ranks under DDP,
        # so rank 0's eval is representative. This is what lets the big models train multi-GPU while
        # still producing the same dynamics logs as the single-GPU runs.
        if not _is_main_process():
            _barrier()
            return
        model = _unwrap(model)
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        metrics_by_name = {}
        for name, rows in self.eval_rows.items():
            metrics_by_name[name] = score_rows(model, self.tokenizer, rows, device, self.max_length)
        # ACS minimal-pairs (local TS-subset csv): default-on downstream validation. NEVER lets a
        # scoring failure kill training — keys stay present (None) so the csv header is stable.
        acs = {"acs_acc_raw": None, "acs_acc_bpb": None, "acs_margin_bpb": None, "acs_n_pairs": None}
        if self.acs_pairs:
            try:
                acs = score_acs(model, self.tokenizer, self.acs_pairs, device,
                                self.acs_max_length, self.acs_batch_size)
            except Exception as exc:  # noqa: BLE001
                print(f"ACS_SCORING_WARNING step={step}: {exc}", flush=True)
        if was_training:
            model.train()

        # Monolingual dense eval = held-out MonoWeb-de/en (same distribution as training), so
        # base = avg(MonoWeb-de, MonoWeb-en) is the headline non-switch baseline. Every CS corpus
        # (denglisch = gold labels, tongueswitcher = auto labels) is recorded with prefixed columns,
        # each with switch-span/switch-point BPB, within-corpus + vs-monolingual excess, and directions.
        eng = metrics_by_name["monoweb_eng"]["total_bpb"]
        deu = metrics_by_name["monoweb_deu"]["total_bpb"]
        base = (eng + deu) / 2 if eng is not None and deu is not None else None
        row = {
            "train_condition": self.condition,
            "step": step,
            "epoch_fraction": step / self.max_steps if self.max_steps else None,
            "monoweb_eng_overall_bpb": eng,
            "monoweb_deu_overall_bpb": deu,
            "monoweb_avg_bpb": base,
            # (per-token NLL columns dropped: recoverable from BPB via the eval set's fixed bytes/token)
        }
        for cs_name in [n for n in metrics_by_name if not n.startswith("monoweb")]:
            m = metrics_by_name[cs_name]
            sp = m.get("switch_token_span_bpb")
            pt = m.get("switch_point_bpb")
            ns = m.get("no_switch_span_bpb")
            row[f"{cs_name}_overall_bpb"] = m.get("total_bpb")
            row[f"{cs_name}_switch_token_span_bpb"] = sp
            row[f"{cs_name}_switch_point_bpb"] = pt
            row[f"{cs_name}_nonswitch_bpb"] = ns
            row[f"{cs_name}_switch_excess_within_bpb"] = (sp - ns) if sp is not None and ns is not None else None
            row[f"{cs_name}_E_to_D_switch_token_span_bpb"] = m.get("E_to_D_switch_token_span_bpb")
            row[f"{cs_name}_D_to_E_switch_token_span_bpb"] = m.get("D_to_E_switch_token_span_bpb")
            row[f"{cs_name}_E_to_D_switch_point_bpb"] = m.get("E_to_D_switch_point_bpb")
            row[f"{cs_name}_D_to_E_switch_point_bpb"] = m.get("D_to_E_switch_point_bpb")
            # headline per corpus: switch cost minus in-distribution monolingual baseline (monoweb_avg)
            row[f"{cs_name}_switch_excess_bpb"] = (sp - base) if sp is not None and base is not None else None
            row[f"{cs_name}_switch_point_excess_bpb"] = (pt - base) if pt is not None and base is not None else None
            row[f"{cs_name}_n_switch_words"] = m.get("n_switch_words")
            row[f"{cs_name}_n_switch_points"] = m.get("n_switch_points")
        # convenience aliases: "headline" = the gold Denglisch corpus
        row["switch_excess_bpb"] = row.get("denglisch_manual_switch_excess_bpb")
        row["switch_point_excess_bpb"] = row.get("denglisch_manual_switch_point_excess_bpb")
        row.update(acs)
        if wandb.run is not None:
            wandb.log(row, step=step)
        self.rows.append(row)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        print(json.dumps(row), flush=True)
        _barrier()   # rank 0 finishes writing; other ranks resume together

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # step-0 eval: the random-init baseline (was requested in the plan but never actually scored).
        if model is not None:
            self._score(model, 0)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        # fire at the LOG-spaced target steps (label with the actual global step reached).
        while self._idx < len(self.targets) and state.global_step >= self.targets[self._idx]:
            self._score(model, state.global_step)
            self._idx += 1
        return control


class EvalLossIncreaseWarning(TrainerCallback):
    def __init__(self, threshold: float, metric: str = "eval_loss") -> None:
        self.threshold = float(threshold)
        self.metric = metric
        self.previous: float | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.threshold <= 0 or not metrics or self.metric not in metrics:
            return control
        current = float(metrics[self.metric])
        if self.previous is not None:
            delta = current - self.previous
            if delta > self.threshold:
                warning = {
                    "step": int(state.global_step),
                    "metric": self.metric,
                    "previous": self.previous,
                    "current": current,
                    "delta": delta,
                    "threshold": self.threshold,
                }
                print("EVAL_LOSS_INCREASE_WARNING " + json.dumps(warning), flush=True)
                if wandb.run is not None:
                    wandb.log(
                        {
                            f"warning/{self.metric}_increase": delta,
                            f"warning/{self.metric}": current,
                        },
                        step=state.global_step,
                    )
        self.previous = current
        return control


class ExactStepCheckpoint(TrainerCallback):
    def __init__(self, output_dir: Path, tokenizer: GPT2TokenizerFast, checkpoint_steps: list[int]) -> None:
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.checkpoint_steps = set(int(step) for step in checkpoint_steps if int(step) > 0)
        self.saved: set[int] = set()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        step = int(state.global_step)
        if step in self.checkpoint_steps and step not in self.saved:
            if _is_main_process():   # DDP-safe: only rank 0 writes the checkpoint (unwrapped module)
                checkpoint_dir = self.output_dir / f"checkpoint-step-{step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                _unwrap(model).save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)
                print(json.dumps({"saved_exact_checkpoint": str(checkpoint_dir), "step": step}), flush=True)
            self.saved.add(step)
            _barrier()
        return control


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, nargs="+", required=True,
                        help="monoweb_eng, monoweb_deu, then one or more CS corpora (denglisch, tongueswitcher)")
    parser.add_argument("--eval-name", nargs="+", required=True)
    parser.add_argument("--dense-eval-limit", type=int, default=0,
                        help="cap docs per eval set for the in-training dense curve (0=all). "
                             "The dense curve is noise-tolerant; final scoring still uses the full sets.")
    parser.add_argument("--train-condition", required=True)
    parser.add_argument("--dynamics-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--checkpoint-step", type=int, action="append", default=[])
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--score-points", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--report-to", default="none", choices=["none", "wandb"])
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--eval-loss-warning-threshold", type=float, default=0.05)
    parser.add_argument("--eval-loss-warning-metric", default="eval_loss")
    parser.add_argument("--random-packed-split", action="store_true")
    parser.add_argument("--allow-resume", action="store_true",
                        help="resume from an existing checkpoint (genuine requeue continuation). Default OFF: "
                             "with --overwrite a fresh run never resumes a leftover checkpoint.")
    parser.add_argument("--eval-max-blocks", type=int, default=0,
                        help="optional hard cap on the HF eval_loss held-out split (0 = use the full 1%% split).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save resumable Trainer checkpoints (optimizer+scheduler+RNG) for long/frontier runs.")
    parser.add_argument("--no-dynamics", action="store_true", help="Skip the in-training dynamics scorer (DDP-safe for multi-GPU frontier).")
    # ACS minimal-pairs benchmark: scored by DEFAULT at every dynamics eval point (local csv, no
    # HF download; the TS-subset pairs are leak-safe vs the leak-free TS training pool).
    parser.add_argument("--acs-pairs-csv", type=Path,
                        default=Path(os.environ["ACS_PAIRS"]) if os.environ.get("ACS_PAIRS") else None,
                        help="ACS pairs CSV (rebuild with metrics/fetch_acs_subset.py). "
                             "Or set ACS_PAIRS. Omit / pass --no-acs to skip.")
    parser.add_argument("--no-acs", action="store_true", help="Skip ACS scoring in the dynamics logs.")
    parser.add_argument("--acs-limit", type=int, default=0, help="cap pairs (0 = all ~7.4k).")
    parser.add_argument("--acs-batch-size", type=int, default=64)
    parser.add_argument("--acs-max-length", type=int, default=256)
    # Exp10 representation diagnostics
    parser.add_argument("--rep-diagnostics", action="store_true",
                        help="Enable Exp10 language-separability / switch-geometry diagnostics.")
    parser.add_argument("--rep-csv", type=Path, default=None)
    parser.add_argument("--rep-reps-dir", type=Path, default=None,
                        help="If set, save compact per-checkpoint representations (npz) here.")
    parser.add_argument("--rep-en-jsonl", type=Path, default=None)
    parser.add_argument("--rep-de-jsonl", type=Path, default=None)
    parser.add_argument("--rep-cs-jsonl", type=Path, default=None)
    parser.add_argument("--rep-n-mono-docs", type=int, default=80)
    parser.add_argument("--rep-max-cs-rows", type=int, default=600)
    parser.add_argument("--rep-max-tokens-per-lang", type=int, default=4000)
    parser.add_argument("--rep-layers", type=int, nargs="*", default=None)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not (args.overwrite or args.allow_resume):
        raise SystemExit(f"Refusing to overwrite non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    meta_path = args.packed_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    block_size = int(meta["block_size"])
    total_blocks = int(meta["packed_blocks"])
    if total_blocks < 2:
        raise SystemExit(f"Not enough packed blocks in {args.packed_dir}")
    # HF eval_loss held-out split: 1% of the packed data (was 5%). This is a genuine held-out signal
    # (overfitting check + independent perplexity), kept meaningful but ~5x cheaper than 5%. An optional
    # --eval-max-blocks can further cap it, but by default we keep the proportional 1% split.
    split_block = max(1, int(0.99 * total_blocks))
    if split_block >= total_blocks:
        split_block = total_blocks - 1
    n_eval = total_blocks - split_block
    eval_cap = args.eval_max_blocks if args.eval_max_blocks > 0 else n_eval
    if args.random_packed_split:
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(total_blocks)
        train_ds = PackedBlockDataset(Path(meta["block_file"]), block_size, block_indices=indices[:split_block])
        eval_ds = PackedBlockDataset(Path(meta["block_file"]), block_size, block_indices=indices[split_block:split_block + eval_cap])
    else:
        train_ds = PackedBlockDataset(Path(meta["block_file"]), block_size, 0, split_block)
        eval_ds = PackedBlockDataset(Path(meta["block_file"]), block_size, split_block, min(n_eval, eval_cap))

    if args.report_to == "wandb":
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group

    tokenizer = GPT2TokenizerFast.from_pretrained(str(args.tokenizer_dir), add_prefix_space=True)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    # Cap only the large monolingual MonoWeb sets (BPB over a few hundred docs is already very stable);
    # always use the FULL code-switch corpus (it is small and switch events are the scarce, noisy signal).
    eval_rows = {name: read_jsonl(path, args.dense_eval_limit if name.startswith("monoweb") else 0)
                 for name, path in zip(args.eval_name, args.eval_jsonl)}
    score_steps = max(1, args.max_steps // args.score_points)

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=args.overwrite,
        max_steps=args.max_steps,
        seed=args.seed,        # drives weight init AND the train dataloader shuffle order -> per-seed data ORDER
        data_seed=args.seed,   # explicit: same global pool, different shuffle order per seed
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=max(10, args.max_steps // 20),
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy=("steps" if args.save_checkpoints else "no"),
        save_steps=args.save_steps,
        save_total_limit=1,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=args.wandb_run_name or args.output_dir.name,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    callbacks = []
    acs_pairs = None
    if not args.no_acs and not args.no_dynamics and args.acs_pairs_csv:
        try:
            acs_pairs = load_acs_pairs(args.acs_pairs_csv, args.acs_limit)
            print(f"ACS: {len(acs_pairs)} minimal pairs from {args.acs_pairs_csv}", flush=True)
        except Exception as exc:  # missing csv on some node/filesystem must never block training
            print(f"ACS_LOAD_WARNING: {exc} — ACS scoring disabled for this run", flush=True)
    if not args.no_dynamics:   # DDP-safe now: scorer runs on rank 0 (unwrapped module) inside a barrier
        callbacks.append(DynamicsScorer(
            tokenizer, eval_rows, args.dynamics_csv, args.train_condition,
            block_size, args.score_points, args.max_steps,
            acs_pairs=acs_pairs, acs_batch_size=args.acs_batch_size,
            acs_max_length=args.acs_max_length,
        ))
    callbacks.append(EvalLossIncreaseWarning(args.eval_loss_warning_threshold, args.eval_loss_warning_metric))
    if args.checkpoint_step:
        callbacks.append(ExactStepCheckpoint(args.output_dir, tokenizer, args.checkpoint_step))
    if args.rep_diagnostics:
        # not used for any paper result; optional module, imported lazily
        from rep_diagnostics import RepresentationDiagnostics, select_layers, rows_with_switches
        rep_csv = args.rep_csv or (args.dynamics_csv.parent / (args.dynamics_csv.stem + "_rep.csv"))
        en_path = args.rep_en_jsonl or args.eval_jsonl[0]
        de_path = args.rep_de_jsonl or args.eval_jsonl[1]
        cs_path = args.rep_cs_jsonl or args.eval_jsonl[2]
        en_texts = [r["text"] for r in read_jsonl(en_path, args.rep_n_mono_docs) if r.get("text")]
        de_texts = [r["text"] for r in read_jsonl(de_path, args.rep_n_mono_docs) if r.get("text")]
        cs_rows = rows_with_switches(read_jsonl(cs_path), args.rep_max_cs_rows)
        rep_layers = args.rep_layers if args.rep_layers else select_layers(args.n_layer)
        callbacks.append(
            RepresentationDiagnostics(
                tokenizer, en_texts, de_texts, cs_rows, rep_csv, args.train_condition,
                rep_layers, block_size, score_steps, args.max_steps,
                reps_dir=args.rep_reps_dir, max_tokens_per_lang=args.rep_max_tokens_per_lang,
                seed=args.seed,
            )
        )
        print(json.dumps({"rep_diagnostics": True, "rep_csv": str(rep_csv),
                          "rep_layers": rep_layers, "n_en": len(en_texts), "n_de": len(de_texts),
                          "n_cs_rows": len(cs_rows)}), flush=True)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks,
    )
    import glob as _glob
    # Resume ONLY from a valid HF Trainer checkpoint (has trainer_state.json). Legacy dynamics
    # snapshots are named "checkpoint-step-*" and lack trainer_state.json, so a bare "checkpoint-*"
    # glob wrongly requests a resume that HF then rejects ("No valid checkpoint found").
    # Resume ONLY when NOT overwriting. With --overwrite (every fresh sweep run), a leftover checkpoint
    # from a prior wave must NOT be resumed: resuming re-ran the LR schedule from peak on an already-
    # annealed model, stitching two anneal cycles into one run (eval-loss spike + epoch~2). --overwrite
    # means "start fresh", so honor it. Use --allow-resume for a genuine requeue continuation.
    _resume = bool(getattr(args, "allow_resume", False) and not args.overwrite and args.save_checkpoints
                   and _glob.glob(str(args.output_dir / "checkpoint-*" / "trainer_state.json")))
    trainer.train(resume_from_checkpoint=_resume)
    metrics = trainer.evaluate()
    if metrics.get("eval_loss", 100) < 50:
        metrics["eval_ppl"] = math.exp(metrics["eval_loss"])
    metrics["packed_metadata"] = meta
    metrics["train_blocks"] = len(train_ds)
    metrics["eval_blocks"] = len(eval_ds)
    trainer.save_model(args.output_dir / "final")
    tokenizer.save_pretrained(args.output_dir / "final")
    (args.output_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    fields = sorted({key for row in trainer.state.log_history for key in row})
    with (args.output_dir / "log_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trainer.state.log_history)
    (args.output_dir / "run_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
