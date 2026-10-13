# Code release: code-switching emergence and scaling in bilingual LMs

Our code release for our paper.

**Figures only** (no corpora downloaded, no GPU): the numeric tables in `figures/data/` and
`fits/data/` are enough.

**Train or rescore models:** download the official corpora (links in
[`DATA.md`](DATA.md)), then run the pipeline below. This repo does not ship
tweet text, FineWeb shards, or ACS sentences due to licensing.

| Corpus | Link |
|---|---|
| FineWeb (EN) | https://huggingface.co/datasets/HuggingFaceFW/fineweb |
| FineWeb-2 (DE) | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| TongueSwitcher | https://zenodo.org/records/10011601 (restricted) |
| ACS | https://huggingface.co/datasets/igorsterner/acs-benchmark (gated) |
| Denglisch | https://github.com/HaifaCLG/Denglisch |

## Reproduce the paper figures

```
pip install -r requirements-figures.txt
python figures/make_figures.py      # Figures 1–4 → figures/out/
python fits/fit_scaling.py          # scaling-fit table
```

Architecture tags in the CSVs: `t11`=11M, `t22`=22M, `gpt2sL4`=52M,
`gpt2sL12`=108M, `gpt2mL16`=232M, `gpt2mL24`=334M, `gpt2lL24`=511M,
`gpt2xlL25`=817M.

## Train models:

1. Copy `paths.env.example` → `paths.env`. Set `CS_ROOT` (writable work dir) and
   `CS_ENV` (venv with `requirements.txt`). Optional Slurm site flags:
   `SLURM_PARTITION`, `SLURM_CPU_PARTITION`, `SLURM_ACCOUNT`.
2. Download FineWeb, TongueSwitcher, ACS, and Denglisch from the links above
   ([`DATA.md`](DATA.md) has access notes). Rebuild ACS sentences with
   `metrics/fetch_acs_subset.py`; filter TongueSwitcher against those pairs with
   `soloweb/prepare_ts_leakfree.py`.
3. Train the joint 30k BPE, then pack and train:

```
python training/train_sw_union_tokenizer.py \
  --soloweb-eng $CS_ROOT/data/fineweb_clean/fineweb_clean_eng_train.jsonl \
  --soloweb-deu $CS_ROOT/data/fineweb_clean/fineweb_clean_deu_train.jsonl \
  --tongueswitcher $CS_ROOT/data/tongueswitcher_train_leakfree.jsonl \
  --denglisch $CS_ROOT/data/denglisch.jsonl \
  --out-dir $CS_ROOT/data/sw_union_tokenizer

# then one of:
bash training/submit_fig1.sh          # DRY=1 to print the job graph
bash training/submit_fig2_sweep.sh
bash training/submit_fig3.sh
bash training/submit_fig4.sh
```

4. Metrics are written into each run’s dynamics CSV during training (only if the
   run started at step 0). Post-hoc scoring: `metrics/score_one.sbatch`.
5. `figures/export_eval_data.py` is the provenance script that built the shipped
   CSVs. End users do not need it.

Python scripts take explicit flags and do not hard-code a cluster. Wrappers
source `_env.sh` and submit through `sbatch_site`.

## Folders

| Folder | What |
|---|---|
| `soloweb/` | Download FineWeb, sentence-LID cleaning, held-out split, pools, leak-free TS filter |
| `training/` | Joint BPE, packing, GPT-2 trainer, `submit_fig*.sh` |
| `metrics/` | Switch BPB, ACS scoring, `acs_pair_ids.txt` + `fetch_acs_subset.py` |
| `figures/` / `fits/` | Shipped eval tables and plotting / scaling-law scripts |
| `generation/` | Temperature-sweep sampling and TongueSwitcher tagging (appendix) |
| `misc/` | MonoWeb residual-language audit |

## Conventions

- Columns named `monoweb_*` are historical; paper runs evaluate on held-out average monolingual docs (we call this SoloWeb internally) (`MONO_HELDOUT`).
- All paper runs use the joint 30k BPE trained on SoloWeb-EN/DE + leak-free
  TongueSwitcher + Denglisch. The BPE is trained with `add_prefix_space=False`;
  every consumer loads with `True`, and the saved tokenizer pins `True`.
- Run-directory prefixes (`exp12_sw7_*`, `exp12_swp0_*`, …) are historical names
  kept so the shipped CSVs still match. They are not site-specific.

## Disclosure:

This repo was packaged together by the authors with the help of Claude Fable 5 and Grok 4.6.