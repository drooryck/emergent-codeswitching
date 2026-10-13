# Where to get the data

This repo ships **code and numeric tables**, not corpora. Figures 1–4 redraw from
`figures/data/` with no download. Training or rescoring needs the official
releases below (log in to Hugging Face first: `huggingface-cli login`).

| What | Get it here | Access |
|---|---|---|
| **FineWeb** (English web) | https://huggingface.co/datasets/HuggingFaceFW/fineweb | public · [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) + [Common Crawl ToU](https://www.commoncrawl.org/terms-of-use) |
| **FineWeb-2** (German `deu_Latn`) | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | same license |
| **TongueSwitcher** | https://zenodo.org/records/10011601 · DOI [10.5281/zenodo.10011601](https://doi.org/10.5281/zenodo.10011601) | **restricted** — request with an academic email; Twitter/X terms; research use only |
| TongueSwitcher tagger (generation only) | https://github.com/igorsterner/TongueSwitcher | public code |
| **ACS** (German–English minimal pairs) | https://huggingface.co/datasets/igorsterner/acs-benchmark · project https://github.com/igorsterner/acs | **gated** — accept the Hub terms (`de-en` config) |
| **Denglisch** | https://github.com/HaifaCLG/Denglisch | public GitHub (no LICENSE file in that repo) |
| fastText `lid.176.bin` | https://fasttext.cc/docs/en/language-identification.html · [direct download](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin) | public |

Papers: Penedo et al. 2024/2025 (FineWeb / FineWeb-2); Sterner & Teufel 2023
(TongueSwitcher); Sterner & Teufel 2025 (ACS); Osmelak & Wintner 2023 (Denglisch).

**Do not redistribute** FineWeb shards, TongueSwitcher tweets, or ACS/Denglisch
sentence text from this bundle. We ship only `metrics/acs_pair_ids.txt` (7,397
pair ids).

## After you have the official files

- FineWeb EN + FineWeb-2 DE → SoloWeb via `soloweb/submit_download_fineweb.sh`
  then `soloweb/submit_clean_fineweb.sh` (needs `lid.176.bin` and NLTK Punkt).
- ACS sentences: `python metrics/fetch_acs_subset.py --out $CS_ROOT/data/acs_pairs.csv`
- TongueSwitcher training pool: drop tweets that contain an ACS observed
  sentence with `soloweb/prepare_ts_leakfree.py`.
- Point `paths.env` at the resulting paths (`paths.env.example`).
