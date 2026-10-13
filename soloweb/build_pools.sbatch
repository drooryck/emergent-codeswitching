#!/bin/bash
#SBATCH --job-name=build_pools
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#
# Build nested SoloWeb train pools (one streaming pass per language).
# Requires paths.env. Set OUT_ROOT and RUNGS_SC (name:tokens_per_lang;... ).
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$_HERE/../_env.sh"

OUT_ROOT=${OUT_ROOT:?set OUT_ROOT (dir holding one pool dir per rung)}
RUNGS_SC=${RUNGS_SC:?set RUNGS_SC like "D0p5B:250000000;D1B:500000000"}
RUNGS=$(echo "$RUNGS_SC" | tr ';' ' ')
UNION_TOK=${TOKENIZER_DIR}
CLEAN=${CLEAN_DIR}
DENGLISCH_VALID=${DENGLISCH_VALID:-$CS_ROOT/data/valid_denglisch_manual_cs.jsonl}

source "$CS_ENV/bin/activate"
export TOKENIZERS_PARALLELISM=true

H="$MONO_HELDOUT"
[[ -s "$H/valid_monoweb_eng.jsonl" && -s "$H/valid_monoweb_deu.jsonl" ]] || \
  python "$CS_CODE/soloweb/make_soloweb_heldout.py" --n 6000 --clean-dir "$CLEAN" --out-dir "$H"

python "$CS_CODE/soloweb/prepare_soloweb_pool.py" \
  --clean-dir "$CLEAN" --tokenizer "$UNION_TOK" \
  --out-root "$OUT_ROOT" --rungs $RUNGS --skip-docs 6000

for name in $(echo "$RUNGS" | tr ' ' '\n' | cut -d: -f1); do
  DATA="$OUT_ROOT/$name/data"
  ln -sfn "$UNION_TOK" "$DATA/tokenizer"
  if [[ -e "$DENGLISCH_VALID" ]]; then
    ln -sfn "$DENGLISCH_VALID" "$DATA/valid_denglisch_manual_cs.jsonl"
  fi
done
echo "BUILD_POOLS_DONE=$OUT_ROOT"
