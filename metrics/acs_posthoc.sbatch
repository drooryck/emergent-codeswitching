#!/bin/bash
#SBATCH --job-name=acs_posthoc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-7
#
# Post-hoc ACS scoring of finished checkpoints listed in RUN_LIST
# (two columns: <final_ckpt_dir> <name>). Idempotent.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$_HERE/../_env.sh"

source "$CS_ENV/bin/activate"
LIST=${RUN_LIST:-$CS_ROOT/acs_posthoc/run_list.txt}
OUT_DIR=${ACS_OUT_DIR:-$CS_ROOT/acs_posthoc}
mkdir -p "$OUT_DIR"
[[ -s "$LIST" ]] || { echo "Missing run list: $LIST" >&2; exit 2; }
[[ -s "$ACS_PAIRS" ]] || { echo "Missing ACS pairs: $ACS_PAIRS" >&2; exit 2; }

N=$(wc -l < "$LIST")
i=0
while read -r FINAL NAME; do
  if (( i % 8 == SLURM_ARRAY_TASK_ID )); then
    OUT="$OUT_DIR/${NAME}.csv"
    if [[ -s "$OUT" ]]; then
      echo "SKIP $NAME (exists)"
    else
      echo "SCORING $NAME"
      python "$CS_CODE/metrics/score_acs_local.py" \
        --model-dir "$FINAL" --pairs-csv "$ACS_PAIRS" \
        --out-csv "$OUT" --model-name "$NAME"
    fi
  fi
  i=$((i+1))
done < "$LIST"
echo "shard $SLURM_ARRAY_TASK_ID done ($N total entries)"
