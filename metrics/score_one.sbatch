#!/bin/bash
#SBATCH --job-name=score_one
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Post-hoc Switch BPB + ACS scoring of one finished checkpoint.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$_HERE/../_env.sh"

RUN_NAME=${RUN_NAME:?Set RUN_NAME}
MODEL_NAME=${MODEL_NAME:?Set MODEL_NAME e.g. gpt2s_L16}

RUN="$CS_ROOT/runs/$RUN_NAME"
MODEL_DIR="$RUN/models_packed/$MODEL_NAME/final"
mkdir -p "$RUN/tables" "$RUN/tmp"
source "$CS_ENV/bin/activate"
export TMPDIR="$RUN/tmp"

if [[ ! -d "$MODEL_DIR" ]]; then echo "Missing model: $MODEL_DIR" >&2; exit 2; fi
if [[ ! -s "${ACS_PAIRS}" ]]; then
  echo "Missing ACS pairs CSV: $ACS_PAIRS (run metrics/fetch_acs_subset.py)" >&2
  exit 2
fi

python "$CS_CODE/metrics/score_switch_bpb.py" \
  --model-dir "$MODEL_DIR" \
  --eval-jsonl "$MONO_HELDOUT/valid_monoweb_eng.jsonl" \
               "$MONO_HELDOUT/valid_monoweb_deu.jsonl" \
               "$TS_HELDOUT/valid_tongueswitcher_cs.jsonl" \
  --eval-name monoweb_eng monoweb_deu tongueswitcher \
  --train-condition "$MODEL_NAME" \
  --out-csv "$RUN/tables/switch_bpb_${MODEL_NAME}.csv"

python "$CS_CODE/metrics/score_acs_local.py" \
  --model-dir "$MODEL_DIR" \
  --pairs-csv "$ACS_PAIRS" \
  --out-csv "$RUN/tables/acs_${MODEL_NAME}.csv" \
  --model-name "$MODEL_NAME"

echo "SCORE_DONE=$MODEL_NAME"
