#!/bin/bash
#SBATCH --job-name=fw_clean_lid
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --requeue
#
# One array task per (shard, row-group chunk). Submit via submit_clean_fineweb.sh.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$_HERE/../_env.sh"

MANIFEST=${MANIFEST:-$CS_ROOT/data/fineweb_clean/clean_manifest.tsv}
export MINCHARS=${MINCHARS:-25}
export CONF=${CONF:-0.50}
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
source "$CS_ENV/bin/activate"

LINE=$(awk -F'\t' -v k="$SLURM_ARRAY_TASK_ID" '$1==k {print; exit}' "$MANIFEST")
if [[ -z "$LINE" ]]; then echo "no manifest line for idx $SLURM_ARRAY_TASK_ID"; exit 1; fi
IFS=$'\t' read -r IDX LANG FMT INPUT START END OUT STATS <<< "$LINE"

echo "[task $SLURM_ARRAY_TASK_ID] lang=$LANG fmt=$FMT $INPUT rg[$START:$END]"
python "$CS_CODE/soloweb/clean_fineweb_lid.py" \
    --lang "$LANG" --fmt "$FMT" --input "$INPUT" --start "$START" --end "$END" \
    --out "$OUT" --stats "$STATS" --minchars "$MINCHARS" --conf "$CONF"
