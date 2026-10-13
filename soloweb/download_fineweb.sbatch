#!/bin/bash
#SBATCH --job-name=fw_download
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --requeue
#
# One array task per FineWeb / FineWeb-2 shard. Submit via submit_download_fineweb.sh.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$_HERE/../_env.sh"

DL_MANIFEST=${DL_MANIFEST:-$CS_ROOT/data/fineweb_raw/download_manifest.tsv}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
source "$CS_ENV/bin/activate"

LINE=$(awk -F'\t' -v k="$SLURM_ARRAY_TASK_ID" '$1==k {print; exit}' "$DL_MANIFEST")
if [[ -z "$LINE" ]]; then echo "no manifest line for idx $SLURM_ARRAY_TASK_ID"; exit 1; fi
IFS=$'\t' read -r IDX LANG DLREPO REMOTE ROOT <<< "$LINE"
echo "[task $SLURM_ARRAY_TASK_ID] $LANG $DLREPO :: $REMOTE -> $ROOT"
python "$CS_CODE/soloweb/download_fineweb_shard.py" --repo "$DLREPO" --remote "$REMOTE" --local-root "$ROOT"
