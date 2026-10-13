#!/bin/bash
#SBATCH --job-name=pack
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#
# Pack a run's document splits into 512-token blocks.
# Requires paths.env (CS_ROOT, CS_ENV). Override partition/account at submit.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$_HERE/../_env.sh"

RUN_NAME=${RUN_NAME:?Set RUN_NAME}
CONDITION=${CONDITION:?Set CONDITION}
BLOCK_SIZE=${BLOCK_SIZE:-512}

RUN="$CS_ROOT/runs/$RUN_NAME"
DATA="$RUN/data"
mkdir -p "$RUN/tmp"
source "$CS_ENV/bin/activate"
export TMPDIR="$RUN/tmp"
export TOKENIZERS_PARALLELISM=true

python "$CS_CODE/training/pack_blocks.py" \
  --splits-dir "$DATA/splits" \
  --tokenizer-dir "$DATA/tokenizer" \
  --out-dir "$DATA/packed_${BLOCK_SIZE}" \
  --block-size "$BLOCK_SIZE" \
  --condition "$CONDITION" \
  --skip-existing

echo "PACK_DONE=$DATA/packed_${BLOCK_SIZE}/$CONDITION"
