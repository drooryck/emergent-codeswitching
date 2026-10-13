#!/bin/bash
# Stage 2: sentence-level LID cleaning of downloaded FineWeb shards.
#   bash soloweb/submit_clean_fineweb.sh [CONC]
# Env overrides: EN_GLOB DE_GLOB OUT_ROOT CHUNK_RG MINCHARS CONF CONCAT
set -euo pipefail
SL="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=../_env.sh
source "$SL/../_env.sh"

OUT_ROOT=${OUT_ROOT:-$CS_ROOT/data/fineweb_clean}
MANIFEST=${MANIFEST:-$OUT_ROOT/clean_manifest.tsv}
CONC=${1:-${CONC:-256}}
CONCAT=${CONCAT:-1}
mkdir -p "$OUT_ROOT"

source "$CS_ENV/bin/activate"
OUT_ROOT="$OUT_ROOT" MANIFEST="$MANIFEST" python "$CS_CODE/soloweb/build_clean_manifest.py"
NTASKS=$(wc -l < "$MANIFEST")
echo "manifest: $NTASKS tasks -> array 0-$((NTASKS-1))%$CONC"

AID=$(sbatch_site --cpu --parsable --array=0-$((NTASKS-1))%"$CONC" \
      --export=ALL,MANIFEST="$MANIFEST",MINCHARS="${MINCHARS:-25}",CONF="${CONF:-0.50}" \
      "$SL/clean_fineweb_lid.sbatch")
echo "submitted array job $AID"

CONCAT_FLAG=""; [[ "$CONCAT" == "1" ]] && CONCAT_FLAG="--concat"
RID=$(sbatch_site --cpu --parsable --cpus-per-task=4 --mem=16G --time=02:00:00 \
      --job-name=fw_clean_reduce --dependency=afterok:"$AID" \
      --wrap "source $CS_ENV/bin/activate; OUT_ROOT=$OUT_ROOT MANIFEST=$MANIFEST python $CS_CODE/soloweb/reduce_clean_manifest.py $CONCAT_FLAG")
echo "submitted reduce job $RID (afterok:$AID)"
echo "watch:  sacct -j $AID,$RID  ;  results: $OUT_ROOT/MANIFEST.md"
