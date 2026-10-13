#!/bin/bash
# Stage 1: download FineWeb (en) + FineWeb-2 (de) shards from the HF Hub.
#   bash soloweb/submit_download_fineweb.sh [CONC]
# Env overrides: EN_REPO EN_PREFIX EN_SHARDS DE_REPO DE_PREFIX DE_SHARDS RAW_ROOT
set -euo pipefail
SL="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=../_env.sh
source "$SL/../_env.sh"

RAW_ROOT=${RAW_ROOT:-$CS_ROOT/data/fineweb_raw}
DL_MANIFEST=${DL_MANIFEST:-$RAW_ROOT/download_manifest.tsv}
CONC=${1:-${CONC:-24}}
mkdir -p "$RAW_ROOT"

source "$CS_ENV/bin/activate"
RAW_ROOT="$RAW_ROOT" DL_MANIFEST="$DL_MANIFEST" python "$CS_CODE/soloweb/build_download_manifest.py"
NTASKS=$(wc -l < "$DL_MANIFEST")
echo "download manifest: $NTASKS shards -> array 0-$((NTASKS-1))%$CONC"

AID=$(sbatch_site --cpu --parsable --array=0-$((NTASKS-1))%"$CONC" \
      --export=ALL,DL_MANIFEST="$DL_MANIFEST" "$SL/download_fineweb.sbatch")
echo "submitted download array $AID"
echo "when done, launch cleaning (stage 2):"
echo "  EN_GLOB='$RAW_ROOT/eng/sample/100BT/*.parquet' DE_GLOB='$RAW_ROOT/deu/data/deu_Latn/train/*.parquet' \\"
echo "    OUT_ROOT=$CS_ROOT/data/fineweb_clean bash $SL/submit_clean_fineweb.sh 256"
