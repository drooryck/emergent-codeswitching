#!/bin/bash
# Train the Figure 1 sizes on TongueSwitcher only, 1 epoch over the packed TS pool, 3 seeds.
# Run AFTER exp12_tsonly_pool is packed.
#   DRY=1 bash training/submit_tsonly.sh
set -euo pipefail
SL="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=../_env.sh
source "$SL/../_env.sh"
DRY=${DRY:-0}
POOL="$CS_ROOT/runs/exp12_tsonly_pool"
META="$POOL/data/packed_512/g/metadata.json"
[[ -s "$META" ]] || { echo "!! TS pool not packed yet ($META)"; exit 1; }
TSTOK=$(python3 -c "import json;print(json.load(open('$META'))['packed_tokens'])")
echo "TS pool packed_tokens = $TSTOK ($((TSTOK/1000000))M) -> FIXED_TOKENS (1 epoch)"
declare -A CFG=( [gpt2sL4]="gpt2s:4:768:12:1" [gpt2sL12]="gpt2s:12:768:12:1" [gpt2mL16]="gpt2m:16:1024:16:4" [gpt2mL24]="gpt2m:24:1024:16:4" )
SEEDS=${SEEDS:-"1 2 3"}; n=0
for size in gpt2sL4 gpt2sL12 gpt2mL16 gpt2mL24; do
  IFS=':' read -r tag nl ne nh ng <<< "${CFG[$size]}"
  for s in $SEEDS; do
    RN="exp12_tsonly_${size}_s${s}"; RUN="$CS_ROOT/runs/$RN"
    if [[ $DRY == 1 ]]; then echo "[DRY] $RN ng=$ng FIXED=$((TSTOK/1000000))M"; n=$((n+1)); continue; fi
    rm -rf "$RUN"; mkdir -p "$RUN/data"
    ln -sfn "$POOL/data/packed_512" "$RUN/data/packed_512"
    ln -sfn "$POOL/data/tokenizer" "$RUN/data/tokenizer"
    if [[ -e "$POOL/data/valid_denglisch_manual_cs.jsonl" ]]; then
      ln -sfn "$(readlink -f "$POOL/data/valid_denglisch_manual_cs.jsonl")" "$RUN/data/valid_denglisch_manual_cs.jsonl"
    fi
    C="RUN_NAME=$RN,CONDITION=g,MODEL_TAG=$tag,N_LAYER=$nl,N_EMBD=$ne,N_HEAD=$nh,BLOCK_SIZE=512,SEED=$s,SCORE_POINTS=20"
    if [[ $tag == gpt2m ]]; then
      jt=$(sbatch_site --parsable --gres=gpu:4 \
        --export=ALL,$C,BATCH_SIZE=16,GRAD_ACCUM=1,NGPUS=4,FIXED_TOKENS=$TSTOK "$SL/train_ddp.sbatch")
    else
      jt=$(sbatch_site --parsable --gres=gpu:1 --time=1-00:00:00 \
        --export=ALL,$C,BATCH_SIZE=64,GRAD_ACCUM=1,TOKEN_BUDGET=fixed,FIXED_TOKENS=$TSTOK "$SL/train_one_gpu.sbatch")
    fi
    echo "$RN -> $jt"; n=$((n+1))
  done
done
echo "tsonly: $n runs"
