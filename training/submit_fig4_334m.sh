#!/bin/bash
# 334M (gpt2mL24) compensated dose ladder at D=2.74B — fourth size on Figure 4.
# 7 dose runs on the shared exp12_swd_pools/dose_D2p74B_cs* pools (CONDITION=dose)
# + 1 zero-dose anchor on exp12_sw7_pools/D2p74B (CONDITION=sw).
set -euo pipefail
SL="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=../_env.sh
source "$SL/../_env.sh"
SWTOK="$TOKENIZER_DIR"; MH="$MONO_HELDOUT"
DENGLISCH_VALID=${DENGLISCH_VALID:-$CS_ROOT/data/valid_denglisch_manual_cs.jsonl}
FIXED=2740000000
launch(){ # RN POOLDIR CONDITION
  local RN=$1 POOL=$2 COND=$3 RUN=$CS_ROOT/runs/$1
  if compgen -G "$RUN/models_packed/*/final" >/dev/null; then echo "[skip] $RN"; return; fi
  rm -rf "$RUN"; mkdir -p "$RUN/data"
  ln -sfn "$POOL/data/packed_512" "$RUN/data/packed_512"
  ln -sfn "$SWTOK" "$RUN/data/tokenizer"
  if [[ -e "$DENGLISCH_VALID" ]]; then
    ln -sfn "$DENGLISCH_VALID" "$RUN/data/valid_denglisch_manual_cs.jsonl"
  fi
  C="RUN_NAME=$RN,CONDITION=$COND,MONO_HELDOUT=$MH"
  C="$C,MODEL_TAG=gpt2m,N_LAYER=24,N_EMBD=1024,N_HEAD=16,BLOCK_SIZE=512"
  C="$C,SEED=1,SCORE_POINTS=20,FIXED_TOKENS=$FIXED"
  jt=$(sbatch_site --parsable --gres=gpu:4 --time=1-00:00:00 \
    --export=ALL,$C,BATCH_SIZE=16,GRAD_ACCUM=1,NGPUS=4 \
    "$SL/train_ddp.sbatch")
  echo "$RN -> $jt"
}
for d in cs9p6M cs24M cs48M cs96M cs240M cs480M csFULL; do
  launch "exp12_swdose_gpt2mL24_D2p74B_${d}_s1" "$CS_ROOT/runs/exp12_swd_pools/dose_D2p74B_$d" dose
done
launch "exp12_sw7_gpt2mL24_D2p74B_s1" "$CS_ROOT/runs/exp12_sw7_pools/D2p74B" sw
