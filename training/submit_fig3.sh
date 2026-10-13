#!/bin/bash
# Figure 3: SoloWeb 5x5 D_en x D_de grid at 108M (gpt2sL12). 16 train-only jobs
# (pools already exist under exp12_swd_pools; balanced diagonals come from the
# Figure 2 rungs D1B / D2p74B / D11B).
set -euo pipefail
SL="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=../_env.sh
source "$SL/../_env.sh"
SWTOK="$TOKENIZER_DIR"; MH="$MONO_HELDOUT"
SW7POOL="${SW7_POOL:-$CS_ROOT/runs/exp12_sw7_pools/D2p74B/data}"
POOLS="$CS_ROOT/runs/exp12_swd_pools"
declare -A TOK=( [0p125B]=125000000 [0p5B]=500000000 [1p37B]=1370000000 [2p75B]=2750000000 [5p5B]=5500000000 )
n=0
for en in 0p125B 0p5B 1p37B 2p75B 5p5B; do for de in 0p125B 0p5B 1p37B 2p75B 5p5B; do
  [[ "$en" == "$de" && ( "$en" == "0p5B" || "$en" == "1p37B" || "$en" == "5p5B" ) ]] && continue
  RN="exp12_swp2b_gpt2sL12_en${en}_de${de}_s1"; RUN="$CS_ROOT/runs/$RN"
  compgen -G "$RUN/models_packed/*/final/model.safetensors" >/dev/null && { echo "[skip] $RN"; continue; }
  cell="p2b_en${en}_de${de}"
  [[ -s "$POOLS/$cell/data/packed_512/bl/metadata.json" ]] || { echo "!! no pool $cell"; continue; }
  rm -rf "$RUN"; mkdir -p "$RUN/data"
  ln -sfn "$POOLS/$cell/data/packed_512" "$RUN/data/packed_512"
  ln -sfn "$SWTOK" "$RUN/data/tokenizer"
  if [[ -e "$SW7POOL/valid_denglisch_manual_cs.jsonl" ]]; then
    ln -sfn "$SW7POOL/valid_denglisch_manual_cs.jsonl" "$RUN/data/valid_denglisch_manual_cs.jsonl"
  fi
  FT=$(( TOK[$en] + TOK[$de] ))
  jt=$(sbatch_site --parsable --gres=gpu:1 --time=1-00:00:00 \
       --export=ALL,RUN_NAME=$RN,CONDITION=bl,MONO_HELDOUT=$MH,MODEL_TAG=gpt2s,N_LAYER=12,N_EMBD=768,N_HEAD=12,BLOCK_SIZE=512,FIXED_TOKENS=$FT,SEED=1,SCORE_POINTS=20,BATCH_SIZE=64,GRAD_ACCUM=1,TOKEN_BUDGET=fixed,EPOCHS=1.0 \
       "$SL/train_one_gpu.sbatch")
  echo "$RN -> $jt (FT=$((FT/1000000))M)"; n=$((n+1))
done; done
echo "submitted $n trains"
