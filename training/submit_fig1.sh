#!/bin/bash
# SoloWeb fraction-of-Chinchilla ladder (Figure 1):
# 4 sizes x 7 fractions {1/20,1/10,1/5,1/4,1/3,1/2,1} of D = 20*N_total, 50/50 SoloWeb, zero CS.
# Seeds: 3 at 52M/108M, 1 at 232M/334M.
#   DRY=1 bash training/submit_fig1.sh
set -euo pipefail
SL="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=../_env.sh
source "$SL/../_env.sh"
SWTOK="$TOKENIZER_DIR"
MH="$MONO_HELDOUT"
SW7POOL="${SW7_POOL:-$CS_ROOT/runs/exp12_sw7_pools/D2p74B/data}"
POOLS="$CS_ROOT/runs/exp12_swd_pools"
CLEAN="$CLEAN_DIR"
DRY=${DRY:-0}; n=0; npool=0
sub(){ [[ $DRY == 1 ]] && { echo "[DRY] sbatch $*" >&2; echo $((RANDOM+900000)); return; }; sbatch_site --parsable "$@"; }

# size -> MODEL_TAG:N_LAYER:N_EMBD:N_HEAD:BS:GA:NGPUS:N_TOTAL:SEEDS
declare -A CFG=( [gpt2sL4]="gpt2s:4:768:12:64:1:1:51400000:1 2 3"
                 [gpt2sL12]="gpt2s:12:768:12:64:1:1:108100000:1 2 3"
                 [gpt2mL16]="gpt2m:16:1024:16:16:1:4:232300000:1"
                 [gpt2mL24]="gpt2m:24:1024:16:16:1:4:333000000:1" )
FR=( "f1o20:20" "f1o10:10" "f1o5:5" "f1o4:4" "f1o3:3" "f1o2:2" "f1x:1" )

declare -A PACKJOB
ensure_pool(){ # cell ENG DEU
  local cell=$1 e=$2 d=$3
  if [[ -n "${PACKJOB[$cell]:-}" ]]; then return; fi
  local P="$POOLS/$cell"
  if [[ -s "$P/data/packed_512/bl/metadata.json" ]]; then PACKJOB[$cell]=done; echo "[pool done] $cell"; return; fi
  if [[ $DRY != 1 ]]; then
    mkdir -p "$P/data"
    ln -sfn "$SWTOK" "$P/data/tokenizer"
    if [[ -e "$SW7POOL/valid_denglisch_manual_cs.jsonl" ]]; then
      ln -sfn "$SW7POOL/valid_denglisch_manual_cs.jsonl" "$P/data/valid_denglisch_manual_cs.jsonl"
    fi
  fi
  local js jk
  js=$(sub --cpu --cpus-per-task=16 --mem=64G --time=12:00:00 --job-name=fig1_setup \
       --wrap "source $CS_ENV/bin/activate && export TOKENIZERS_PARALLELISM=true && python $CS_CODE/soloweb/prepare_sw_biling_pool.py --run-dir $P --condition bl --tokens-eng $e --tokens-deu $d --tokens-cs 0 --tokenizer $SWTOK --clean-dir $CLEAN")
  jk=$(sub --cpu --dependency=afterok:$js --time=1-00:00:00 \
       --export=ALL,RUN_NAME=exp12_swd_pools/$cell,CONDITION=bl,BLOCK_SIZE=512 "$SL/pack.sbatch")
  PACKJOB[$cell]=$jk; npool=$((npool+1))
  echo "pool $cell: setup $js -> pack $jk (en=$((e/1000000))M de=$((d/1000000))M)"
}

for size in gpt2sL4 gpt2sL12 gpt2mL16 gpt2mL24; do
  IFS=':' read -r tag nl ne nh bs ga ng NT SEEDS <<< "${CFG[$size]}"
  for fr in "${FR[@]}"; do
    ftag=${fr%%:*}; den=${fr##*:}
    D=$(( 20 * NT / den )); D=$(( (D + 500000) / 1000000 * 1000000 ))   # round to 1M
    EN_=$((D/2)); DE_=$((D - EN_))
    cell="p0sw_${size}_${ftag}"
    ensure_pool "$cell" "$EN_" "$DE_"
    for s_ in $SEEDS; do
      RN="exp12_swp0_${size}_${ftag}_s${s_}"
      RUN="$CS_ROOT/runs/$RN"
      if compgen -G "$RUN/models_packed/*/final/model.safetensors" >/dev/null; then echo "[skip] $RN"; continue; fi
      if [[ $DRY != 1 ]]; then
        rm -rf "$RUN"; mkdir -p "$RUN/data"
        ln -sfn "$POOLS/$cell/data/packed_512" "$RUN/data/packed_512"
        ln -sfn "$SWTOK" "$RUN/data/tokenizer"
        if [[ -e "$SW7POOL/valid_denglisch_manual_cs.jsonl" ]]; then
          ln -sfn "$SW7POOL/valid_denglisch_manual_cs.jsonl" "$RUN/data/valid_denglisch_manual_cs.jsonl"
        fi
      fi
      DEP=""; [[ "${PACKJOB[$cell]}" != done ]] && DEP="--dependency=afterok:${PACKJOB[$cell]}"
      C="RUN_NAME=$RN,CONDITION=bl,MONO_HELDOUT=$MH"
      COMMON="MODEL_TAG=$tag,N_LAYER=$nl,N_EMBD=$ne,N_HEAD=$nh,BLOCK_SIZE=512,FIXED_TOKENS=$D,SEED=$s_,SCORE_POINTS=20"
      if [[ $ng == 1 ]]; then
        jt=$(sub $DEP --gres=gpu:1 --time=1-00:00:00 \
             --export=ALL,$C,$COMMON,BATCH_SIZE=$bs,GRAD_ACCUM=$ga,TOKEN_BUDGET=fixed,EPOCHS=1.0 \
             "$SL/train_one_gpu.sbatch")
      else
        jt=$(sub $DEP --gres=gpu:4 --time=2-00:00:00 \
             --export=ALL,$C,$COMMON,BATCH_SIZE=$bs,GRAD_ACCUM=$ga,NGPUS=4 \
             "$SL/train_ddp.sbatch")
      fi
      echo "$RN -> $jt (D=$((D/1000000))M seed=$s_ ${ng}gpu)"; n=$((n+1))
    done
  done
done
echo "total: $n trains, $npool new pools"
