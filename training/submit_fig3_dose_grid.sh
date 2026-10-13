#!/bin/bash
# Dose factorial at 108M: D_en x D_de in {0.5, 1.37, 5.5}B  x  dose in
# {0, 0.1, 1, 10, 100}% of the leak-free TongueSwitcher pool.
# TDE/TEN come from $CS_ROOT/data/ts_prefix_tags/CONSTANTS.json
# (run generation/ts_prefix_constants.py first).
#   DRY=1 bash training/submit_fig3_dose_grid.sh
set -euo pipefail
SL="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=../_env.sh
source "$SL/../_env.sh"
SWTOK="$TOKENIZER_DIR"
MH="$MONO_HELDOUT"
SW7POOL="${SW7_POOL:-$CS_ROOT/runs/exp12_sw7_pools/D2p74B/data}"
POOLS="$CS_ROOT/runs/exp12_swfact_pools"
CLEAN="$CLEAN_DIR"
TS=${TONGUESWITCHER:-$CS_ROOT/data/tongueswitcher_train_leakfree.jsonl}
CONST=$CS_ROOT/data/ts_prefix_tags/CONSTANTS.json
DRY=${DRY:-0}; n=0; nskip=0
sub(){ [[ $DRY == 1 ]] && { echo "[DRY] sbatch $*" >&2; echo $((RANDOM+800000)); return; }; sbatch_site --parsable "$@"; }

[[ -s $CONST ]] || { echo "missing $CONST — run generation/ts_prefix_constants.py first" >&2; exit 2; }
FULL=983519712
DOSES=(0 983520 9835197 98351971 $FULL)
declare -A TDE TEN CSTAG=( [0]=cs0 [983520]=cs0p98M [9835197]=cs9p8M [98351971]=cs98M [$FULL]=csFULL )
TDE[0]=0; TEN[0]=0
while read -r d tde ten; do TDE[$d]=$tde; TEN[$d]=$ten; done < <("$CS_ENV/bin/python" - <<EOF
import json
c = json.load(open("$CONST"))
for d in (983520, 9835197, 98351971, $FULL):
    print(d, c[str(d)]["TDE"], c[str(d)]["TEN"])
EOF
)

declare -A BTAG=( [500000000]=0p5B [1370000000]=1p37B [5500000000]=5p5B )
covered(){
  local en=$1 de=$2 cs=$3
  [[ $en != $de ]] && return 1
  if [[ $cs == 0 ]]; then return 0; fi
  if [[ $en == 1370000000 && ( $cs == 9835197 || $cs == 98351971 || $cs == $FULL ) ]]; then return 0; fi
  if [[ $en == 5500000000 && ( $cs == 98351971 || $cs == $FULL ) ]]; then return 0; fi
  return 1
}

for EN_ in 500000000 1370000000 5500000000; do
  for DE_ in 500000000 1370000000 5500000000; do
    for CS_ in "${DOSES[@]}"; do
      CE_=$(( EN_ - TEN[$CS_] )); CD_=$(( DE_ - TDE[$CS_] ))
      if (( CE_ < 0 || CD_ < 0 )); then echo "[infeasible] en=${BTAG[$EN_]} de=${BTAG[$DE_]} ${CSTAG[$CS_]}"; nskip=$((nskip+1)); continue; fi
      if covered "$EN_" "$DE_" "$CS_"; then echo "[covered] en=${BTAG[$EN_]} de=${BTAG[$DE_]} ${CSTAG[$CS_]}"; nskip=$((nskip+1)); continue; fi
      cell="fact_en${BTAG[$EN_]}_de${BTAG[$DE_]}_${CSTAG[$CS_]}"
      cond=dose; [[ $CS_ == 0 ]] && cond=bl
      RN="exp12_swfact_gpt2sL12_en${BTAG[$EN_]}_de${BTAG[$DE_]}_${CSTAG[$CS_]}_s1"
      RUN="$CS_ROOT/runs/$RN"; P="$POOLS/$cell"; FT=$(( EN_ + DE_ ))
      if compgen -G "$RUN/models_packed/*/final/model.safetensors" >/dev/null; then echo "[skip done] $RN"; continue; fi
      if [[ $DRY != 1 ]]; then
        mkdir -p "$P/data"
        ln -sfn "$SWTOK" "$P/data/tokenizer"
        rm -rf "$RUN"; mkdir -p "$RUN/data"
        ln -sfn "$P/data/packed_512" "$RUN/data/packed_512"
        ln -sfn "$SWTOK" "$RUN/data/tokenizer"
        if [[ -e "$SW7POOL/valid_denglisch_manual_cs.jsonl" ]]; then
          ln -sfn "$SW7POOL/valid_denglisch_manual_cs.jsonl" "$RUN/data/valid_denglisch_manual_cs.jsonl"
        fi
      fi
      TS_ARG=""; [[ $CS_ -gt 0 ]] && TS_ARG="--tongueswitcher $TS"
      js=$(sub --cpu --cpus-per-task=16 --mem=64G --time=12:00:00 --job-name=fig3_setup \
           --wrap "source $CS_ENV/bin/activate && export TOKENIZERS_PARALLELISM=true && python $CS_CODE/soloweb/prepare_sw_biling_pool.py --run-dir $P --condition $cond --tokens-eng $CE_ --tokens-deu $CD_ --tokens-cs $CS_ --tokenizer $SWTOK --clean-dir $CLEAN $TS_ARG")
      jk=$(sub --cpu --dependency=afterok:$js --time=1-00:00:00 \
           --export=ALL,RUN_NAME=exp12_swfact_pools/$cell,CONDITION=$cond,BLOCK_SIZE=512 "$SL/pack.sbatch")
      jt=$(sub --dependency=afterok:$jk --gres=gpu:1 --time=1-00:00:00 \
           --export=ALL,RUN_NAME=$RN,CONDITION=$cond,MONO_HELDOUT=$MH,MODEL_TAG=gpt2s,N_LAYER=12,N_EMBD=768,N_HEAD=12,BLOCK_SIZE=512,FIXED_TOKENS=$FT,SEED=1,SCORE_POINTS=20,BATCH_SIZE=64,GRAD_ACCUM=1,TOKEN_BUDGET=fixed,EPOCHS=1.0 \
           "$SL/train_one_gpu.sbatch")
      echo "$RN: setup $js -> pack $jk -> train $jt (clean en=$((CE_/1000000))M de=$((CD_/1000000))M cs=$((CS_/1000000))M FT=$((FT/1000000))M)"
      n=$((n+1))
    done
  done
done
echo "fig3 dose grid: $n chains submitted, $nskip skipped (infeasible/covered)"
