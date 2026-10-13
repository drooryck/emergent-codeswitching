#!/bin/bash
# Figure 4 family: SoloWeb CS-injection sweeps (shared pools under exp12_swd_pools).
#   A) D_en x D_de grid at 52M
#   B) composition-compensated dose ladder
#   C) dose x composition cross at 52M, D=2.74B
#   DRY=1 bash training/submit_fig4.sh
set -euo pipefail
SL="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=../_env.sh
source "$SL/../_env.sh"
SWTOK="$TOKENIZER_DIR"
MH="$MONO_HELDOUT"
SW7POOL="${SW7_POOL:-$CS_ROOT/runs/exp12_sw7_pools/D2p74B/data}"
POOLS="$CS_ROOT/runs/exp12_swd_pools"
CLEAN="$CLEAN_DIR"
TS=${TONGUESWITCHER:-$CS_ROOT/data/tongueswitcher_train_leakfree.jsonl}
DRY=${DRY:-0}; n=0; npool=0
sub(){ [[ $DRY == 1 ]] && { echo "[DRY] sbatch $*" >&2; echo $((RANDOM+700000)); return; }; sbatch_site --parsable "$@"; }

declare -A CFG=( [gpt2sL4]="gpt2s:4:768:12:64:1:1" [gpt2sL12]="gpt2s:12:768:12:64:1:1"
                 [gpt2mL16]="gpt2m:16:1024:16:16:1:4" )
FULL=961656924
declare -A TDE=( [9600000]=8173069   [24000000]=20449444  [48000000]=41270102 [96000000]=83587632
                 [240000000]=208433879 [480000000]=417757909 [$FULL]=838614141 )
declare -A TEN=( [9600000]=1426931   [24000000]=3550556   [48000000]=6729898  [96000000]=12412368
                 [240000000]=31566121  [480000000]=62242091  [$FULL]=123042783 )
declare -A CSTAG=( [9600000]=cs9p6M [24000000]=cs24M [48000000]=cs48M [96000000]=cs96M
                   [240000000]=cs240M [480000000]=cs480M [$FULL]=csFULL )

declare -A PACKJOB
ensure_pool(){ # cell cond ENG DEU CS
  local cell=$1 cond=$2 e=$3 d=$4 c=$5
  if [[ -n "${PACKJOB[$cell]:-}" ]]; then return; fi
  local P="$POOLS/$cell"
  if [[ -s "$P/data/packed_512/$cond/metadata.json" ]]; then PACKJOB[$cell]=done; echo "[pool done] $cell"; return; fi
  if [[ $DRY != 1 ]]; then
    mkdir -p "$P/data"
    ln -sfn "$SWTOK" "$P/data/tokenizer"
    if [[ -e "$SW7POOL/valid_denglisch_manual_cs.jsonl" ]]; then
      ln -sfn "$SW7POOL/valid_denglisch_manual_cs.jsonl" "$P/data/valid_denglisch_manual_cs.jsonl"
    fi
  fi
  local js jk
  TS_ARG=""; [[ $c -gt 0 ]] && TS_ARG="--tongueswitcher $TS"
  js=$(sub --cpu --cpus-per-task=16 --mem=64G --time=12:00:00 --job-name=fig4_setup \
       --wrap "source $CS_ENV/bin/activate && export TOKENIZERS_PARALLELISM=true && python $CS_CODE/soloweb/prepare_sw_biling_pool.py --run-dir $P --condition $cond --tokens-eng $e --tokens-deu $d --tokens-cs $c --tokenizer $SWTOK --clean-dir $CLEAN $TS_ARG")
  jk=$(sub --cpu --dependency=afterok:$js --time=1-00:00:00 \
       --export=ALL,RUN_NAME=exp12_swd_pools/$cell,CONDITION=$cond,BLOCK_SIZE=512 "$SL/pack.sbatch")
  PACKJOB[$cell]=$jk; npool=$((npool+1))
  echo "pool $cell: setup $js -> pack $jk (en=$((e/1000000))M de=$((d/1000000))M cs=$((c/1000000))M)"
}

launch_train(){ # RN cell cond size FT seed
  local RN=$1 cell=$2 cond=$3 size=$4 FT=$5 seed=$6
  IFS=':' read -r tag nl ne nh bs ga ng <<< "${CFG[$size]}"
  local RUN="$CS_ROOT/runs/$RN"
  if compgen -G "$RUN/models_packed/*/final/model.safetensors" >/dev/null; then echo "[skip] $RN"; return; fi
  if [[ $DRY != 1 ]]; then
    rm -rf "$RUN"; mkdir -p "$RUN/data"
    ln -sfn "$POOLS/$cell/data/packed_512" "$RUN/data/packed_512"
    ln -sfn "$SWTOK" "$RUN/data/tokenizer"
    if [[ -e "$SW7POOL/valid_denglisch_manual_cs.jsonl" ]]; then
      ln -sfn "$SW7POOL/valid_denglisch_manual_cs.jsonl" "$RUN/data/valid_denglisch_manual_cs.jsonl"
    fi
  fi
  local DEP=""; [[ "${PACKJOB[$cell]}" != done ]] && DEP="--dependency=afterok:${PACKJOB[$cell]}"
  local C="RUN_NAME=$RN,CONDITION=$cond,MONO_HELDOUT=$MH"
  local COMMON="MODEL_TAG=$tag,N_LAYER=$nl,N_EMBD=$ne,N_HEAD=$nh,BLOCK_SIZE=512,FIXED_TOKENS=$FT,SEED=$seed,SCORE_POINTS=20"
  local jt
  if [[ $ng == 1 ]]; then
    jt=$(sub $DEP --gres=gpu:1 --time=1-00:00:00 \
         --export=ALL,$C,$COMMON,BATCH_SIZE=$bs,GRAD_ACCUM=$ga,TOKEN_BUDGET=fixed,EPOCHS=1.0 \
         "$SL/train_one_gpu.sbatch")
  else
    jt=$(sub $DEP --gres=gpu:4 --time=2-00:00:00 \
         --export=ALL,$C,$COMMON,BATCH_SIZE=$bs,GRAD_ACCUM=$ga,NGPUS=4 "$SL/train_ddp.sbatch")
  fi
  echo "$RN -> $jt (cell=$cell FT=$((FT/1000000))M seed=$seed)"; n=$((n+1))
}

echo "### A) D_en x D_de at 52M (zero dose) ###"
declare -A BTAG=( [125000000]=0p125B [500000000]=0p5B [1370000000]=1p37B [2750000000]=2p75B )
for EN_ in 125000000 500000000 1370000000 2750000000; do
  for DE_ in 125000000 500000000 1370000000 2750000000; do
    cell="p2b_en${BTAG[$EN_]}_de${BTAG[$DE_]}"
    ensure_pool "$cell" bl "$EN_" "$DE_" 0
    launch_train "exp12_swp2b_gpt2sL4_en${BTAG[$EN_]}_de${BTAG[$DE_]}_s1" "$cell" bl gpt2sL4 $((EN_+DE_)) 1
  done
done

echo "### B) compensated dose ladder (p=0.5) ###"
declare -A DTAG=( [2740000000]=D2p74B [11000000000]=D11B )
declare -A DOSES=( [2740000000]="9600000 24000000 48000000 96000000 240000000 480000000 $FULL"
                   [11000000000]="24000000 96000000 240000000 $FULL" )
for D_ in 2740000000 11000000000; do
  for CS_ in ${DOSES[$D_]}; do
    ENG_=$(( D_/2 - TEN[$CS_] )); DEU_=$(( D_/2 - TDE[$CS_] ))
    cell="dose_${DTAG[$D_]}_${CSTAG[$CS_]}"
    ensure_pool "$cell" dose "$ENG_" "$DEU_" "$CS_"
    for size in gpt2sL4 gpt2sL12 gpt2mL16; do
      SEEDS_="1"; [[ $size == gpt2sL4 && $D_ == 2740000000 && $CS_ == 9600000 ]] && SEEDS_="1 2 3"
      for s_ in $SEEDS_; do
        launch_train "exp12_swdose_${size}_${DTAG[$D_]}_${CSTAG[$CS_]}_s${s_}" "$cell" dose "$size" "$D_" "$s_"
      done
    done
  done
done

echo "### C) dose x composition at 52M, D=2.74B ###"
DX=2740000000
for g_ in 25 75; do
  for CS_ in 0 24000000 96000000 480000000; do
    if [[ $CS_ == 0 ]]; then
      ENG_=$(( DX*(100-g_)/100 )); DEU_=$(( DX*g_/100 )); cond=bl; ctag=cs0
    else
      ENG_=$(( DX*(100-g_)/100 - TEN[$CS_] )); DEU_=$(( DX*g_/100 - TDE[$CS_] )); cond=dose; ctag=${CSTAG[$CS_]}
    fi
    cell="cross_D2p74B_g${g_}_${ctag}"
    ensure_pool "$cell" $cond "$ENG_" "$DEU_" "$CS_"
    launch_train "exp12_swcross_gpt2sL4_D2p74B_g${g_}_${ctag}_s1" "$cell" $cond gpt2sL4 "$DX" 1
  done
done
echo "total: $n trains, $npool new pools"
