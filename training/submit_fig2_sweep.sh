#!/bin/bash
# SoloWeb 7x7 (N,D) sweep (Figure 2): 7 sizes x 7 nested data rungs, p=0.5 SoloWeb, zero CS.
# seed 1 everywhere + seeds 2,3 at (52M, D2p74B). Chain: setup -> 7 packs -> trains.
#   DRY=1 bash training/submit_fig2_sweep.sh
set -euo pipefail
SL="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=../_env.sh
source "$SL/../_env.sh"
DRY=${DRY:-0}; POOLS=$CS_ROOT/runs/exp12_sw7_pools
MONO_H=$MONO_HELDOUT

RUNGS="D0p5B:250000000 D1B:500000000 D2p74B:1370000000 D5p5B:2750000000 D11B:5500000000 D24B:12000000000 D52B:26000000000"
declare -A DTOK=( [D0p5B]=500000000 [D1B]=1000000000 [D2p74B]=2740000000 [D5p5B]=5500000000 [D11B]=11000000000 [D24B]=24000000000 [D52B]=52000000000 )
declare -A CFG=( [t11]="gpt2s:4:256:4:64:1:1"   [t22]="gpt2s:6:384:6:64:1:1"
                 [gpt2sL4]="gpt2s:4:768:12:64:1:1" [gpt2sL12]="gpt2s:12:768:12:64:1:1"
                 [gpt2mL16]="gpt2m:16:1024:16:16:1:4"
                 [gpt2lL24]="gpt2l:24:1280:20:8:2:4" [gpt2xlL25]="gpt2xl:25:1600:25:4:4:4" )
SIZES="t11 t22 gpt2sL4 gpt2sL12 gpt2mL16 gpt2lL24 gpt2xlL25"

sub(){ [[ $DRY == 1 ]] && { echo "[DRY] sbatch $*" >&2; echo $((RANDOM+900000)); return; }
      sbatch_site --parsable "$@"; }

RUNGS_SC=$(echo "$RUNGS" | tr ' ' ';')
DEP=""; [[ -n "${SETUP_DEP:-}" ]] && DEP="--dependency=afterok:$SETUP_DEP"
TOKD="$TOKENIZER_DIR"
if [[ "${SKIP_SETUP:-0}" == 1 ]]; then
  js="none"; echo "setup skipped (pools exist)"
else
  js=$(sub --cpu $DEP --export=ALL,OUT_ROOT=$POOLS,RUNGS_SC="$RUNGS_SC",TOKENIZER_DIR="$TOKD" \
       "$SL/../soloweb/build_pools.sbatch")
  echo "setup -> $js"
fi

declare -A PACK
for r in $RUNGS; do name=${r%%:*}
  PDEP=""; [[ "$js" != "none" ]] && PDEP="--dependency=afterok:$js"
  jk=$(sub --cpu $PDEP --time=1-00:00:00 \
       --export=ALL,RUN_NAME=exp12_sw7_pools/$name,CONDITION=sw,BLOCK_SIZE=512 "$SL/pack.sbatch")
  PACK[$name]=$jk; echo "pack $name -> $jk"
done

n=0
for size in $SIZES; do
  IFS=':' read -r tag nl ne nh bs ga ng <<< "${CFG[$size]}"
  for r in $RUNGS; do name=${r%%:*}; FT=${DTOK[$name]}
    SEEDS="1"; [[ $size == gpt2sL4 && $name == D2p74B ]] && SEEDS="1 2 3"
    for s in $SEEDS; do
      RN="exp12_sw7_${size}_${name}_s${s}"; RUN="$CS_ROOT/runs/$RN"
      if compgen -G "$RUN/models_packed/*/final/model.safetensors" >/dev/null; then echo "[skip] $RN"; continue; fi
      if [[ $DRY != 1 ]]; then
        rm -rf "$RUN"; mkdir -p "$RUN/data"
        ln -sfn "$POOLS/$name/data/packed_512" "$RUN/data/packed_512"
        ln -sfn "$POOLS/$name/data/tokenizer"  "$RUN/data/tokenizer"
        if [[ -e "$POOLS/$name/data/valid_denglisch_manual_cs.jsonl" ]]; then
          ln -sfn "$POOLS/$name/data/valid_denglisch_manual_cs.jsonl" "$RUN/data/valid_denglisch_manual_cs.jsonl"
        fi
      fi
      C="RUN_NAME=$RN,CONDITION=sw,MONO_HELDOUT=$MONO_H"
      COMMON="MODEL_TAG=$tag,N_LAYER=$nl,N_EMBD=$ne,N_HEAD=$nh,BLOCK_SIZE=512,FIXED_TOKENS=$FT,SEED=$s,SCORE_POINTS=20"
      ngc=$ng; bsc=$bs
      if [[ $size == gpt2sL12 && ( $name == D24B || $name == D52B ) ]]; then ngc=4; bsc=16; fi
      if [[ $ngc == 1 ]]; then
        T=1-00:00:00; [[ $name == D24B || $name == D52B ]] && T=2-00:00:00
        jt=$(sub --dependency=afterok:${PACK[$name]} --gres=gpu:1 --time=$T \
             --export=ALL,$C,$COMMON,BATCH_SIZE=$bsc,GRAD_ACCUM=$ga,TOKEN_BUDGET=fixed,EPOCHS=1.0 \
             "$SL/train_one_gpu.sbatch")
      else
        T=2-00:00:00
        [[ $name == D24B || $name == D52B ]] && T=3-00:00:00
        jt=$(sub --dependency=afterok:${PACK[$name]} --gres=gpu:4 --time=$T \
             --export=ALL,$C,$COMMON,BATCH_SIZE=$bsc,GRAD_ACCUM=$ga,NGPUS=4 \
             "$SL/train_ddp.sbatch")
      fi
      echo "$RN -> $jt (FT=$((FT/1000000))M)"; n=$((n+1))
    done
  done
done
echo "fig2: $n trains submitted (setup $js)"
