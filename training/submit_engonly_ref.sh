#!/bin/bash
# SoloWeb English-only reference for Figure 1: 52M, 1x Chinchilla (1.028B tok),
# 99.9% SoloWeb-EN / 0.1% DE, zero CS.
set -euo pipefail
SL="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# shellcheck source=../_env.sh
source "$SL/../_env.sh"
SWTOK="$TOKENIZER_DIR"
MH="$MONO_HELDOUT"
SW7POOL="${SW7_POOL:-$CS_ROOT/runs/exp12_sw7_pools/D2p74B/data}"
CLEAN="$CLEAN_DIR"
D=1028000000; DE_=1000000; EN_=$((D-DE_))
P=$CS_ROOT/runs/exp12_swd_pools/p0sw_engonly_f1x
mkdir -p "$P/data"
ln -sfn "$SWTOK" "$P/data/tokenizer"
if [[ -e "$SW7POOL/valid_denglisch_manual_cs.jsonl" ]]; then
  ln -sfn "$SW7POOL/valid_denglisch_manual_cs.jsonl" "$P/data/valid_denglisch_manual_cs.jsonl"
fi
js=$(sbatch_site --cpu --parsable --cpus-per-task=16 --mem=64G --time=12:00:00 \
  --job-name=engonly_setup \
  --wrap "source $CS_ENV/bin/activate && export TOKENIZERS_PARALLELISM=true && python $CS_CODE/soloweb/prepare_sw_biling_pool.py --run-dir $P --condition bl --tokens-eng $EN_ --tokens-deu $DE_ --tokens-cs 0 --tokenizer $SWTOK --clean-dir $CLEAN")
jk=$(sbatch_site --cpu --parsable --dependency=afterok:$js --time=1-00:00:00 \
  --export=ALL,RUN_NAME=exp12_swd_pools/p0sw_engonly_f1x,CONDITION=bl,BLOCK_SIZE=512 "$SL/pack.sbatch")
RN=exp12_swref_engonly_gpt2sL4_s1; RUN=$CS_ROOT/runs/$RN
rm -rf "$RUN"; mkdir -p "$RUN/data"
ln -sfn "$P/data/packed_512" "$RUN/data/packed_512"
ln -sfn "$SWTOK" "$RUN/data/tokenizer"
if [[ -e "$SW7POOL/valid_denglisch_manual_cs.jsonl" ]]; then
  ln -sfn "$SW7POOL/valid_denglisch_manual_cs.jsonl" "$RUN/data/valid_denglisch_manual_cs.jsonl"
fi
jt=$(sbatch_site --parsable --dependency=afterok:$jk \
  --gres=gpu:1 --time=1-00:00:00 \
  --export=ALL,RUN_NAME=$RN,CONDITION=bl,MONO_HELDOUT=$MH,MODEL_TAG=gpt2s,N_LAYER=4,N_EMBD=768,N_HEAD=12,BLOCK_SIZE=512,FIXED_TOKENS=$D,SEED=1,SCORE_POINTS=20,BATCH_SIZE=64,GRAD_ACCUM=1,TOKEN_BUDGET=fixed,EPOCHS=1.0 \
  "$SL/train_one_gpu.sbatch")
echo "setup=$js pack=$jk train=$jt"
