#!/bin/bash
#SBATCH --job-name=train_ddp
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-12:00:00
#
# Multi-GPU (DDP) trainer. Same effective batch as train_one_gpu.sbatch.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$_HERE/../_env.sh"

RUN_NAME=${RUN_NAME:?}; CONDITION=${CONDITION:?}
MODEL_TAG=${MODEL_TAG:?}; N_LAYER=${N_LAYER:?}; N_EMBD=${N_EMBD:-768}; N_HEAD=${N_HEAD:-12}
BLOCK_SIZE=${BLOCK_SIZE:-512}; BATCH_SIZE=${BATCH_SIZE:-64}; GRAD_ACCUM=${GRAD_ACCUM:-1}
FIXED_TOKENS=${FIXED_TOKENS:?}; LEARNING_RATE=${LEARNING_RATE:-0.0005}
NGPUS=${NGPUS:-${SLURM_GPUS_ON_NODE:-4}}

RUN="$CS_ROOT/runs/$RUN_NAME"; DATA="$RUN/data"; PACKED="$DATA/packed_${BLOCK_SIZE}/$CONDITION"
TOKENIZER="$DATA/tokenizer"; MODEL_NAME="${MODEL_TAG}_L${N_LAYER}${MODEL_SUFFIX:-}"
MODEL_DIR="$RUN/models_packed/$MODEL_NAME"; DYN="$RUN/dynamics_packed/${MODEL_NAME}_dynamics.csv"
mkdir -p "$RUN/models_packed" "$RUN/dynamics_packed" "$RUN/tmp"
source "$CS_ENV/bin/activate"; export TMPDIR="$RUN/tmp"
[[ -s "$PACKED/metadata.json" ]] || { echo "missing packed: $PACKED" >&2; exit 2; }

SCORE_POINTS=${SCORE_POINTS:-20}
MONO_HELDOUT=${MONO_HELDOUT:-$CS_ROOT/data/soloweb_heldout}
TS_HELDOUT=${TS_HELDOUT:-$CS_ROOT/data/ts_heldout}
read MAX_STEPS NPARAMS CKPTS < <(python - "$BATCH_SIZE" "$BLOCK_SIZE" "$GRAD_ACCUM" "$NGPUS" "$FIXED_TOKENS" "$N_LAYER" "$N_EMBD" "$N_HEAD" "$TOKENIZER" <<'PY'
import sys, math
b,block,grad,ng,fixed,nl,ne,nh = (int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),
                                  int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]),int(sys.argv[8]))
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
vocab=len(GPT2TokenizerFast.from_pretrained(sys.argv[9], add_prefix_space=True))
cfg=GPT2Config(vocab_size=vocab,n_positions=block,n_ctx=block,n_embd=ne,n_layer=nl,n_head=nh)
npar=sum(p.numel() for p in GPT2LMHeadModel(cfg).parameters())
tokens_per_step = b*block*grad*ng
max_steps=max(1, math.ceil(fixed/tokens_per_step))
ms=[1,2,4,5,20/3.0,10,20,40,80]
ck=sorted(set(min(max_steps,max(1,int(round(k*npar/tokens_per_step)))) for k in ms if k*npar<=fixed*1.001))
print(max_steps, npar, ",".join(str(c) for c in ck))
PY
)
EVAL_STEPS=$(( MAX_STEPS / SCORE_POINTS )); [[ $EVAL_STEPS -lt 1 ]] && EVAL_STEPS=1
CKPT_ARGS=(); IFS=',' read -ra CKARR <<< "$CKPTS"; for c in "${CKARR[@]}"; do CKPT_ARGS+=(--checkpoint-step "$c"); done

torchrun --standalone --redirects 3 --tee 3 --nproc_per_node="$NGPUS" \
  "$CS_CODE/training/train_gpt2.py" \
  --packed-dir "$PACKED" --tokenizer-dir "$TOKENIZER" --output-dir "$MODEL_DIR" \
  --eval-jsonl "$MONO_HELDOUT/valid_monoweb_eng.jsonl" "$MONO_HELDOUT/valid_monoweb_deu.jsonl" \
               "$DATA/valid_denglisch_manual_cs.jsonl" "$TS_HELDOUT/valid_tongueswitcher_cs.jsonl" \
  --eval-name monoweb_eng monoweb_deu denglisch_manual tongueswitcher \
  --dense-eval-limit "${DENSE_EVAL_LIMIT:-400}" \
  --train-condition "$MODEL_NAME" --dynamics-csv "$DYN" --seed "${SEED:-13}" \
  --max-steps "$MAX_STEPS" --eval-steps "$EVAL_STEPS" --score-points "$SCORE_POINTS" --logging-steps 50 \
  --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
  --n-layer "$N_LAYER" --n-head "$N_HEAD" --n-embd "$N_EMBD" --learning-rate "$LEARNING_RATE" \
  --random-packed-split --save-checkpoints --save-steps 2000 "${CKPT_ARGS[@]}" \
  --report-to none ${OVERWRITE_FLAG---overwrite}
echo "TRAIN_DDP_DONE=$MODEL_DIR"
