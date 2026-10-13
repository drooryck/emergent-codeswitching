#!/bin/bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#
# Single-GPU trainer. Requires paths.env. Override partition/account/gres at submit.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$_HERE/../_env.sh"

RUN_NAME=${RUN_NAME:?Set RUN_NAME}
CONDITION=${CONDITION:?Set CONDITION}
MODEL_TAG=${MODEL_TAG:?Set MODEL_TAG}
N_LAYER=${N_LAYER:?Set N_LAYER}
N_EMBD=${N_EMBD:-768}
N_HEAD=${N_HEAD:-12}
BLOCK_SIZE=${BLOCK_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-64}
GRAD_ACCUM=${GRAD_ACCUM:-1}
EPOCHS=${EPOCHS:-1.0}
TOKEN_BUDGET=${TOKEN_BUDGET:-fixed}
CHINCHILLA_MULT=${CHINCHILLA_MULT:-20}
FIXED_TOKENS=${FIXED_TOKENS:-0}
SCORE_POINTS=${SCORE_POINTS:-50}
LOGGING_STEPS=${LOGGING_STEPS:-50}
LEARNING_RATE=${LEARNING_RATE:-0.0005}
REPORT_TO=${REPORT_TO:-none}

RUN="$CS_ROOT/runs/$RUN_NAME"
DATA="$RUN/data"
PACKED="$DATA/packed_${BLOCK_SIZE}/$CONDITION"
TOKENIZER="$DATA/tokenizer"
MODEL_NAME="${MODEL_TAG}_L${N_LAYER}${MODEL_SUFFIX:-}"
MODEL_DIR="$RUN/models_packed/$MODEL_NAME"
DYNAMICS_CSV="$RUN/dynamics_packed/${MODEL_NAME}_dynamics.csv"
mkdir -p "$RUN/models_packed" "$RUN/dynamics_packed" "$RUN/tmp"
source "$CS_ENV/bin/activate"
export TMPDIR="$RUN/tmp"

if [[ ! -s "$PACKED/metadata.json" ]]; then
  echo "Missing packed metadata: $PACKED/metadata.json" >&2
  exit 2
fi

read MAX_STEPS CKPTS TARGET_TOKENS NPARAMS < <(python - "$PACKED/metadata.json" "$BATCH_SIZE" "$BLOCK_SIZE" "$GRAD_ACCUM" "$EPOCHS" "$TOKEN_BUDGET" "$CHINCHILLA_MULT" "$N_LAYER" "$N_EMBD" "$N_HEAD" "$TOKENIZER" "$FIXED_TOKENS" <<'PY'
import json, math, sys
meta = json.loads(open(sys.argv[1]).read())
batch, block, grad = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
epochs = float(sys.argv[5]); budget = sys.argv[6]; mult = float(sys.argv[7])
n_layer, n_embd, n_head = int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])
fixed_tokens = int(sys.argv[12])
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
vocab = len(GPT2TokenizerFast.from_pretrained(sys.argv[11], add_prefix_space=True))
cfg = GPT2Config(vocab_size=vocab, n_positions=block, n_ctx=block, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
nparams = sum(p.numel() for p in GPT2LMHeadModel(cfg).parameters())
packed = int(meta["packed_tokens"])
if budget == "chinchilla":
    target = int(mult * nparams)
elif budget == "fixed":
    target = int(fixed_tokens)
else:
    target = int(packed * epochs)
max_steps = max(1, math.ceil(target / (batch * block * grad)))
per_step = batch * block * grad
milestones = [1, 2, 4, 5, 20/3.0, 10, 20, 40, 80]
ck = sorted(set(min(max_steps, max(1, int(round(k * nparams / per_step)))) for k in milestones
                if k * nparams <= target * 1.001))
print(max_steps, ",".join(str(c) for c in ck), target, nparams)
PY
)
EVAL_STEPS=$(( MAX_STEPS / SCORE_POINTS )); [[ "$EVAL_STEPS" -lt 1 ]] && EVAL_STEPS=1
CKPT_ARGS=()
IFS=',' read -ra CKARR <<< "$CKPTS"
for c in "${CKARR[@]}"; do CKPT_ARGS+=(--checkpoint-step "$c"); done

python "$CS_CODE/training/train_gpt2.py" \
  --packed-dir "$PACKED" \
  --tokenizer-dir "$TOKENIZER" \
  --output-dir "$MODEL_DIR" \
  --eval-jsonl "${MONO_HELDOUT:-$CS_ROOT/data/soloweb_heldout}/valid_monoweb_eng.jsonl" \
               "${MONO_HELDOUT:-$CS_ROOT/data/soloweb_heldout}/valid_monoweb_deu.jsonl" \
               "$DATA/valid_denglisch_manual_cs.jsonl" \
               "${TS_HELDOUT:-$CS_ROOT/data/ts_heldout}/valid_tongueswitcher_cs.jsonl" \
  --eval-name monoweb_eng monoweb_deu denglisch_manual tongueswitcher \
  --dense-eval-limit "${DENSE_EVAL_LIMIT:-400}" \
  --train-condition "$MODEL_NAME" \
  --dynamics-csv "$DYNAMICS_CSV" \
  --seed "${SEED:-13}" \
  --max-steps "$MAX_STEPS" \
  --eval-steps "$EVAL_STEPS" \
  --score-points "$SCORE_POINTS" \
  --logging-steps "$LOGGING_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --n-layer "$N_LAYER" \
  --n-head "$N_HEAD" \
  --n-embd "$N_EMBD" \
  --learning-rate "$LEARNING_RATE" \
  --random-packed-split \
  "${CKPT_ARGS[@]}" \
  --report-to "$REPORT_TO" \
  --overwrite

echo "TRAIN_DONE=$MODEL_DIR"
