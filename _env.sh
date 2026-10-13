# Shared path setup for every shell/slurm wrapper.
# Copy paths.env.example to paths.env and fill in, or export the variables yourself.
_THIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CS_CODE="${CS_CODE:-$_THIS}"
if [[ -f "$CS_CODE/paths.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$CS_CODE/paths.env"
  set +a
fi
: "${CS_ROOT:?Set CS_ROOT in paths.env or the environment (writable data/runs directory)}"
: "${CS_ENV:?Set CS_ENV in paths.env or the environment (path to the Python venv)}"
export CS_CODE CS_ROOT CS_ENV
export HF_HOME="${HF_HOME:-$CS_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export ACS_PAIRS="${ACS_PAIRS:-$CS_ROOT/data/acs_pairs.csv}"
export CLEAN_DIR="${CLEAN_DIR:-$CS_ROOT/data/fineweb_clean}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-$CS_ROOT/data/sw_union_tokenizer}"
export MONO_HELDOUT="${MONO_HELDOUT:-$CS_ROOT/data/soloweb_heldout}"
export TS_HELDOUT="${TS_HELDOUT:-$CS_ROOT/data/ts_heldout}"
mkdir -p "$CS_ROOT/logs" "$CS_ROOT/runs"

# Site-specific Slurm flags. Leave SLURM_PARTITION / SLURM_ACCOUNT unset to use cluster defaults.
# Usage: sbatch_site [--cpu] [sbatch args...]
sbatch_site() {
  local extra=()
  local part="${SLURM_PARTITION:-}"
  if [[ "${1:-}" == "--cpu" ]]; then
    shift
    part="${SLURM_CPU_PARTITION:-${SLURM_PARTITION:-}}"
  fi
  [[ -n "$part" ]] && extra+=(--partition="$part")
  [[ -n "${SLURM_ACCOUNT:-}" ]] && extra+=(--account="$SLURM_ACCOUNT")
  extra+=(--output="$CS_ROOT/logs/%x_%j.out" --error="$CS_ROOT/logs/%x_%j.err")
  sbatch "${extra[@]}" "$@"
}
