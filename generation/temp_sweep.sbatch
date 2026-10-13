#!/bin/bash
#SBATCH --job-name=temp_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# Temperature-sweep generation, then TongueSwitcher tagging.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_env.sh
source "$_HERE/../_env.sh"

source "$CS_ENV/bin/activate"
export TOKENIZERS_PARALLELISM=false
python "$CS_CODE/generation/gen_temp_sweep.py"
python "$CS_CODE/generation/tag_temp_sweep.py"
python "$CS_CODE/generation/make_generation_plots.py" || true
echo "TEMP_SWEEP_DONE=$CS_ROOT/generation_out"
