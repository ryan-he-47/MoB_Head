#!/bin/bash
#SBATCH --exclude=gpu-v100s-06
#SBATCH --job-name=diag_load
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-v100s-07
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/slurm_diag_load_%j.out

# Decisive test for the race3 mid-load deaths: both jobs died at ~40% of weight
# loading (exit 120, 31s, same node gpu-v100s-07). Loader code is identical to
# the proven probe. If this reproduces on 07 -> bad node, add to exclude list.
set -euxo pipefail
cd "${MOB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=${PYTHON:-python}
$PY -c "import torch, transformers; print('transformers', transformers.__version__); from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('models/SmolLM2-1.7B', dtype=torch.float16).cuda(); print('LOAD_OK', sum(p.numel() for p in m.parameters()))"
echo DIAG_DONE
