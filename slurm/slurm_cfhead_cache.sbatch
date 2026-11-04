#!/bin/bash
#SBATCH --job-name=cfhead_cache
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --exclude=gpu-v100s-06
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_cfhead_cache_%j.out

set -e
cd "${MOB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
${PYTHON:-python} -m tools.build_cfhead_cache \
  --model-dir ${MOB_SCRATCH:?set MOB_SCRATCH}/models/Qwen3.5-2B \
  --out-dir ${MOB_SCRATCH:?set MOB_SCRATCH}/cfhead_cache \
  --tile 8 --clusters 64 4 --iters 10 --seed 99
