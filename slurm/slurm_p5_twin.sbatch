#!/bin/bash
#SBATCH --job-name=mob_p5_twin
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_p5_twin_%j.out

# P5 / Rung-3 from-scratch twin pretrain (docs/01-method.md).
# Usage: sbatch slurm_p5_twin.sbatch {native|mob}
# Two identical 212M Llama-mini models, full fp32, same seed/data order,
# differ ONLY in the output head. 60k steps x batch 16 x 512 = 0.49B tokens.
# NOTE: gpu-v100s-06 excluded — known-bad node (jobs land there and never get

set -euxo pipefail
cd "${MOB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=${PYTHON:-python}
HEAD_TYPE=$1

nvidia-smi
$PY -m train.p5_twin_pretrain \
  --head-type "$HEAD_TYPE" \
  --data-root . \
  --token-cache fineweb_train_tokens.npy \
  --dev-cache fineweb_dev_tokens.npy \
  --steps 60000 --batch 16 \
  --lr 6e-4 --warmup-frac 0.02 \
  --k 128 --width 922 --lambda-ent 0.01 \
  --panel-every 2000 --save-every 5000 \
  --outdir "runs/p5_twin_${HEAD_TYPE}"

echo P5_TWIN_${HEAD_TYPE}_DONE
