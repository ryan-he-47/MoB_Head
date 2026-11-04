#!/bin/bash
#SBATCH --job-name=qwen35_verify
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/slurm_qwen35_verify_%j.out

# Qwen3.5-2B tap-point verification + 18-bit codebook build (GPU).
# Gates the distillation: if TAP_OK fails, stop and reassess the tap point.
set -euxo pipefail
cd "${MOB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=${PYTHON:-python}
MODEL=${MOB_SCRATCH:?set MOB_SCRATCH}/models/Qwen3.5-2B

nvidia-smi
$PY -u -m tools.verify_qwen35_tap $MODEL

$PY -u -m tools.build_codebook \
  --model-dir $MODEL \
  --out-npz ${MOB_SCRATCH:?set MOB_SCRATCH}/data/codebook_qwen35.npz \
  --out-meta ${MOB_SCRATCH:?set MOB_SCRATCH}/data/codebook_qwen35_meta.json \
  --device cuda

echo QWEN35_VERIFY_DONE
