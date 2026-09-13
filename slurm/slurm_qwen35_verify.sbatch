#!/bin/bash
#SBATCH --job-name=qwen35_verify
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/qwen35_verify_%j.out

# Qwen3.5-2B tap-point verification + 18-bit codebook build (GPU).
# Gates the distillation: if TAP_OK fails, stop and reassess the tap point.
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python
MODEL=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B

nvidia-smi
$PY -u tools/verify_qwen35_tap.py $MODEL

$PY -u tools/build_codebook.py \
  --model-dir $MODEL \
  --out-npz /gpfs1/scratch/yihe47/qwen35/data/codebook_qwen35.npz \
  --out-meta /gpfs1/scratch/yihe47/qwen35/data/codebook_qwen35_meta.json \
  --device cuda

echo QWEN35_VERIFY_DONE
