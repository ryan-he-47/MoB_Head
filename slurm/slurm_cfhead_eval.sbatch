#!/bin/bash
#SBATCH --job-name=cfhead_tf_eval
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs1/scratch/yihe47/qwen35/logs/cfhead_eval_%j.out

set -e
SCRATCH=/gpfs1/scratch/yihe47/qwen35
cd /gpfs1/home/yihe47/llm_codec
/home/yihe47/miniconda3/envs/cose2/bin/python -m eval.cfhead_eval \
  --model-dir $SCRATCH/models/Qwen3.5-2B \
  --head $SCRATCH/runs/p9_qwen35_swiglu/head_latest.pt \
  --codebook $SCRATCH/data/codebook_qwen35.npz \
  --cfhead-cache-dir $SCRATCH/cfhead_cache \
  --dev-dir $SCRATCH/cache --dev-names en zh code instruct \
  --head-type swiglu --hidden 7300 --k 128 \
  --n-chunks 32 --batch 4 --pos-chunk 8 \
  --out-dir $SCRATCH/runs/cfhead_teacher_forced
