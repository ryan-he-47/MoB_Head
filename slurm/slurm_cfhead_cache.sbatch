#!/bin/bash
#SBATCH --job-name=cfhead_cache
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --exclude=gpu-v100s-06
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs1/scratch/yihe47/qwen35/logs/cfhead_cache_%j.out

set -e
cd /gpfs1/home/yihe47/llm_codec
/home/yihe47/miniconda3/envs/cose2/bin/python -m tools.build_cfhead_cache \
  --model-dir /gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B \
  --out-dir /gpfs1/scratch/yihe47/qwen35/cfhead_cache \
  --tile 8 --clusters 64 4 --iters 10 --seed 99
