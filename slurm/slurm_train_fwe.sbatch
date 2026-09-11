#!/bin/bash
#SBATCH --job-name=mob_fwe
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/train_%j.out

# B+A quick verification run on the cluster: 1024 steps (max-chunks 8192),
# K=128 / width 3072, FineWeb-Edu caches (built by slurm_setup.sbatch).
# Validates: fp16 backbone on V100, cache paths, train loop, eval chain.
# For the long run: drop --max-chunks / raise it (scale = user decision).
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python

nvidia-smi
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

$PY -m train.p2_train \
  --data-root . \
  --data-format hf --dev-format hf \
  --token-cache fineweb_train_tokens.npy \
  --k 128 --width 3072 \
  --epochs 1 --batch 8 --topk 1024 --loss-chunk 2048 \
  --backbone-dtype float16 \
  --max-chunks 8192 \
  --outdir runs/p2_fineweb_smoke_cluster

$PY -m eval.p3_eval \
  --data-root . \
  --head runs/p2_fineweb_smoke_cluster/head_final.pt \
  --k 128 --width 3072 \
  --token-cache data/fineweb_dev_tokens.npy \
  --n-chunks 16 --batch 4 \
  --backbone-dtype float16 \
  --outdir runs/p3_fineweb_cluster_eval

echo TRAIN_JOB_DONE
