#!/bin/bash
#SBATCH --job-name=mob_shallow_dual
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/shallow_dual_%j.out

# 'Is shallower actually better — even linear?' probe (user question 2026-09-06,
# docs/16 Q2): single-hidden-layer (width 1024, depth 1) vs BARE LINEAR head,
# same distillation recipe/data as the scale dual — paired batches, same
# backbone, identical comparison ruler (original WSD run).
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python

nvidia-smi
$PY scripts/p2_train_dual.py \
  --data-root . \
  --arm-set shallow \
  --token-cache fineweb_train_tokens.npy \
  --dev-cache fineweb_dev_tokens.npy \
  --epochs 1 --batch 8 --topk 1024 --loss-chunk 2048 \
  --lr 7e-4 --warmup 2500 --decay-frac 0.35 \
  --ckpt-every 20000 --ckpt-keep 2 \
  --outdir runs/p2_shallow_dual

echo SHALLOW_DUAL_DONE
