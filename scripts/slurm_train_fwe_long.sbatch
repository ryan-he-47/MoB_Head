#!/bin/bash
#SBATCH --job-name=mob_fwe_long
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/train_long_%j.out

# B+A long run (user decision 2026-09-01, option A): full 1B-token FineWeb-Edu
# cache x 1 epoch (~245k steps, ~24.5h at the measured 0.36s/step on V100).
# K=128 / width 3072, fp16 backbone. head_latest.pt saved every 20k steps.
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python

nvidia-smi
$PY scripts/p2_train.py \
  --data-root . \
  --data-format hf --dev-format hf \
  --token-cache fineweb_train_tokens.npy \
  --k 128 --width 3072 \
  --epochs 1 --batch 8 --topk 1024 --loss-chunk 2048 \
  --backbone-dtype float16 \
  --ckpt-every 20000 \
  --outdir runs/p2_fineweb_full

$PY scripts/p3_eval.py \
  --data-root . \
  --head runs/p2_fineweb_full/head_final.pt \
  --k 128 --width 3072 \
  --token-cache data/fineweb_dev_tokens.npy \
  --n-chunks 64 --batch 4 \
  --backbone-dtype float16 \
  --outdir runs/p3_fineweb_full_eval

echo LONG_TRAIN_DONE
