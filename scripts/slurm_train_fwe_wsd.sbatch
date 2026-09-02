#!/bin/bash
#SBATCH --job-name=mob_fwe_wsd
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/train_wsd_%j.out

# Schedule-variant run (user decision 2026-09-01, parallel to job 519759):
# peak LR 2e-3 -> 7e-4, warmup 200 -> 2500 (1% of 245k), cosine -> WSD
# (stable at peak, linear decay over last 35% of steps).
# Same data/head as the long run: FineWeb-Edu 1B-token cache x 1 epoch,
# K=128 / width 3072, fp16 backbone, head_latest.pt every 20k steps.
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
  --lr 7e-4 --warmup 2500 --schedule wsd --decay-frac 0.35 \
  --ckpt-every 20000 \
  --outdir runs/p2_fineweb_wsd

$PY scripts/p3_eval.py \
  --data-root . \
  --head runs/p2_fineweb_wsd/head_final.pt \
  --k 128 --width 3072 \
  --token-cache data/fineweb_dev_tokens.npy \
  --n-chunks 64 --batch 4 \
  --backbone-dtype float16 \
  --outdir runs/p3_fineweb_wsd_eval

echo WSD_TRAIN_DONE
