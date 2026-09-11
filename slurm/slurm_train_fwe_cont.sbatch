#!/bin/bash
#SBATCH --job-name=mob_fwe_cont
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/train_cont_%j.out

# Continuation run (user decision 2026-09-03, docs/14): probe the head's
# ceiling with a low-LR polish on top of the WSD final head.
#   --resume <weights-only> = weights init + fresh optimizer/schedule.
# peak 2e-4 cosine -> ~0 over 50k steps (--max-chunks 400k, batch 8),
# same 1B-token cache re-seeded (order reshuffle), DAEM off (T=1).
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python

nvidia-smi
$PY -m train.p2_train \
  --data-root . \
  --data-format hf --dev-format hf \
  --token-cache fineweb_train_tokens.npy \
  --k 128 --width 3072 \
  --epochs 1 --batch 8 --topk 1024 --loss-chunk 2048 \
  --backbone-dtype float16 \
  --resume runs/p2_fineweb_wsd/head_final.pt \
  --lr 2e-4 --warmup 200 --schedule cosine \
  --max-chunks 400000 --seed 43 --daem-t0 1.0 \
  --ckpt-every 10000 --ckpt-keep 2 \
  --outdir runs/p2_fineweb_cont

$PY -m eval.p3_eval \
  --data-root . \
  --head runs/p2_fineweb_cont/head_final.pt \
  --k 128 --width 3072 \
  --token-cache data/fineweb_dev_tokens.npy \
  --n-chunks 64 --batch 4 \
  --backbone-dtype float16 \
  --outdir runs/p3_fineweb_cont_eval

echo CONT_TRAIN_DONE
