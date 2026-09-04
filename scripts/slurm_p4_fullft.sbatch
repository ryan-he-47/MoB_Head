#!/bin/bash
#SBATCH --job-name=mob_p4_fullft
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/p4_fullft_%j.out

# P4 escalation ladder rung 1 (user direction 2026-09-04, docs/14): LoRA's
# low-rank assumption does not fit "reshape h for a new output space" — so
# FULL-PARAMETER unfreeze of the last 4 layers + final norm (~268M trainable),
# pure MoB-NLL, from the WSD final head. v2: first attempt NaN'd on V100 within
# 400 warmup steps (bf16 runs through emulation on pre-Ampere; LoRA runs
# survived because adapters stayed ~0 so the bf16 path was barely exercised).
# Fix = the proven pre-Ampere stack: fp16 trunk + fp32 trainable layers +
# autocast fp16 + GradScaler; LRs halved (bb 1e-5 / head 5e-6), warmup 1000.
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python

nvidia-smi
$PY scripts/p4_joint_train.py \
  --data-root . \
  --mode unfreeze --n-layers 4 --compute-dtype float16 \
  --head runs/p2_fineweb_wsd/head_final.pt \
  --token-cache fineweb_train_tokens.npy \
  --dev-cache fineweb_dev_tokens.npy \
  --ood-cache babylm_dev_tokens.npy \
  --steps 20000 --batch 8 \
  --lr-backbone 1e-5 --lr-head 5e-6 --warmup 1000 \
  --panel-every 2000 --save-every 5000 \
  --outdir runs/p4_fullft

echo P4_FULLFT_DONE
