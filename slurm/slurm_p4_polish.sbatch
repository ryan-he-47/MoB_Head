#!/bin/bash
#SBATCH --job-name=mob_p4_polish
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-10:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/p4_polish_%j.out

# P4 polish (user-accepted option A, 2026-09-04, docs/14): decay-tail polish
# from the v2 step-10000 checkpoint — tests "schedule noise-floor vs capacity"
# and cashes the basin v2 found. LR ratio per the local control arms
# (backbone-fast/head-slow): bb 1e-5 / head 1e-6, cosine -> 5% over 5k steps.
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python

nvidia-smi
$PY -m train.p4_joint_train \
  --data-root . \
  --resume-adapter runs/p4_joint_lowlr/adapter_step10000 \
  --head runs/p4_joint_lowlr/head_step10000.pt \
  --token-cache fineweb_train_tokens.npy \
  --dev-cache fineweb_dev_tokens.npy \
  --ood-cache babylm_dev_tokens.npy \
  --steps 5000 --batch 8 \
  --lr-backbone 1e-5 --lr-head 1e-6 --warmup 200 --schedule cosine \
  --panel-every 1000 --save-every 5000 --seed 45 \
  --outdir runs/p4_polish

echo P4_POLISH_DONE
