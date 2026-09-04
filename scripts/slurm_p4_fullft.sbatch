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
# pure MoB-NLL, from the WSD final head. Validated recipe: bf16 frozen trunk +
# fp32 trainable layers + autocast. Same panel protocol as the LoRA runs.
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python

nvidia-smi
$PY scripts/p4_joint_train.py \
  --data-root . \
  --mode unfreeze --n-layers 4 \
  --head runs/p2_fineweb_wsd/head_final.pt \
  --token-cache fineweb_train_tokens.npy \
  --dev-cache fineweb_dev_tokens.npy \
  --ood-cache babylm_dev_tokens.npy \
  --steps 20000 --batch 8 \
  --lr-backbone 2e-5 --lr-head 1e-5 --warmup 500 \
  --panel-every 2000 --save-every 5000 \
  --outdir runs/p4_fullft

echo P4_FULLFT_DONE
