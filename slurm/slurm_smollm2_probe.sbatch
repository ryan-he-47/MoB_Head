#!/bin/bash
#SBATCH --job-name=mob_smollm2_probe
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/slurm_smollm2_probe_%j.out

# Frankenstein-hypothesis probe (docs/01-method.md): SmolLM2-1.7B (NATIVELY trained) as
# the subject — identical distillation recipe as the Llama WSD run, 50k steps.
# Metrics are self-normalizing per model (PPL ratio vs its own native head),
# so the trajectory is directly comparable with the Llama line.
set -euxo pipefail
cd "${MOB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=${PYTHON:-python}

nvidia-smi
$PY -m train.p2_train \
  --data-root . \
  --model-dir models/SmolLM2-1.7B \
  --data-format hf --dev-format hf \
  --token-cache fineweb_smollm2_tokens.npy \
  --dev-cache fineweb_smollm2_dev_tokens.npy \
  --codebook codebook_smollm2.npz \
  --k 128 --width 3072 \
  --epochs 1 --batch 8 --topk 1024 --loss-chunk 2048 \
  --max-chunks 400000 \
  --backbone-dtype float16 \
  --lr 7e-4 --warmup 2500 --schedule wsd --decay-frac 0.35 \
  --ckpt-every 10000 --ckpt-keep 2 \
  --outdir runs/p6_smollm2_probe

$PY -m eval.p3_eval \
  --data-root . \
  --model-dir models/SmolLM2-1.7B \
  --head runs/p6_smollm2_probe/head_final.pt \
  --k 128 --width 3072 \
  --codebook codebook_smollm2.npz \
  --token-cache data/fineweb_smollm2_dev_tokens.npy \
  --n-chunks 64 --batch 4 \
  --backbone-dtype float16 \
  --outdir runs/p6_smollm2_probe_eval

echo SMOLLM2_PROBE_DONE
