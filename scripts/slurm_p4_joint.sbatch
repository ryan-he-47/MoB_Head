#!/bin/bash
#SBATCH --job-name=mob_p4_joint
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/p4_joint_%j.out

# P4 joint fine-tune (user decision 2026-09-03, docs/14): LoRA r32 last-8
# layers + final norm + MoB head, PURE MoB-NLL loss (no teacher) — MoB head is
# the system's only output head. Panel: MoB dev NLL (acceptance vs pure-native
# baseline), native drift NLL (diagnostic), BabyLM OOD MoB NLL (canary).
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python

nvidia-smi
$PY scripts/p4_joint_train.py \
  --data-root . \
  --head runs/p2_fineweb_wsd/head_final.pt \
  --token-cache fineweb_train_tokens.npy \
  --dev-cache fineweb_dev_tokens.npy \
  --ood-cache babylm_dev_tokens.npy \
  --steps 20000 --batch 8 \
  --lr-backbone 2e-5 --lr-head 1e-4 --warmup 500 \
  --lora-r 32 --lora-layers 8 \
  --panel-every 2000 --save-every 10000 \
  --outdir runs/p4_joint

echo P4_SBatch_DONE
