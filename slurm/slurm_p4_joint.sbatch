#!/bin/bash
#SBATCH --exclude=gpu-v100s-06
#SBATCH --job-name=mob_p4_joint
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/slurm_p4_joint_%j.out

# P4 joint fine-tune (docs/01-method.md): LoRA r32 last-8
# layers + final norm + MoB head, PURE MoB-NLL loss (no teacher) — MoB head is
# the system's only output head. Panel: MoB dev NLL (acceptance vs pure-native
# baseline), native drift NLL (diagnostic), BabyLM OOD MoB NLL (canary).

# 500->2000, LR down ~0.7 order (backbone 2e-5->5e-6, head 1e-4->2e-5);
# priority is a clean declining curve, slow is fine. Old run's log kept in
# runs/p4_joint/ for comparison.
set -euxo pipefail
cd "${MOB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=${PYTHON:-python}

nvidia-smi
$PY -m train.p4_joint_train \
  --data-root . \
  --head runs/p2_fineweb_wsd/head_final.pt \
  --token-cache fineweb_train_tokens.npy \
  --dev-cache fineweb_dev_tokens.npy \
  --ood-cache babylm_dev_tokens.npy \
  --steps 20000 --batch 8 \
  --lr-backbone 5e-6 --lr-head 2e-5 --warmup 2000 \
  --lora-r 32 --lora-layers 8 \
  --panel-every 2000 --save-every 5000 \
  --outdir runs/p4_joint_lowlr

echo P4_SBatch_DONE
