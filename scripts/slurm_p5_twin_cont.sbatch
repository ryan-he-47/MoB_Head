#!/bin/bash
#SBATCH --job-name=mob_p5_twin_cont
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/p5_twin_cont_%j.out

# P5 twin continuation (user direction 2026-09-05): probe the convergence FLOOR
# of both heads. Weights-only resume from each twin's 60k latest.pt; fresh
# optimizer; LR mild restart peak 1e-4 (previous cosine ended at 6e-5) with
# cosine -> 2% over another 60k steps; data permutation CONTINUES past the
# consumed prefix (skip 60000*16 chunks; cache capacity covers 120k steps
# without repeats).
# Usage: sbatch slurm_p5_twin_cont.sbatch {native|mob}
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python
HEAD_TYPE=$1

nvidia-smi
$PY scripts/p5_twin_pretrain.py \
  --head-type "$HEAD_TYPE" \
  --data-root . \
  --token-cache fineweb_train_tokens.npy \
  --dev-cache fineweb_dev_tokens.npy \
  --resume "runs/p5_twin_${HEAD_TYPE}/latest.pt" \
  --skip-steps 60000 \
  --steps 60000 --batch 16 \
  --lr 1e-4 --warmup-frac 0.01 --final-frac 0.02 \
  --k 128 --width 922 --lambda-ent 0.01 \
  --panel-every 2000 --save-every 5000 \
  --outdir "runs/p5_twin_${HEAD_TYPE}_cont"

echo P5_TWIN_CONT_${HEAD_TYPE}_DONE
