#!/bin/bash
#SBATCH --job-name=mob_distill
#SBATCH --partition=FILL_ME            # TODO(user): cluster partition for V100
#SBATCH --account=FILL_ME              # TODO(user): delete this line if not needed
#SBATCH --gres=gpu:v100:1              # TODO(user): adjust to cluster GPU syntax (maybe just gpu:1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/p2_%j.out

# ---- TODO(user): environment on the cluster ----
# module load cuda/12.x anaconda3
# source activate <env>    # needs: torch(cu), transformers, numpy, safetensors

cd ~/llm_codec           # TODO(user): repo root on the cluster (upload weights + data + scripts)

python -m train.p2_train \
  --data-root ~/llm_codec \
  --data-format hf \
  --k 128 --width 3072 \
  --outdir runs/p2_fineweb \
  --epochs 1 --batch 8 --topk 1024 --loss-chunk 2048 \
  --backbone-dtype float16    # V100 (Volta) tensor cores: fp16 yes, bf16 no
# epochs/data fraction is a submission-time user decision. Scale check for
# FineWeb-Edu sample-10BT: ~10-12B tokens -> ~20M chunks (512 tok) -> ~2.4M
# steps/epoch at batch 8 -> several hundred GPU-hours for a full epoch on one
# V100. A fraction (e.g. --hf-max-docs) or a multi-job split will be needed.

# acceptance eval afterwards (same job; cheap):
python -m eval.p3_eval \
  --data-root ~/llm_codec \
  --head runs/p2_fineweb/head_final.pt \
  --k 128 --width 3072 \
  --outdir runs/p3_eval --n-chunks 64 --batch 4 \
  --backbone-dtype float16
