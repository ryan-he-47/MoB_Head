#!/bin/bash
#SBATCH --job-name=mob_qwen35_swiglu
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/qwen35_distill_%j.out

# Qwen3.5-2B (instruct) distillation — the big-vocab mainline (2026-09-15):
# swiglu head (race3 winner) K=128, m=18 from codebook, cosine 7e-4 warmup 4000
# (race-validated swiglu comfort at h=2048), 1 epoch over the 2B-token
# EN/ZH/Code/Instruct 50/30/15/5 mix (~488k steps, ~60h est).
# All artifacts in SCRATCH (home quota). Gates: verify job must have produced
# the codebook + TAP_OK first.
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python
SCRATCH=/gpfs1/scratch/yihe47/qwen35

# combined dev cache (4 sources x 2M tokens, concatenated in source order)
$PY - <<'EOF'
import numpy as np
parts = [np.load(f"/gpfs1/scratch/yihe47/qwen35/cache/qwen35_dev_{s}_tokens.npy")
         for s in ("en", "zh", "code", "instruct")]
np.save("/gpfs1/scratch/yihe47/qwen35/cache/qwen35_dev_mix_tokens.npy",
        np.concatenate(parts))
print("dev_mix:", sum(len(p) for p in parts))
EOF

nvidia-smi
$PY -u -m train.p2_train \
  --data-root . \
  --model-dir $SCRATCH/models/Qwen3.5-2B \
  --data-format hf --dev-format hf \
  --token-cache $SCRATCH/cache/qwen35_mix_train_tokens.npy \
  --dev-cache $SCRATCH/cache/qwen35_dev_mix_tokens.npy \
  --codebook $SCRATCH/data/codebook_qwen35.npz \
  --head-type swiglu --hidden 7300 \
  --k 128 --epochs 1 --batch 8 --topk 1024 --loss-chunk 2048 \
  --warmup 4000 --schedule cosine \
  --backbone-dtype float16 \
  --ckpt-every 20000 --ckpt-keep 3 \
  --outdir $SCRATCH/runs/p9_qwen35_swiglu

# chained acceptance evals: mix dev + per-source devs (coverage breakdown)
for S in mix en zh code instruct; do
  $PY -u -m eval.p3_eval --data-root . --model-dir $SCRATCH/models/Qwen3.5-2B \
    --head $SCRATCH/runs/p9_qwen35_swiglu/head_final.pt --head-type swiglu --k 128 \
    --codebook $SCRATCH/data/codebook_qwen35.npz \
    --token-cache $SCRATCH/cache/qwen35_dev_${S}_tokens.npy \
    --n-chunks 64 --batch 4 --backbone-dtype float16 \
    --outdir $SCRATCH/runs/p9_qwen35_swiglu_eval_${S}
done

echo QWEN35_DISTILL_DONE
