#!/bin/bash
#SBATCH --job-name=cfhead_longform
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs1/scratch/yihe47/qwen35/logs/cfhead_longform_%j.out

# J4 of docs/26-cfhead-headtohead-plan.md section 3 (protocol A): long-form
# autoregressive head-to-head on Qwen3.5-2B — native / CFHead-Q(k=256) /
# CFHead-M(k=8192) / MoB cascade (roundflip + embedding-row rerank).
# 6 prompts (ShareGPT 2 + LMSYS 2 + LongBench hotpotqa/samsum) x 8192 greedy
# steps, EOS disabled, batch 1, KV cache; per-step fp32 native shadow scoring
# (hit = containment, agreement = token match) with repeat/novel bucketing
# (docs/25 appendix C anti-confound). All four arms run sequentially in this
# one job.
# Gates: the J1 cache-build job must have produced both CFHead caches under
# $SCRATCH/cfhead_cache, the bench_data exports must be uploaded, and the p9
# swiglu head_latest.pt must exist.

set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python
SCRATCH=/gpfs1/scratch/yihe47/qwen35
mkdir -p $SCRATCH/logs $SCRATCH/runs/cfhead_longform

nvidia-smi
$PY -u -m eval.cfhead_longform \
  --model-dir $SCRATCH/models/Qwen3.5-2B \
  --cfhead-cache-dir $SCRATCH/cfhead_cache \
  --mob-head $SCRATCH/runs/p9_qwen35_swiglu/head_latest.pt \
  --codebook $SCRATCH/data/codebook_qwen35.npz \
  --data-dir $SCRATCH/bench_data \
  --out-dir $SCRATCH/runs/cfhead_longform \
  --steps 8192 --seed 99 \
  --arms native,cfhead_q,cfhead_m,mob_cascade

echo CFHEAD_LONGFORM_DONE
