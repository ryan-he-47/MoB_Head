#!/bin/bash
#SBATCH --job-name=cfhead_benchmarks
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_cfhead_benchmarks_%j.out

# CFHead head-to-head, benchmark arm (docs/01-method.md,
# job J3): Original vs CFHead-Q (C=64, k=256) vs CFHead-M (C=4, k=8192) vs
# MoB-cascade (SwiGLU K=128 roundflip+rerank) on MGSM en/zh, MMLU, GSM8K, ARC-C.
# All four tasks x four arms run sequentially inside ONE python invocation
# (model + caches + MoB head are loaded once). Gates: J1 cache build must have
# produced cfhead_v3_*_T8_C{64,4}_*.pt in $SCRATCH/cfhead_cache, bench_data must
# be uploaded to $SCRATCH/bench_data, and the p9 distill head_latest.pt must exist.
set -euxo pipefail
cd "${MOB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=${PYTHON:-python}
SCRATCH=${MOB_SCRATCH:?set MOB_SCRATCH}

nvidia-smi
$PY -u -m eval.cfhead_benchmarks \
  --model-dir $SCRATCH/models/Qwen3.5-2B \
  --bench-root $SCRATCH/bench_data \
  --cfhead-cache-dir $SCRATCH/cfhead_cache \
  --mob-head $SCRATCH/runs/p9_qwen35_swiglu/head_latest.pt \
  --codebook $SCRATCH/data/codebook_qwen35.npz \
  --out $SCRATCH/runs/cfhead_benchmarks

echo CFHEAD_BENCHMARKS_DONE
