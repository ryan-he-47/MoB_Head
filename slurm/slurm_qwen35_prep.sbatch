#!/bin/bash
#SBATCH --exclude=gpu-v100s-06
#SBATCH --job-name=qwen35_prep
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/slurm_qwen35_prep_%j.out

# Qwen3.5-2B prep (2026-09-14 decisions: instruct variant; EN/ZH/Code/Instruct
# 50/30/15/5 ~2B tokens; SCRATCH dir, download only needed shards):
#   step 1: weights + dataset shards download (tools/qwen35_download.py)
#   step 2: interleaved token cache build (tools/build_multimix_cache.py)
# tiny partition (CPU-only job; small CPU jobs are rejected on batch).
set -euxo pipefail
cd "${MOB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=${PYTHON:-python}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

$PY -u -m tools.qwen35_download
$PY -u -m tools.build_multimix_cache

echo QWEN35_PREP_DONE
