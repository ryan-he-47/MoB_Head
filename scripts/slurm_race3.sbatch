#!/bin/bash
#SBATCH --job-name=mob_race3
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-12:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/race3_%j.out

# Three-head race on SmolLM2-1.7B (docs/16 §5, user design 2026-09-07; cosine
# per user ruling 2026-09-09: fixed budget -> full-length decay tail, peak at
# the top of each architecture's comfort band from the local LR scan).
# 312.5M-token cache, batch 8 x 512 -> ~76.1k steps/epoch, 3 epochs = ~228k
# steps (~93% of the Llama-line 245k budget; data repetition is shared across
# arms, so no internal confound). Est. ~0.7s/step -> ~45h + 3 chained evals.
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python

nvidia-smi
$PY scripts/p2_train_dual.py \
  --data-root . \
  --model-dir models/SmolLM2-1.7B \
  --token-cache fineweb_smollm2_tokens.npy \
  --dev-cache fineweb_smollm2_dev_tokens.npy \
  --codebook codebook_smollm2.npz \
  --arm-set race3 --schedule cosine \
  --peaks control=1.5e-3,resmlp=1e-3,swiglu=7e-4 \
  --k 128 --epochs 3 --batch 8 --topk 1024 --loss-chunk 2048 \
  --warmup 4000 --backbone-dtype float16 \
  --ckpt-every 20000 --ckpt-keep 3 \
  --outdir runs/p7_race3

# chained acceptance evals (p3_eval, 64 chunks, fineweb in-domain dev)
$PY scripts/p3_eval.py --data-root . --model-dir models/SmolLM2-1.7B \
  --head runs/p7_race3/head_control_final.pt --head-type mob --k 128 --width 3072 --depth 3 \
  --codebook codebook_smollm2.npz --token-cache data/fineweb_smollm2_dev_tokens.npy \
  --n-chunks 64 --batch 4 --backbone-dtype float16 --outdir runs/p7_race3_eval_control
$PY scripts/p3_eval.py --data-root . --model-dir models/SmolLM2-1.7B \
  --head runs/p7_race3/head_resmlp_final.pt --head-type resmlp --k 128 --width 3072 --depth 5 \
  --codebook codebook_smollm2.npz --token-cache data/fineweb_smollm2_dev_tokens.npy \
  --n-chunks 64 --batch 4 --backbone-dtype float16 --outdir runs/p7_race3_eval_resmlp
$PY scripts/p3_eval.py --data-root . --model-dir models/SmolLM2-1.7B \
  --head runs/p7_race3/head_swiglu_final.pt --head-type swiglu --k 128 --hidden 7300 --blocks 1 \
  --codebook codebook_smollm2.npz --token-cache data/fineweb_smollm2_dev_tokens.npy \
  --n-chunks 64 --batch 4 --backbone-dtype float16 --outdir runs/p7_race3_eval_swiglu

echo RACE3_DONE
