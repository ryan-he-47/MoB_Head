#!/bin/bash
#SBATCH --job-name=mob_headscale_dual
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_p2_headscale_dual_%j.out


# (~51M each) distilled side by side on ONE frozen Llama backbone — shared
# backbone forward (~1.2x cost of a single run) and identical batch stream
# (paired comparison). Recipe = WSD proven in job 520825.
set -euxo pipefail
cd "${MOB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=${PYTHON:-python}

nvidia-smi
$PY -m train.p2_train_dual \
  --data-root . \
  --token-cache fineweb_train_tokens.npy \
  --dev-cache fineweb_dev_tokens.npy \
  --epochs 1 --batch 8 --topk 1024 --loss-chunk 2048 \
  --lr 7e-4 --warmup 2500 --decay-frac 0.35 \
  --ckpt-every 20000 --ckpt-keep 2 \
  --outdir runs/p2_headscale_dual

for ARM in wide deep; do
  if [ "$ARM" = "wide" ]; then WIDTH=4096; DEPTH=3; else WIDTH=3072; DEPTH=5; fi
  $PY -m eval.p3_eval \
    --data-root . \
    --head "runs/p2_headscale_dual/head_${ARM}_final.pt" \
    --k 128 --width $WIDTH --depth $DEPTH \
    --token-cache data/fineweb_dev_tokens.npy \
    --n-chunks 64 --batch 4 \
    --backbone-dtype float16 \
    --outdir "runs/p3_headscale_${ARM}_eval"
done

echo DUAL_ALL_DONE
