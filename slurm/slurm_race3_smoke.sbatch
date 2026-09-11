#!/bin/bash
#SBATCH --job-name=mob_race3_smoke
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/race3_smoke_%j.out

# Race3 pre-flight smoke: 8-step training (arm construction, kmeans init,
# per-arm cosine peaks) + all three chained evals on 2 chunks each (exercises
# the new p3_eval --head-type paths). ~15-20 min total. Green -> submit race3.
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python

nvidia-smi
$PY -m train.p2_train_dual \
  --data-root . \
  --model-dir models/SmolLM2-1.7B \
  --token-cache fineweb_smollm2_tokens.npy \
  --dev-cache fineweb_smollm2_dev_tokens.npy \
  --codebook codebook_smollm2.npz \
  --arm-set race3 --schedule cosine \
  --peaks control=1.5e-3,resmlp=1e-3,swiglu=7e-4 \
  --k 128 --epochs 1 --batch 8 --topk 1024 --loss-chunk 2048 \
  --warmup 4000 --backbone-dtype float16 \
  --max-chunks 64 --log-every 1 \
  --outdir runs/p7_race3_smoke

for arm in control resmlp swiglu; do
  EXTRA="--width 3072 --depth 3"
  TYPE="mob"
  if [ "$arm" = "resmlp" ]; then TYPE="resmlp"; EXTRA="--width 3072 --depth 5"; fi
  if [ "$arm" = "swiglu" ]; then TYPE="swiglu"; EXTRA="--hidden 7300 --blocks 1"; fi
  $PY -m eval.p3_eval --data-root . --model-dir models/SmolLM2-1.7B \
    --head runs/p7_race3_smoke/head_${arm}_final.pt --head-type $TYPE --k 128 $EXTRA \
    --codebook codebook_smollm2.npz --token-cache data/fineweb_smollm2_dev_tokens.npy \
    --n-chunks 2 --batch 4 --backbone-dtype float16 --outdir runs/p7_race3_smoke_eval_${arm}
done

echo RACE3_SMOKE_DONE
